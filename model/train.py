# inbox-cleaner/train.py
import os
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from model.tokenization.tokenizer import TextTokenizer
from model.data.dataset import SpamDataset
from model.spam_classifier import SpamClassifier


# === Настройки ===
CONFIG = {
    "train_csv": "model/data/train_split.csv",
    "val_csv": "model/data/val_split.csv",
    "test_csv": "model/data/test.csv",
    "cache_dir": "model/data/cache",
    "checkpoint_dir": "model/checkpoints",
    "model_name": "DeepPavlov/rubert-base-cased",
    "max_length": 128,
    "batch_size": 32,
    "epochs": 10,
    "lr_encoder": 2e-5,
    "lr_head": 1e-3,
    "weight_decay": 0.01,
    "dropout": 0.4,
    "patience": 2,
    "grad_clip": 1.0,
    "threshold": 0.7,
    "use_amp": True,          # mixed precision — только для CUDA
    "num_workers": 0,         # на Windows с precompute=True лучше 0
}


def get_device():
    """Выбирает устройство и печатает диагностику."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   PyTorch:      {torch.__version__}")
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA недоступна, обучение на CPU")
        print(f"   PyTorch: {torch.__version__}")
        print("   Это будет ОЧЕНЬ медленно для RuBERT. Рассмотри Google Colab.")
    return device


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, use_amp=False):
    """Считает метрики на val/test."""
    model.eval()
    preds, targets = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Mixed precision для инференса — тоже ускоряет
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)

        probs = torch.sigmoid(logits.float())   # вернуть во float32 для стабильности
        preds.extend((probs > threshold).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    return {
        "f1": f1_score(targets, preds),
        "precision": precision_score(targets, preds),
        "recall": recall_score(targets, preds),
        "preds": preds,
        "targets": targets,
    }


def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    grad_clip, use_amp):
    """Одна эпоха обучения с mixed precision."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision: forward + loss в float16
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            # Backward через scaler: масштабирует градиенты,
            # чтобы float16 не терял мелкие значения
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Обычный float32 путь
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = get_device()

    # === 1. Токенизатор ===
    print("\n[1/6] Загрузка токенизатора...")
    tok = TextTokenizer(
        model_name=CONFIG["model_name"],
        max_length=CONFIG["max_length"],
    )

    # === 2. Датасеты с кэшем ===
    print("[2/6] Подготовка датасетов...")
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    # Имя кэша зависит от параметров — чтобы не перемешать конфиги
    tag = f"{CONFIG['model_name'].replace('/', '_')}_{CONFIG['max_length']}"

    train_ds = SpamDataset(
        CONFIG["train_csv"],
        tokenizer=tok,
        precompute=True,
        cache_path=f"{CONFIG['cache_dir']}/train_{tag}.pt",
    )
    val_ds = SpamDataset(
        CONFIG["val_csv"],
        tokenizer=tok,
        precompute=True,
        cache_path=f"{CONFIG['cache_dir']}/val_{tag}.pt",
    )

    print(f"  train: {len(train_ds)} примеров")
    print(f"  val:   {len(val_ds)} примеров")

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),   # ускоряет передачу CPU→GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # === 3. Модель ===
    print("[3/6] Создание модели...")
    model = SpamClassifier(
        model_name=CONFIG["model_name"],
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Всего параметров:      {n_params:,}")
    print(f"  Обучаемых параметров:  {n_trainable:,}")

    # === 4. Оптимизатор с двумя lr + AMP scaler ===
    print("[4/6] Настройка оптимизатора...")
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": CONFIG["lr_encoder"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["lr_head"]},
    ], weight_decay=CONFIG["weight_decay"])

    criterion = torch.nn.BCEWithLogitsLoss()

    # GradScaler нужен только для CUDA + AMP
    use_amp = CONFIG["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  Mixed precision (AMP): {'включён' if use_amp else 'выключен'}")

    # === 5. Обучение ===
    print("[5/6] Обучение...")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    best_path = os.path.join(CONFIG["checkpoint_dir"], "best.pt")

    best_f1 = 0.0
    no_improve = 0
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, CONFIG["grad_clip"], use_amp,
        )
        metrics = evaluate(model, val_loader, device, CONFIG["threshold"], use_amp)

        epoch_time = time.time() - epoch_start

        # Память GPU, если CUDA
        mem_info = ""
        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1e9
            mem_info = f" | gpu_mem={mem:.2f}GB"
            torch.cuda.reset_peak_memory_stats()

        print(f"epoch {epoch:2d} | loss={train_loss:.4f} | "
              f"val_f1={metrics['f1']:.4f} | "
              f"prec={metrics['precision']:.4f} | "
              f"rec={metrics['recall']:.4f} | "
              f"time={epoch_time:.1f}s{mem_info}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), best_path)
            no_improve = 0
            print(f"          → saved best (f1={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"          → no improvement ({no_improve}/{CONFIG['patience']})")
            if no_improve >= CONFIG["patience"]:
                print("Early stopping.")
                break

    total_time = time.time() - start_time
    print(f"\nОбщее время обучения: {total_time/60:.1f} мин")

    # === 6. Финальная оценка на test ===
    print("\n[6/6] Финальная оценка на test...")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    print(f"Загружен лучший чекпоинт (val_f1={best_f1:.4f})")

    test_ds = SpamDataset(
        CONFIG["test_csv"],
        tokenizer=tok,
        precompute=True,
        cache_path=f"{CONFIG['cache_dir']}/test_{tag}.pt",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    test_metrics = evaluate(model, test_loader, device, CONFIG["threshold"], use_amp)

    print("\n=== Test ===")
    print(classification_report(
        test_metrics["targets"],
        test_metrics["preds"],
        target_names=["ham", "spam"],
        digits=4,
    ))

    print(f"Лучший чекпоинт: {best_path}")


if __name__ == "__main__":
    main()