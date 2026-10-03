# inbox-cleaner/train.py
import logging
import os
import time

import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from common.logging_setup import get_logger, setup_logging
from model.data.dataset import SpamDataset
from model.spam_classifier import SpamClassifier
from model.tokenization.tokenizer import TextTokenizer

logger = get_logger(__name__)
# INFO видят только логгеры пакета model, а `python -m model.train` выполняет
# файл как __main__, и этот логгер пакетной настройке не подчиняется.
logger.setLevel(logging.INFO)

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
    """Выбирает устройство и пишет диагностику в лог."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA доступна: %s", torch.cuda.get_device_name(0))
        logger.info("   CUDA version: %s", torch.version.cuda)
        logger.info(
            "   GPU memory:   %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
        logger.info("   PyTorch:      %s", torch.__version__)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA недоступна, обучение на CPU")
        logger.info("   PyTorch: %s", torch.__version__)
        logger.warning("   Это будет ОЧЕНЬ медленно для RuBERT. Рассмотри Google Colab.")
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
    logger.info("[1/6] Загрузка токенизатора...")
    tok = TextTokenizer(
        model_name=CONFIG["model_name"],
        max_length=CONFIG["max_length"],
    )

    # === 2. Датасеты с кэшем ===
    logger.info("[2/6] Подготовка датасетов...")
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

    logger.info("  train: %d примеров", len(train_ds))
    logger.info("  val:   %d примеров", len(val_ds))

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
    logger.info("[3/6] Создание модели...")
    model = SpamClassifier(
        model_name=CONFIG["model_name"],
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Всего параметров:      %s", f"{n_params:,}")
    logger.info("  Обучаемых параметров:  %s", f"{n_trainable:,}")

    # === 4. Оптимизатор с двумя lr + AMP scaler ===
    logger.info("[4/6] Настройка оптимизатора...")
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": CONFIG["lr_encoder"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["lr_head"]},
    ], weight_decay=CONFIG["weight_decay"])

    criterion = torch.nn.BCEWithLogitsLoss()

    # GradScaler нужен только для CUDA + AMP
    use_amp = CONFIG["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info("  Mixed precision (AMP): %s", "включён" if use_amp else "выключен")

    # === 5. Обучение ===
    logger.info("[5/6] Обучение...")
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
        gpu_mem = None
        if device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

        logger.info(
            "epoch=%d loss=%.4f val_f1=%.4f prec=%.4f rec=%.4f time=%.1fs gpu_mem=%s",
            epoch,
            train_loss,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            epoch_time,
            f"{gpu_mem:.2f}GB" if gpu_mem is not None else "n/a",
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), best_path)
            no_improve = 0
            logger.info("saved best (f1=%.4f)", best_f1)
        else:
            no_improve += 1
            logger.info("no improvement (%d/%d)", no_improve, CONFIG["patience"])
            if no_improve >= CONFIG["patience"]:
                logger.info("Ранняя остановка.")
                break

    total_time = time.time() - start_time
    logger.info("Общее время обучения: %.1f мин", total_time / 60)

    # === 6. Финальная оценка на test ===
    logger.info("[6/6] Финальная оценка на test...")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    logger.info("Загружен лучший чекпоинт (val_f1=%.4f)", best_f1)

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

    logger.info(
        " Test \n%s",
        classification_report(
            test_metrics["targets"],
            test_metrics["preds"],
            target_names=["ham", "spam"],
            digits=4,
        ),
    )

    logger.info("Лучший чекпоинт: %s", best_path)


if __name__ == "__main__":
    setup_logging()
    main()
