# inbox-cleaner/train.py
import argparse
import logging
import os
import time

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from common.config import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    MAX_LENGTH,
    METRIC_NAMES,
    MODEL_NAME,
    RUBERT_METRICS_CSV,
    RUBERT_PLOT,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)
from common.logging_setup import get_logger, setup_logging
from common.metrics import compute_all, find_best_threshold, ordered_row
from common.plots import plot_test_comparison
from model.data.dataset import SpamDataset
from model.spam_classifier import SpamClassifier
from model.tokenization.tokenizer import TextTokenizer

logger = get_logger(__name__)
# INFO видят только логгеры пакета model, а `python -m model.train` выполняет
# файл как __main__, и этот логгер пакетной настройке не подчиняется.
logger.setLevel(logging.INFO)

# === Настройки ===
# Рядом стоят измеренные на RTX 3060 Ti цифры, потому что без них не видно,
# откуда взялось время: модель даёт ~28k токенов/с при fp16 и не зависит от
# размера батча (упёрлась в вычисления, а не в память). Значит время обучения
# определяется произведением «эпохи × длина письма», и единственный способ
# уложиться в минуты — считать меньше токенов, а не подбирать батч.
CONFIG = {
    "cache_dir": CACHE_DIR,
    "checkpoint_dir": CHECKPOINT_DIR,
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    # 16, а не 32: при 512 токенах батч 32 занимает ~8.5 ГБ, а на карте 8 ГБ
    # это ровно на грани OOM. 16 укладывается в 4.6 ГБ.
    "batch_size": 16,
    # 10 эпох на 512 токенах — это 23 минуты. Ранняя остановка с patience=2
    # всё равно срабатывает раньше, а 4 эпохи хватает: на 5870 примерах
    # RuBERT выходит на плато за 3-4 прохода.
    "epochs": 4,
    "lr_encoder": 2e-5,
    "lr_head": 1e-3,
    "weight_decay": 0.01,
    "dropout": 0.4,
    "patience": 2,
    "grad_clip": 1.0,
    "use_amp": True,          # mixed precision — только для CUDA
    "num_workers": 0,         # на Windows с precompute=True лучше 0
}


def parse_args():
    """Разбирает ключи запуска.

    Единственный ключ — allow-cpu. Обучение RuBERT на CPU идёт часами (замер:
    ~25-30 минут на эпоху против 28 секунд на RTX 3060 Ti), и раньше оно
    стартовало молча, потому что uv.lock тянул CPU-колесо torch: человек
    ждал результата 3.5 часа и получал те же цифры, что за 9 минут на GPU.
    Теперь это ошибка с понятным текстом, а не вопросительный знак.
    """
    parser = argparse.ArgumentParser(description="Fine-tuning RuBERT на почтовом спаме")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="разрешить обучение на CPU (медленно: часы, а не минуты)",
    )
    return parser.parse_args()


def get_device():
    """Выбирает устройство и пишет диагностику в лог."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # TF32 — обмен matmul на 10-битную мантиссу. На Ampere это заметно
        # быстрее float32 и для fine-tuning достаточно точно; на Ampere и новее
        # mantissa 10 бит — нативный формат, терять нечего.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
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
        logger.warning("   Это будет ОЧЕНЬ медленно для RuBERT.")
    return device


def save_checkpoint(model, path):
    """Сохраняет веса в fp16.

    Fine-tuning идёт в смешанной точности, поэтому сохранять fp32-веса — это
    678 МБ на каждую улучшенную эпоху ради битов, которые в fp32 уже пусты.
    В fp16 чекпоинт 339 МБ. load_state_dict сам приводит тип к параметрам
    модели, поэтому сервис читает такой файл без изменений.
    """
    state = {name: tensor.half() for name, tensor in model.state_dict().items()}
    torch.save(state, path)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, use_amp=False):
    """Собирает предсказания и вероятности на выборке.

    Метрики здесь намеренно не считаются: их считает common.metrics, ровно тем
    же кодом, что и для baseline. Раньше evaluate звал sklearn напрямую, и
    accuracy в ней просто не было — в rubert_metrics.csv попадали три метрики
    против четырёх у baseline, графики различались по длине, а compare.py на
    настоящем CSV RuBERT падал с «в файле нет метрик: accuracy».

    threshold нужен только чтобы вернуть preds; рабочий порог подбирается
    позже по probs на val.
    """
    model.eval()
    preds, probs, targets = [], [], []

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

        batch_probs = torch.sigmoid(logits.float()).cpu().numpy()  # float32 для стабильности
        probs.extend(batch_probs)
        preds.extend((batch_probs > threshold).tolist())
        targets.extend(labels.cpu().numpy())

    return {
        "preds": preds,
        "probs": probs,
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


def main(allow_cpu=False):
    torch.manual_seed(SEED)
    logger.info("Зерно случайных чисел: %d", SEED)

    device = get_device()
    if device.type == "cpu" and not allow_cpu:
        logger.error(
            "Обучение остановлено: CUDA недоступна. На CPU RuBERT идёт часами "
            "(~25-30 мин на эпоху вместо 28 с на GPU). Поставь GPU-колесо torch "
            "командой `uv sync --extra gpu --dev` и повтори, либо запусти с "
            "--allow-cpu, если CPU-обучение действительно нужно.",
        )
        raise SystemExit(2)

    # === 1. Токенизатор ===
    logger.info("[1/6] Загрузка токенизатора...")
    tok = TextTokenizer(
        model_name=CONFIG["model_name"],
        max_length=CONFIG["max_length"],
    )

    # === 2. Датасеты с кэшем ===
    logger.info("[2/6] Подготовка датасетов...")
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    # Имя кэша зависит от параметров — чтобы не перемешать конфиги.
    # Смена max_length автоматически даёт другое имя, поэтому старый кэш
    # другой длины просто не подхватывается.
    tag = f"{CONFIG['model_name'].replace('/', '_')}_{CONFIG['max_length']}"

    train_ds = SpamDataset(
        TRAIN_CSV,
        tokenizer=tok,
        precompute=True,
        cache_path=f"{CONFIG['cache_dir']}/train_{tag}.pt",
    )
    val_ds = SpamDataset(
        VAL_CSV,
        tokenizer=tok,
        precompute=True,
        cache_path=f"{CONFIG['cache_dir']}/val_{tag}.pt",
    )

    logger.info("  train: %d примеров", len(train_ds))
    logger.info("  val:   %d примеров", len(val_ds))
    logger.info(
        "  max_length=%d, batch_size=%d, эпох до %d",
        CONFIG["max_length"],
        CONFIG["batch_size"],
        CONFIG["epochs"],
    )

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
        # 0.5 — только чтобы выбрать лучшую эпоху; рабочий порог
        # подбирается на val отдельно, после обучения.
        val_pred = evaluate(model, val_loader, device, 0.5, use_amp)
        epoch_metrics = compute_all(val_pred["preds"], val_pred["targets"])

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
            epoch_metrics["f1"],
            epoch_metrics["precision"],
            epoch_metrics["recall"],
            epoch_time,
            f"{gpu_mem:.2f}GB" if gpu_mem is not None else "n/a",
        )

        if epoch_metrics["f1"] > best_f1:
            best_f1 = epoch_metrics["f1"]
            save_checkpoint(model, best_path)
            no_improve = 0
            logger.info("сохранён лучший чекпоинт (f1=%.4f)", best_f1)
        else:
            no_improve += 1
            logger.info("no improvement (%d/%d)", no_improve, CONFIG["patience"])
            if no_improve >= CONFIG["patience"]:
                logger.info("Ранняя остановка.")
                break

    total_time = time.time() - start_time
    logger.info("Общее время обучения: %.1f мин", total_time / 60)

    # === 6. Порог по val и финальная оценка на test ===
    logger.info("[6/6] Подбор порога на val и оценка на test...")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    logger.info("Загружен лучший чекпоинт (val_f1=%.4f)", best_f1)

    # Рабочий порог выбирается на val, а метрики на test считаются уже при
    # нём. Сам порог попадает в CSV: по цифрам должно быть видно, при каком
    # значении они получены.
    val_pred = evaluate(model, val_loader, device, 0.5, use_amp)
    threshold, _ = find_best_threshold(val_pred["probs"], val_pred["targets"])
    logger.info("порог по f1 на val: %.2f", threshold)

    test_ds = SpamDataset(
        TEST_CSV,
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

    test_pred = evaluate(model, test_loader, device, threshold, use_amp)
    test_metrics = compute_all(test_pred["preds"], test_pred["targets"])

    logger.info(
        " Test \n%s",
        classification_report(
            test_pred["targets"],
            test_pred["preds"],
            target_names=["ham", "spam"],
            digits=4,
        ),
    )

    save_artifacts(test_metrics, threshold)

    logger.info("Лучший чекпоинт: %s", best_path)


def save_artifacts(metrics, threshold, csv_path=RUBERT_METRICS_CSV, plot_path=RUBERT_PLOT):
    """Кладёт метрики RuBERT в CSV и график в PNG — в том же формате, что и у
    baseline, иначе scripts/compare.py их не прочитает.

    metrics приходит из common.metrics.compute_all, то есть в нём уже есть и
    доли, и счётчики tp/fp/tn/fn. Формат нужен один и тот же у обеих моделей.

    Набор метрик на графике задаётся METRIC_NAMES, а не тем, что нашлось в
    словаре: раньше отсутствующий accuracy молча укорачивал график RuBERT, и
    два графика одной модели оказывались разной длины.
    """
    row = ordered_row(metrics, threshold)

    pd.DataFrame([row]).to_csv(csv_path, index=False)
    plot_test_comparison(
        {name: row[name] for name in METRIC_NAMES},
        title="RuBERT-base-cased (fine-tune)",
        save_path=plot_path,
    )
    logger.info("метрики сохранены в %s, график — %s", csv_path, plot_path)


if __name__ == "__main__":
    setup_logging()
    main(allow_cpu=parse_args().allow_cpu)
