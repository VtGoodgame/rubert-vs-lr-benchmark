# scripts/error_analysis.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.config import CHECKPOINT_PATH, MAX_LENGTH, MODEL_NAME, SPAM_THRESHOLD, TEST_CSV
from common.logging_setup import get_logger, setup_logging
from model.data.dataset import SpamDataset
from model.spam_classifier import SpamClassifier
from model.tokenization.tokenizer import TextTokenizer

logger = get_logger(__name__)

# Кодировку вывода и повторный вызов setup_logging больше не нужны:
# и то и другое делает common.logging_setup.
setup_logging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = TextTokenizer(model_name=MODEL_NAME, max_length=MAX_LENGTH)
model = SpamClassifier(model_name=MODEL_NAME).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
model.eval()

test_ds = SpamDataset(TEST_CSV, tokenizer=tok)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Собираем предсказания
all_probs, all_preds, all_labels = [], [], []
df = pd.read_csv(TEST_CSV).fillna("")

with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        # Порог из конфига, а не 0.5: иначе примеры FP/FN не соответствовали
        # бы метрикам, посчитанным в бенчмарке.
        all_preds.extend((probs > SPAM_THRESHOLD).astype(int))
        all_labels.extend(batch["labels"].numpy())

# FP и FN
df["prob"] = all_probs
df["pred"] = all_preds
df["true"] = all_labels

fp = df[(df["pred"] == 1) & (df["true"] == 0)]
fn = df[(df["pred"] == 0) & (df["true"] == 1)]

logger.info("False Positives: %d", len(fp))
logger.info("False Negatives: %d", len(fn))

# Сохраняем для просмотра
fp.head(30).to_csv("error_analysis_fp.csv", index=False)
fn.head(30).to_csv("error_analysis_fn.csv", index=False)
logger.info("первые 30 FP и FN выгружены в error_analysis_fp.csv и error_analysis_fn.csv")

# Текст писем в лог не пишется: консоль Windows в cp1251, а письма содержат
# символы вне кодировки. Смотреть примеры нужно в выгруженных CSV.