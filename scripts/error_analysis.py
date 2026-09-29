# scripts/error_analysis.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.logging_setup import get_logger, setup_logging
from model.data.dataset import SpamDataset
from model.spam_classifier import SpamClassifier
from model.tokenization.tokenizer import TextTokenizer

logger = get_logger(__name__)

# Кодировку вывода и повторный вызов setup_logging больше не нужны:
# и то и другое делает common.logging_setup.
setup_logging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = TextTokenizer(max_length=128)
model = SpamClassifier().to(device)
model.load_state_dict(torch.load("model/checkpoints/best.pt", weights_only=True))
model.eval()

test_ds = SpamDataset("model/data/test.csv", tokenizer=tok)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Собираем предсказания
all_probs, all_preds, all_labels = [], [], []
df = pd.read_csv("model/data/test.csv").fillna("")

with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend((probs > 0.5).astype(int))
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

# Смотрим примеры
logger.info("=== Примеры FP (модель сказала spam, а это ham) ===")
for _, row in fp.head(5).iterrows():
    logger.info("[%.3f] %s...", row["prob"], str(row["text"])[:200])

logger.info("=== Примеры FN (модель сказала ham, а это spam) ===")
for _, row in fn.head(5).iterrows():
    logger.info("[%.3f] %s...", row["prob"], str(row["text"])[:200])