# scripts/error_analysis.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model.tokenization.tokenizer import TextTokenizer
from model.data.dataset import SpamDataset
from model.spam_classifier import SpamClassifier
from model.preprocessing.clean import clean_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = TextTokenizer(max_length=128)
model = SpamClassifier().to(device)
model.load_state_dict(torch.load("model/checkpoints/best.pt", weights_only=True))
model.eval()

test_ds = SpamDataset("model/data/test.csv", tokenizer=tok)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Собираем предсказания
all_probs, all_preds, all_labels, all_texts = [], [], [], []
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

print(f"False Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")

# Сохраняем для просмотра
fp.head(30).to_csv("error_analysis_fp.csv", index=False)
fn.head(30).to_csv("error_analysis_fn.csv", index=False)

# Смотрим примеры
print("\n=== Примеры FP (модель сказала spam, а это ham) ===")
for _, row in fp.head(5).iterrows():
    print(f"[{row['prob']:.3f}] {str(row['text'])[:200]}...")

print("\n=== Примеры FN (модель сказала ham, а это spam) ===")
for _, row in fn.head(5).iterrows():
    print(f"[{row['prob']:.3f}] {str(row['text'])[:200]}...")