# model/data/dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from model.preprocessing.clean import clean_text
from model.tokenization.tokenizer import TextTokenizer

LABEL_MAP = {"ham": 0.0, "spam": 1.0}

class SpamDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: TextTokenizer,
                 precompute: bool = True, cache_path: str | None = None):
        #Если кэш есть — загружаем и выходим
        if cache_path and os.path.exists(cache_path):
            print(f"загружаю кэш: {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            self.input_ids = cached["input_ids"]
            self.attention_mask = cached["attention_mask"]
            self.labels = cached["labels"]
            return

        # === Обрабатываем с нуля ===
        df = pd.read_csv(csv_path).fillna("")
        texts = [clean_text(str(t)) for t in df["text"]]

        mapped = df["target"].str.strip().str.lower().map(LABEL_MAP)
        if mapped.isna().any():
            bad = df.loc[mapped.isna(), "target"].unique()
            raise ValueError(f"Неизвестные метки: {bad}")
        labels = mapped.values.astype("float32")

        if precompute:
            enc = tokenizer.batch(texts)
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
        else:
            self.texts = texts
            self.tokenizer = tokenizer

        self.labels = torch.tensor(labels, dtype=torch.float32)

        # Сохраняем кэш
        if cache_path and precompute:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
                "labels": self.labels,
            }, cache_path)
            print(f"сохранил кэш: {cache_path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        if hasattr(self, "input_ids"):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }
        else:
            enc = self.tokenizer(self.texts[idx])
            return {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": self.labels[idx],
            }