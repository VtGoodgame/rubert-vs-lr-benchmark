"""Обучение модели на данных из data/ и сохранение артефакта."""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from model.config import DATA_DIR, settings


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def load_split(file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / file_name)


def train() -> Pipeline:
    train_df = load_split(settings.train_file)
    pipeline = build_pipeline()
    pipeline.fit(train_df[settings.text_column], train_df[settings.target_column])
    return pipeline


def main() -> None:
    pipeline = train()

    test_df = load_split(settings.test_file)
    predictions = pipeline.predict(test_df[settings.text_column])
    print(classification_report(test_df[settings.target_column], predictions))

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.model_path)
    print(f"Модель сохранена: {settings.model_path}")


if __name__ == "__main__":
    main()
