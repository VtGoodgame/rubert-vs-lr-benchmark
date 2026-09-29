from transformers import AutoTokenizer

class TextTokenizer:
    def __init__(self, model_name: str = "DeepPavlov/rubert-base-cased",
                 max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, text: str) -> dict:
        """Токенизирует одну строку."""
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def batch(self, texts: list[str]) -> dict:
        """Токенизирует батч строк — быстрее, чем по одной."""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size