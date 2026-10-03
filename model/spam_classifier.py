# model/spam_classifier.py
import torch.nn as nn
from transformers import AutoModel


class SpamClassifier(nn.Module):
    """Классификатор писем: энкодер RuBERT плюс один линейный слой на [CLS].

    encoder задаётся явно только в тестах и CI: настоящие веса rubert занимают
    ~700 МБ и тянутся из сети, а нужен модуль нужной формы. Без этого аргумента
    энкодер грузится из HuggingFace как обычно.
    """

    def __init__(self, model_name="DeepPavlov/rubert-base-cased", dropout=0.3, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size   # 768

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]       # [CLS]-токен
        return self.classifier(self.dropout(cls)).squeeze(-1)