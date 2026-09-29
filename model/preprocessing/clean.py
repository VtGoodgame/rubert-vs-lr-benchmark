# clean.py
import re


def clean_text(text):
    """Функция для очистки текста от лишних символов и ссылок."""
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text) # Заменяем ссылки на токен [URL]
    text = re.sub(r'[\"\*]', '', text) # Удаляем лишние кавычки и звездочки
    text = re.sub(r'\s+', ' ', text).strip() # Удаляем лишние пробелы и переносы строк
    return text
