"""Streamlit-интерфейс: streamlit run interface/app.py."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="inbox-cleaner", layout="center")
st.title("inbox-cleaner")

texts = st.text_area("Тексты писем", height=200).splitlines()

if st.button("Классифицировать") and [t for t in texts if t.strip()]:
    payload = {"texts": [t.strip() for t in texts if t.strip()]}
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Сервис недоступен: {exc}")
    else:
        for text, label in zip(payload["texts"], response.json()["labels"], strict=True):
            st.write(f"**{label}** — {text}")
