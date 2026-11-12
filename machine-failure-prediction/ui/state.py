"""Tải mô hình, thông tin huấn luyện và dữ liệu tham chiếu (có cache)."""
import json

import streamlit as st

from src.config import INFO_PATH, MODEL_NAMES, MODEL_SLUGS, MODELS_DIR
from src.predict import load_bundle
from src.preprocess import RAW_NUMERIC, load_data


@st.cache_resource(show_spinner=False)
def get_bundle(name: str) -> dict:
    return load_bundle(name)


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_data()


@st.cache_data(show_spinner=False)
def get_reference() -> dict:
    """Giá trị tham chiếu (trung vị) dùng để giải thích dự đoán."""
    return {c: float(get_dataset()[c].median()) for c in RAW_NUMERIC}


def models_ready() -> bool:
    return all((MODELS_DIR / f"{MODEL_SLUGS[n]}.joblib").exists() for n in MODEL_NAMES)


def get_info():
    return json.loads(INFO_PATH.read_text(encoding="utf-8")) if INFO_PATH.exists() else None
