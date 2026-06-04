"""Модуль конфигурации и утилит приложения."""

import json
import os
from datetime import date
from pathlib import Path
import sys

import toml

CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config.toml"))
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        APP_CONFIG = toml.load(f)
else:
    APP_CONFIG = {}

db_cfg = APP_CONFIG.get("database", {})
DSN = f"postgresql://{db_cfg.get('user', '')}:{db_cfg.get('password', '')}@{db_cfg.get('host', 'localhost')}:{db_cfg.get('port', 5432)}/{db_cfg.get('database', '')}"

app_cfg = APP_CONFIG.get("application", {})
APP_MODE = app_cfg.get("mode", "live")
APP_DEMO_DATE = app_cfg.get("demo_date")
MIN_DATE = date.fromisoformat(app_cfg.get("min_date", "2000-01-01"))

PROJECT_ROOT = os.path.dirname(os.path.abspath(CONFIG_PATH))
path_cfg = APP_CONFIG.get("paths", {})
MODEL_BASE_DIR = os.path.join(PROJECT_ROOT, path_cfg.get("model_base_dir", "models/"))

param_cfg = APP_CONFIG.get("parameters", {})
COMMISSION_RATE = param_cfg.get("commission", 0.002)
RISKFREE_RATE = param_cfg.get("riskfree_rate", 0.0002)
MIN_POSITION_WEIGHT = param_cfg.get("min_position_weight", 0.005)


def load_model_metadata(model_dir: str) -> dict:
    """
    Загружает метаданные модели из JSON файла.
    
    Args:
        model_dir: Директория модели

    Returns:
        Словарь с метаданными модели. Если файл не найден, возвращает пустой словарь.
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_money(value: float, decimals: int = 2) -> str:
    """
    Форматирует числовое значение в денежный формат.
    
    Args:
        value: Числовое значение для форматирования
        decimals: Количество знаков после запятой

    Returns:
        Строка с отформатированным значением
    """
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", " ")
