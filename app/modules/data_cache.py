"""Модуль кэширования и работы с данными."""

import pandas as pd
import streamlit as st

from bsac_portfolio.bsac import BSACAgent, BSACLSTMFeatureExtractor
from bsac_portfolio.data_preparation.feature_engineer import FeatureEngineer
from bsac_portfolio.database.db_manager import DatabaseManager

db = None
profiles_df = None


def init_data_cache(dsn: str) -> tuple[DatabaseManager, pd.DataFrame]:
    """
    Инициализирует глобальные объекты подключения к БД и кэшированных данных.
    
    Args:
        dsn: Строка подключения к базе данных

    Returns:
        Кортеж из экземпляра DatabaseManager и DataFrame с профилями риска
    """
    global db, profiles_df
    if db is None:
        db = get_db(dsn)
        profiles_df = get_active_risk_profiles(db).sort_values(by="rebalance_policy_planned")
    return db, profiles_df


@st.cache_resource
def get_db(dsn: str) -> DatabaseManager:
    """
    Создаёт и кэширует экземпляр DatabaseManager для подключения к БД.
    
    Args:
        dsn: Строка подключения к базе данных

    Returns:
        Экземпляр DatabaseManager
    """
    return DatabaseManager(dsn=dsn)


@st.cache_resource(show_spinner=False)
def _get_cached_agent(model_dir: str) -> BSACAgent:
    """
    Загружает и кэширует агент DRL модели.

    Args:
        model_dir: Директория модели

    Returns:
        Экземпляр BSACAgent
    """
    agent = BSACAgent(model_dir=model_dir, feature_extractor_class=BSACLSTMFeatureExtractor, device='auto')
    agent.load(filename='best_model.zip')
    return agent


@st.cache_data(ttl=3600)
def get_active_risk_profiles(_db: DatabaseManager) -> pd.DataFrame:
    """
    Получает список активных профилей риска из базы данных.

    Args:
        _db: Экземпляр DatabaseManager

    Returns:
        DataFrame с активными профилями риска
    """
    return _db.get_active_risk_profiles()


@st.cache_data(ttl=3600)
def get_available_tickers(_db: DatabaseManager) -> pd.DataFrame:
    """
    Получает список доступных тикеров из базы данных.

    Args:
        _db: Экземпляр DatabaseManager

    Returns:
        DataFrame с доступными тикерами
    """
    return _db.get_available_tickers()


@st.cache_data(ttl=1800)
def get_market_data(_db: DatabaseManager, tickers: list, start: str, end: str, warmup_days: int = 0) -> pd.DataFrame:
    """
    Загружает рыночные данные по указанным тикерам за период.

    Args:
        _db: Экземпляр DatabaseManager
        tickers: Список тикеров
        start: Начальная дата
        end: Конечная дата
        warmup_days: Количество дней для разминки

    Returns:
        DataFrame с рыночными данными
    """
    df = _db.get_market_data(tickers, start, end, warmup_days=warmup_days)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'value']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    return df


@st.cache_data(ttl=3600)
def calculate_indicators(df_raw: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Вычисляет технические индикаторы на основе данных.

    Args:
        df_raw: DataFrame с рыночными данными
        **kwargs: Аргументы для вычисления индикаторов

    Returns:
        DataFrame с вычисленными индикаторами
    """
    if df_raw.empty:
        return df_raw
    engineer = FeatureEngineer(df_raw)
    return engineer.calculate_indicators(drop_na=False, **kwargs)


@st.cache_data(ttl=3600)
def get_security_info(_db: DatabaseManager, ticker: str) -> pd.DataFrame:
    """
    Получает информацию о ценной бумаге по тикеру.

    Args:
        _db: Экземпляр DatabaseManager
        ticker: Тикер ценной бумаги

    Returns:
        DataFrame с информацией о ценной бумаге
    """
    return _db.get_security_info(ticker)
