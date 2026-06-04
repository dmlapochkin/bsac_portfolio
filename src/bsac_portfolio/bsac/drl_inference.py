"""Инференс для DRL модели портфеля."""
from datetime import date, timedelta

import pandas as pd

from bsac_portfolio.bsac.agent import BSACAgent
from bsac_portfolio.bsac.extractor import BSACLSTMFeatureExtractor
from bsac_portfolio.data_preparation.feature_engineer import FeatureEngineer
from bsac_portfolio.data_preparation.lstm_feature_preparator import LSTMFeaturePreparator
from bsac_portfolio.database.db_manager import DatabaseManager


def predict_weights(agent: BSACAgent, db: DatabaseManager, tickers: list[str], model_dir: str, feature_cols: list[str], window_size: int = 60, reference_date: date = None) -> dict[str, float]:
    """
    Рассчитывает текущие веса портфеля на основе модели.

    Args:
        agent: Экземпляр агента BSAC для инференса
        db: Экземпляр DatabaseManager
        tickers: Список тикеров активов
        model_dir: Путь к директории модели
        feature_cols: Список колонок с признаками
        window_size: Размер окна временного ряда
        reference_date: Дата для расчета (по умолчанию - сегодня)

    Returns:
        Словарь с весами портфеля для каждого актива.
    """
    ref = reference_date or date.today()
    end_date = ref.strftime('%Y-%m-%d')
    start_date = (ref - timedelta(days=window_size + 150)).strftime('%Y-%m-%d')
    
    df_raw = db.get_market_data(tickers, start_date, end_date)
    if df_raw.empty:
        raise ValueError("Нет рыночных данных для расчета")

    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.sort_values(['ticker', 'date']).reset_index(drop=True)

    engineer = FeatureEngineer(df_raw)
    df_features = engineer.calculate_indicators(drop_na=False)

    if 'ret' not in df_features.columns:
        df_features['ret'] = df_features.groupby('ticker')['close'].pct_change()

    preparator = LSTMFeaturePreparator(df_features, window_size=window_size)
    wide_data = preparator.prepare_wide_dataset({'inference': df_features}, feature_cols)
    df_drl_input = wide_data['inference']

    if df_drl_input.empty or len(df_drl_input) < window_size + 2:
        raise ValueError("Недостаточно исторических данных для инициализации окна наблюдений")

    if agent is None:
        agent = BSACAgent(
            model_dir=model_dir, env_config=None, risk_profile=None, training_cfg=None, policy_cfg=None,
            feature_extractor_class=BSACLSTMFeatureExtractor, device='cpu'
        )
        agent.load(filename='best_model.zip')

    trajectory = agent.predict(df_drl_input)
    model_tickers = agent.env_config.tickers
    last_weights = trajectory['weights'][-1]
    weights_dict = {t: float(last_weights[i]) for i, t in enumerate(model_tickers)}

    total_weight = sum(weights_dict.values())
    if total_weight > 0:
        weights_dict = {k: v / total_weight for k, v in weights_dict.items()}

    return weights_dict