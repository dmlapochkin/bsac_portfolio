"""Скрипт для инициализации таблиц: securities, indices, risk_profiles"""

import pandas as pd

from app.modules.config import DSN

import src.bsac_portfolio.database.db_manager as db_manager
import src.bsac_portfolio.market_client.market_client as market_client

db = db_manager.DatabaseManager(dsn=DSN)
client = market_client.MarketDataClient()

print("Загрузка списка тикеров...")
tickers_df = client.fetch_ticker_list(fetch_sectors=True)
if not tickers_df.empty:
    print(f"Загружено {len(tickers_df)} тикеров. Обновление в БД...")
    db.upsert_securities(tickers_df)
else:
    print("Не удалось загрузить список тикеров.")

print("Загрузка списка индексов...")
indices_df = client.fetch_indices_list()
if not indices_df.empty:
    print(f"Загружено {len(indices_df)} индексов. Обновление в БД...")
    db.upsert_indices(indices_df)
else:
    print("Не удалось загрузить список индексов.")

print("Вставка профилей риска...")
risk_profiles_data = [
    ('Агрессивный', 'Модель терпима к просадкам и быстро адаптирует структуру портфеля под рыночные сигналы.\nПодходит инвесторам, готовым к высокой волатильности ради потенциально высокой прибыли.', True, 15, 0.06, 0.06, 'bolt', 'aggressive_v1', 0.8),
    ('Умеренный', 'Модель готова к умеренным просадкам ради роста капитала, но избегает излишней волатильности.\nПодходит инвесторам, стремящимся к росту капитала при контролируемом уровне риска.', True, 20, 0.06, 0.08, 'balance', 'moderate_v1', 0.86),
    ('Консервативный', 'Модель чувствительна к убыткам и предпочитает стабильность высокой доходности.\nПодходит инвесторам, для которых защита средств важнее потенциальной прибыли.', True, 25, 0.06, 0.10, 'shield', 'conservative_v1', 0.91)
]
cols = ['display_name', 'description', 'is_active', 'rebalance_policy_planned', 'rebalance_policy_benchmark', 'rebalance_policy_weights', 'icon', 'model_name', 'eta']
risk_df = pd.DataFrame(risk_profiles_data, columns=cols)
db.upsert_risk_profiles(risk_df)

print("Готово!")