"""Подготовка данных для обучения агента BSAC."""
import numpy as np
import pandas as pd

from bsac_portfolio.data_preparation.feature_engineer import FeatureEngineer


class LSTMFeaturePreparator:
    """
    Класс для подготовки данных в wide-формате.
    """
    def __init__(self, raw_df: pd.DataFrame, window_size: int = 60) -> None:
        """
        Инициализация подготовщика признаков.
        
        Args:
            raw_df: DataFrame с рыночными данными.
            window_size: Размер окна для временного ряда.
        """
        self.raw_df = raw_df
        self.window_size = window_size
        self.feature_cols: list[str] = []

    def prepare_splits(self, train_end: str, val_start: str, val_end: str, test_start: str, test_end: str) -> dict[str, pd.DataFrame]:
        """
        Подготавливает сплиты данных для обучения, валидации и теста.
        
        Args:
            train_end: Дата окончания обучающей выборки.
            val_start: Дата начала валидационной выборки.
            val_end: Дата окончания валидационной выборки.
            test_start: Дата начала тестовой выборки.
            test_end: Дата окончания тестовой выборки.
            
        Returns:
            Словарь со сплитами данных (train/val/test).
        """
        fe = FeatureEngineer(self.raw_df)
        
        full_df = fe.calculate_indicators(
            sma_periods=[10, 20, 50],
            ema_fast=12, ema_slow=26, macd_signal=9,
            bb_period=20, bb_std=2.0,
            atr_period=14, rsi_period=14, stoch_period=14, stoch_k_period=3,
            var_period=20, var_quantile=0.05, volatility_period=20,
            drop_na=False 
        )

        min_valid = 50
        full_df = full_df.groupby('ticker').apply(
            lambda x: x.iloc[min_valid:] if len(x) > min_valid else x
        ).reset_index(drop=True)

        exclude_cols = [
            'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'value', 
            'return', 'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d', 'obv'
        ]
        
        self.feature_cols = [c for c in full_df.columns if c not in exclude_cols]

        splits_config = {
            'train': (None, train_end, True),
            'val': (val_start, val_end, False),
            'test': (test_start, test_end, False)
        }

        result_dfs: dict[str, pd.DataFrame] = {}
        metadata: dict = {}

        for name, (start, end, is_first) in splits_config.items():
            chunks = []
            for ticker, group in full_df.groupby('ticker'):
                group = group.sort_values('date')
                
                mask_main = group['date'].between(
                    start if start else group['date'].min(), 
                    end
                )
                main_data = group[mask_main]

                if not is_first and len(main_data) > 0:
                    history_mask = group['date'] < start
                    history = group[history_mask].sort_values('date').tail(self.window_size)
                    chunk = pd.concat([history, main_data])
                else:
                    chunk = main_data
                
                if not chunk.empty:
                    chunks.append(chunk)

            if not chunks:
                continue

            split_df = pd.concat(chunks).sort_values(['ticker', 'date']).reset_index(drop=True)

            result_dfs[name] = split_df

            metadata[name] = {
                'start_date': str(split_df['date'].min()),
                'end_date': str(split_df['date'].max()),
                'main_period_start': str(start if start else split_df['date'].min()),
                'total_rows': len(split_df)
            }

        metadata['feature_cols'] = self.feature_cols
        metadata['window_size'] = self.window_size
        
        return result_dfs
    
    def prepare_wide_dataset(self, splits_result: dict[str, pd.DataFrame],  features: list[str]) -> dict[str, pd.DataFrame]:
        """
        Преобразование данных из long-формата в wide-format.
        
        Args:
            splits_result: Словарь сплитов в long-формате.
            features: Список признаков.
            
        Returns:
            Словарь сплитов в wide-формате.
        """
        if not splits_result:
            return {}

        first_df = next(iter(splits_result.values()))
        if 'ticker' not in first_df.columns:
            raise ValueError("Колонка 'ticker' отсутствует в предоставленных данных.")
        all_tickers = sorted(first_df['ticker'].unique())
        
        wide_dfs: dict[str, pd.DataFrame] = {}
        
        for split_name, df_long in splits_result.items():
            if df_long.empty:
                continue
                
            if 'date' in df_long.columns:
                df_long = df_long.copy()
                df_long['date'] = pd.to_datetime(df_long['date'])
                df_long = df_long.set_index('date').sort_index()
            elif not isinstance(df_long.index, pd.DatetimeIndex):
                raise ValueError(
                    f"Сплит '{split_name}' не содержит колонки 'date' "
                    "и индекс не является DatetimeIndex."
                )
                
            pivot_frames = []
            for feat in features:
                if feat not in df_long.columns:
                    continue
                    
                pivoted = df_long.pivot(columns='ticker', values=feat)
                pivoted = pivoted.reindex(columns=all_tickers, fill_value=np.nan)
                
                prefix = "ret_" if feat == 'return' else f"{feat}_"
                pivoted.columns = [f"{prefix}{c}" for c in pivoted.columns]
                pivot_frames.append(pivoted)
                
            if not pivot_frames:
                continue
                
            df_wide = pd.concat(pivot_frames, axis=1).sort_index()
            df_wide = df_wide.ffill().bfill()
            
            wide_dfs[split_name] = df_wide
            
        return wide_dfs