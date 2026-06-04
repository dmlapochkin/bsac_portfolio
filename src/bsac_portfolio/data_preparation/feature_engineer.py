"""Инженерия признаков для рыночных данных."""
import os

import json
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Класс для расчёта технических индикаторов и признаков.
    """
    def __init__(self, market_data: pd.DataFrame) -> None:
        """
        Инициализация инженера признаков.
        
        Args:
            market_data: DataFrame с рыночными данными (OHLCV).
        """
        self.data = market_data.copy()
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value']
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                
        self.data = self.data.sort_values(['ticker', 'date']).reset_index(drop=True)

    def calculate_indicators(
        self,
        sma_periods: list[int] = None,
        ema_fast: int = 12,
        ema_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        rsi_period: int = 14,
        stoch_period: int = 14,
        stoch_k_period: int = 3,
        var_period: int = 60,
        var_quantile: float = 0.05,
        volatility_period: int = 20,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Рассчитывает технические индикаторы для всех тикеров.
        
        Args:
            sma_periods: Периоды для скользящих средних SMA.
            ema_fast: Быстрый период EMA для MACD.
            ema_slow: Медленный период EMA для MACD.
            macd_signal: Период сигнальной линии MACD.
            bb_period: Период полос Боллинджера.
            bb_std: Количество стандартных отклонений для Bollinger Bands.
            atr_period: Период ATR.
            rsi_period: Период RSI.
            stoch_period: Период стохастического осциллятора.
            stoch_k_period: Период сглаживания %K.
            var_period: Период для VaR/CVaR.
            var_quantile: Квантиль для VaR.
            volatility_period: Период для расчёта волатильности.
            drop_na: Удалять ли строки с NaN.
            
        Returns:
            DataFrame с рассчитанными индикаторами.
        """
        if sma_periods is None:
            sma_periods = [10, 20, 50]
            
        results = []
        for ticker, df in self.data.groupby('ticker'):
            df = df.copy()
            
            df['return'] = df['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df[f'volatility_{volatility_period}'] = df['log_return'].rolling(volatility_period).std()

            for w in sma_periods:
                df[f'sma_{w}'] = df['close'].rolling(w).mean()
            
            ema_fast_series = df['close'].ewm(span=ema_fast).mean()
            ema_slow_series = df['close'].ewm(span=ema_slow).mean()
            df['macd'] = ema_fast_series - ema_slow_series
            df['macd_signal'] = df['macd'].ewm(span=macd_signal).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            sma_bb = df['close'].rolling(bb_period).mean()
            std_bb = df['close'].rolling(bb_period).std()
            df['bb_upper'] = sma_bb + bb_std * std_bb
            df['bb_middle'] = sma_bb
            df['bb_lower'] = sma_bb - bb_std * std_bb

            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            df[f'atr_{atr_period}'] = tr.rolling(atr_period).mean()

            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
            avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
            df[f'rsi_{rsi_period}'] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss).replace(0, np.nan)))

            low_stoch = df['low'].rolling(stoch_period).min()
            high_stoch = df['high'].rolling(stoch_period).max()
            df['stoch_k'] = 100.0 * (df['close'] - low_stoch) / (high_stoch - low_stoch).replace(0, np.nan)
            df['stoch_d'] = df['stoch_k'].rolling(stoch_k_period).mean()

            direction = np.sign(df['close'].diff().fillna(1))
            df['obv'] = (df['volume'] * direction).cumsum()

            returns = df['log_return']
            df[f'var_{int((1-var_quantile)*100)}'] = returns.rolling(var_period).quantile(var_quantile)
            df[f'cvar_{int((1-var_quantile)*100)}'] = returns.rolling(var_period).apply(
                lambda x: x[x <= x.quantile(var_quantile)].mean() if not x[x <= x.quantile(var_quantile)].empty else np.nan
            )

            results.append(df)

        self.data = pd.concat(results, ignore_index=True)
        if drop_na:
            self.data = self.data.dropna().reset_index(drop=True)
        else:
            self.data = self.data.reset_index(drop=True)
        return self.data

    def save_features(self, filepath: str) -> None:
        """
        Сохраняет признаки в CSV и метаданные в JSON.
        
        Args:
            filepath: Путь к файлу для сохранения CSV.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.data.to_csv(filepath, index=False, float_format='%.6f')
        meta_path = os.path.splitext(filepath)[0] + '_features.json'
        features_map = {
            "price_ohlcv": ["open", "high", "low", "close", "volume", "value"],
            "returns_risk": ["return", "log_return", "volatility_20", "var_95", "cvar_95"],
            "trend": ["sma_10", "sma_20", "sma_50", "macd", "macd_signal", "macd_hist"],
            "volatility": ["bb_upper", "bb_middle", "bb_lower", "atr_14"],
            "momentum": ["rsi_14", "stoch_k", "stoch_d"],
            "volume": ["obv"],
            "state_space_keys": ["ticker", "date"]
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(features_map, f, indent=2, ensure_ascii=False)
