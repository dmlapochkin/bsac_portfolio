"""Среда для агента BSAC."""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from bsac_portfolio.bsac.configs import EnvConfig, RiskProfile, TrainingConfig


class BSACEnv(gym.Env):
    """
    Среда для обучения агента BSAC.
    
    Реализует окружение Gymnasium с поведенческой функцией полезности на основе модели Барбериса, Хуанга и Сантоса.
    """

    def __init__(self, df: pd.DataFrame, env_config: EnvConfig, risk_profile: RiskProfile, training_cfg: TrainingConfig = None) -> None:
        """
        Инициализация среды.
        
        Args:
            df: DataFrame с рыночными данными и признаками.
            env_config: Конфигурация окружения.
            risk_profile: Конфигурация профиля риска.
            training_cfg: Конфигурация обучения.
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.config = env_config
        self.profile = risk_profile
        self.training_cfg = training_cfg or TrainingConfig()

        self.temporal_cols = [c for c in self.df.columns if c.lower() not in ['date', 'ticker']]
        self.n_temporal = len(self.temporal_cols)
        self.n_assets = len(self.config.tickers)

        self.ret_cols = [c for c in self.temporal_cols if 'ret' in c.lower()]
        if len(self.ret_cols) != self.n_assets:
            raise ValueError(
                f"Ожидалось {self.n_assets} - колонки доходности, найдено: {len(self.ret_cols)}"
            )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        static_dim = self.n_assets + 4
        obs_dim = (self.config.window_size * self.n_temporal) + static_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.portfolio_value = self.config.initial_balance
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        self.z = 1.0
        self.history_buffer: list[np.ndarray] = []
        self.prev_action: np.ndarray = None

    def _calculate_barberis_reward(self, portfolio_return: float) -> tuple[float, float]:
        """
        Рассчитывает значение функции полезности и обновляет состояние z (нормированный бенчмарк).

        Args:
            portfolio_return: Доходность портфеля за текущий шаг

        Returns:
            Кортеж (reward, z_next) - значение функции полезности и обновленное значение z для следующего шага
        """
        R_p = 1.0 + portfolio_return
        R_f = 1.0 + self.config.risk_free_rate_daily

        if self.z <= 1.0:
            if R_p >= self.z * R_f:
                utility = R_p - R_f
            else:
                utility = self.z * R_f - R_f + self.profile.lambda_val * (R_p - self.z * R_f)
        else:
            if R_p >= R_f:
                utility = R_p - R_f
            else:
                utility = (R_p - R_f) * (self.profile.lambda_val + self.profile.k * (self.z - 1.0))

        z_next = self.profile.eta * self.z * (R_f / R_p) + (1.0 - self.profile.eta)
        z_next = np.clip(z_next, 0.5, 2.0)

        reward = utility / self.config.scale_reward
        return reward, z_next

    def _get_features_vector(self, step: int) -> np.ndarray:
        """
        Извлекает вектор признаков для указанного шага.
        
        Args:
            step: Индекс шага в DataFrame

        Returns:
            Вектор признаков
        """
        return self.df.iloc[step][self.temporal_cols].values.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Формирует вектор состояния для агента.
        
        Returns:
            Вектор состояния, включающий признаки за окно и статические характеристики портфеля и профиля риска.
        """
        window_features = np.array(self.history_buffer[-self.config.window_size:]).flatten()
        static_features = np.concatenate([
            self.portfolio_weights,
            [self.z],
            [self.profile.lambda_val, self.profile.k, self.profile.eta]
        ]).astype(np.float32)
        return np.concatenate([window_features, static_features])

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """
        Сбрасывает среду к начальному состоянию.

        Args:
            seed: Сид для генератора случайных чисел
            options: Дополнительные опции для сброса

        Returns:
            Кортеж (obs, info) - начальное наблюдение и словарь с информа
        """
        super().reset(seed=seed, options=options)
        self.current_step = self.config.window_size
        self.portfolio_value = self.config.initial_balance
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        self.z = 1.0
        self.prev_action = None
        self.history_buffer = [
            self._get_features_vector(i) for i in range(self.config.window_size)
        ]
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Выполняет шаг среды с применением действия агента.
        Применяет действие, вычисляет доходность, комиссию и награду, обновляет состояние.

        Args:
            action: Вектор действий

        Returns:
            Кортеж (obs, reward, done, truncated, info) - новое наблюдение, награда, флаг окончания эпизода, флаг усечения и словарь с информацией
        """
        if self.config.action_smoothing_alpha < 1.0 and self.prev_action is not None:
            action = (
                self.config.action_smoothing_alpha * action
                + (1.0 - self.config.action_smoothing_alpha) * self.prev_action
            )

        target_weights = np.exp(action) / np.sum(np.exp(action))
        turnover = np.sum(np.abs(target_weights - self.portfolio_weights))
        commission = turnover * self.config.transaction_cost_pct

        if self.current_step >= len(self.df):
            return (
                np.zeros(self.observation_space.shape[0], dtype=np.float32),
                0.0, False, True, {}
            )

        asset_returns = self.df.iloc[self.current_step][self.ret_cols].values.astype(float)
        portfolio_return = np.dot(self.portfolio_weights, asset_returns) - commission

        self.portfolio_value *= (1.0 + portfolio_return)
        self.portfolio_weights = target_weights.copy()
        self.prev_action = action.copy()

        raw_utility, self.z = self._calculate_barberis_reward(portfolio_return)

        reward = raw_utility - (turnover * self.profile.rebalance_penalty)

        self.history_buffer.append(self._get_features_vector(self.current_step))
        self.current_step += 1

        truncated = self.current_step >= len(self.df)
        obs = (
            self._get_obs()
            if not truncated
            else np.zeros(self.observation_space.shape[0], dtype=np.float32)
        )

        info = {
            "portfolio_return": float(portfolio_return),
            "turnover": float(turnover),
            "commission": float(commission),
            "portfolio_value": float(self.portfolio_value),
            "z_value": float(self.z),
            "raw_utility": float(raw_utility)
        }
        return obs, reward, False, truncated, info
