"""Архитектура нейронной сети для BSAC агента."""
from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import BaseFeaturesExtractor


class BSACLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Экстрактор признаков на основе LSTM для временных рядов.
    Обрабатывает временные и статические признаки через LSTM и полносвязные слои.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int, n_assets: int, window_size: int, n_layers: int, dropout: float, **kwargs: Any) -> None:
        """
        Инициализация экстрактора признаков.
        
        Args:
            observation_space: Пространство наблюдений Gymnasium.
            features_dim: Размерность выходных признаков.
            n_assets: Количество активов в портфеле.
            window_size: Размер окна временного ряда.
            n_layers: Количество слоёв LSTM.
            dropout: Коэффициент dropout для регуляризации.
        """
        super().__init__(observation_space, features_dim)

        self.window_size = window_size
        self.static_dim = n_assets + 4
        total_obs = observation_space.shape[0]

        self.temporal_dim = (total_obs - self.static_dim) // window_size
        if self.temporal_dim * window_size + self.static_dim != total_obs:
            raise ValueError(
                f"Dimension mismatch: obs={total_obs}, window={window_size}, static={self.static_dim}"
            )

        self.lstm = nn.LSTM(
            input_size=self.temporal_dim,
            hidden_size=features_dim // 2,
            num_layers=n_layers,
            batch_first=True
        )

        self.fc_static = nn.Linear(self.static_dim, features_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(features_dim, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть извлечения признаков.

        Args:
            observations: Тензор наблюдений с формой (batch_size, obs_dim)
        
        Returns:
            Тензор извлеченных признаков с формой (batch_size, features_dim)
        """
        temporal_part = observations[:, :-self.static_dim].reshape(
            -1, self.window_size, self.temporal_dim
        )
        static_part = observations[:, -self.static_dim:]

        lstm_out, _ = self.lstm(temporal_part)
        temporal_flat = lstm_out[:, -1, :]

        static_out = self.relu(self.fc_static(static_part))

        combined = torch.cat([temporal_flat, static_out], dim=1)
        combined = self.dropout(combined)
        return self.relu(self.fc_out(combined))
