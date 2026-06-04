"""Конфигурации для BSAC агента."""
from dataclasses import dataclass, field
from typing import Union


@dataclass
class EnvConfig:
    """
    Конфигурация среды.
    """
    tickers: list[str]
    feature_columns: list[str]
    window_size: int = 60
    initial_balance: float = 1.0
    transaction_cost_pct: float = 0.0003
    risk_free_rate_daily: float = 0.0002
    action_smoothing_alpha: float = 1.0
    scale_reward: float = 1.0


@dataclass
class RiskProfile:
    """
    Конфигурация профиля риска.
    """
    profile_id: str
    name: str
    lambda_val: float
    k: float
    eta: float
    rebalance_penalty: float
    max_lambda: float = 6.0


@dataclass
class PolicyConfig:
    """
    Конфигурация политики.
    """
    features_dim: int = 64
    dropout: float = 0.2
    n_layers: int = 2
    pi_net: list[int] = field(default_factory=lambda: [128, 64])
    qf_net: list[int] = field(default_factory=lambda: [128, 64])


@dataclass
class TrainingConfig:
    """
    Конфигурация обучения.
    """
    total_timesteps: int = 150000
    learning_rate: float = 3e-4
    batch_size: int = 512
    buffer_size: int = 150000
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = -5.0
    train_freq: int = 4
    gradient_steps: int = 4
    n_envs: int = 2
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    eval_freq: int = 5000
    n_eval_episodes: int = 5
    max_no_improvement_evals: int = 10
    min_evals: int = 3
    verbose: int = 2
    seed: int = 0
