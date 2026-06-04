"""BSAC агент для обучения и инференса."""
import json
import os
from datetime import datetime
from typing import Type

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize

from bsac_portfolio.bsac.callbacks import SaveVecNormalizeCallback
from bsac_portfolio.bsac.configs import EnvConfig, PolicyConfig, RiskProfile, TrainingConfig
from bsac_portfolio.bsac.environment import BSACEnv


class BSACAgent:
    """
    Управляющий класс агента SAC.
    
    Реализует обучение, сохранение, загрузку и инференс модели SAC.
    """

    def __init__(self, model_dir: str, env_config: EnvConfig = None, risk_profile: RiskProfile = None, training_cfg: TrainingConfig = None, policy_cfg: PolicyConfig = None, feature_extractor_class: type[BaseFeaturesExtractor] = None, device: str = "auto") -> None:
        """Инициализация BSAC агента.
        
        Args:
            model_dir: Директория модели.
            env_config: Конфигурация окружения.
            risk_profile: Конфигурация профиля риска.
            training_cfg: Конфигурация обучения.
            policy_cfg: Конфигурация политики.
            feature_extractor_class: Класс экстрактора признаков.
            device: Устройство для вычислений (cpu/cuda/auto).
        """
        self.model_dir = model_dir
        self.env_config = env_config
        self.risk_profile = risk_profile
        self.training_cfg = training_cfg
        self.policy_cfg = policy_cfg
        self.extractor_class = feature_extractor_class
        self.device = device
        if self.training_cfg is not None:
            np.random.seed(self.training_cfg.seed)
            torch.manual_seed(self.training_cfg.seed)
            torch.cuda.manual_seed_all(self.training_cfg.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.model: SAC = None
        self.train_env: VecNormalize = None
        self.eval_env: VecNormalize = None
        self.metadata: dict = {}
        os.makedirs(self.model_dir, exist_ok=True)

    def _make_env(self, df: pd.DataFrame, training: bool = False) -> VecNormalize:
        """Создание векторизованного нормализованного окружения.
        
        Args:
            df: DataFrame с данными для окружения.
            training: Флаг режима обучения (для нормализации).
            
        Returns:
            Векторизованное нормализованное окружение.
        """
        def make_env() -> BSACEnv:
            return BSACEnv(
                df=df,
                env_config=self.env_config,
                risk_profile=self.risk_profile,
                training_cfg=self.training_cfg
            )
        vec_env = make_vec_env(make_env, n_envs=self.training_cfg.n_envs)
        return VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=self.training_cfg.clip_obs,
            clip_reward=self.training_cfg.clip_reward,
            gamma=self.training_cfg.gamma,
            training=training
        )

    def _build_policy_kwargs(self) -> dict:
        """Построение параметров архитектуры политики.
        
        Returns:
            Словарь параметров для политики SAC.
        """
        return dict(
            features_extractor_class=self.extractor_class,
            features_extractor_kwargs=dict(
                features_dim=self.policy_cfg.features_dim,
                n_assets=len(self.env_config.tickers),
                window_size=self.env_config.window_size,
                n_layers=self.policy_cfg.n_layers,
                dropout=self.policy_cfg.dropout
            ),
            net_arch=dict(pi=self.policy_cfg.pi_net, qf=self.policy_cfg.qf_net)
        )

    def _validate_inference_data(self, df: pd.DataFrame) -> None:
        """
        Валидирует входные данные для инференса.
        
        Args:
            df: DataFrame с данными для инференса.
        """
        try:
            tickers = getattr(self.env_config, 'tickers', None)
            features = getattr(self.env_config, 'feature_columns', None)
            window_size = getattr(self.env_config, 'window_size', 60)
            if not tickers or not features:
                return
        except Exception:
            return

        expected_cols = [
            f"{feat}_{t}" for feat in features for t in tickers
        ]

        actual_cols = list(df.columns)

        if actual_cols != expected_cols:
            missing = [c for c in expected_cols if c not in actual_cols]
            extra = [c for c in actual_cols if c not in expected_cols]
            msg = "Входные данные не соответствуют конфигурации модели."
            if missing:
                msg += f" Отсутствуют колонки: {missing}."
            if extra:
                msg += f" Найдены лишние: {extra}."
            if not missing and not extra:
                msg += " Нарушен порядок колонок."
            raise ValueError(msg)

        if len(df) < window_size:
            raise ValueError(
                f"Недостаточно данных для окна наблюдения. "
                f"Требуется минимум {window_size} строк, получено {len(df)}."
            )

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """
        Обучение модели на обучающих данных.
        
        Args:
            train_df: DataFrame с обучающими данными.
            val_df: DataFrame с валидационными данными.
        """
        self.train_env = self._make_env(train_df, training=True)
        eval_env = self._make_env(val_df, training=False)
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=self.training_cfg.max_no_improvement_evals,
            min_evals=self.training_cfg.min_evals,
            verbose=self.training_cfg.verbose
        )
        save_best_norm_cb = SaveVecNormalizeCallback(
            self.train_env,
            os.path.join(self.model_dir, "vec_normalize_best.pkl")
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=os.path.join(self.model_dir, "eval_logs"),
            eval_freq=self.training_cfg.eval_freq,
            n_eval_episodes=self.training_cfg.n_eval_episodes,
            deterministic=True,
            callback_after_eval=stop_cb,
            callback_on_new_best=save_best_norm_cb,
            verbose=self.training_cfg.verbose
        )
        self.model = SAC(
            "MlpPolicy",
            self.train_env,
            policy_kwargs=self._build_policy_kwargs(),
            learning_rate=self.training_cfg.learning_rate,
            ent_coef=self.training_cfg.ent_coef,
            target_entropy=self.training_cfg.target_entropy,
            batch_size=self.training_cfg.batch_size,
            buffer_size=self.training_cfg.buffer_size,
            gamma=self.training_cfg.gamma,
            tau=self.training_cfg.tau,
            train_freq=self.training_cfg.train_freq,
            gradient_steps=self.training_cfg.gradient_steps,
            device=self.device,
            verbose=self.training_cfg.verbose
        )
        self.model.learn(
            total_timesteps=self.training_cfg.total_timesteps,
            callback=eval_cb,
            progress_bar=True
        )
        self.save(filename="final_model.zip")

    def save(self, filename: str = "best_model.zip") -> None:
        """
        Сохраняет модель и метаданные.
        
        Args:
            filename: Имя файла для сохранения модели.
        """
        if self.model is None:
            raise RuntimeError("Обучите модель перед сохранением")
        model_path = os.path.join(self.model_dir, filename)
        self.model.save(model_path)
        if self.train_env is not None:
            norm_name = (
                "vec_normalize_best.pkl"
                if "best" in filename
                else "vec_normalize.pkl"
            )
            self.train_env.save(os.path.join(self.model_dir, norm_name))
        self.metadata = {
            "env_config": self.env_config.__dict__,
            "risk_profile": self.risk_profile.__dict__,
            "training_cfg": self.training_cfg.__dict__,
            "policy_cfg": self.policy_cfg.__dict__,
            "extractor_module": self.extractor_class.__module__,
            "extractor_class": self.extractor_class.__name__,
            "timestamp": datetime.now().isoformat(),
            "saved_filename": filename
        }
        with open(os.path.join(self.model_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"Модель сохранена в {model_path}")

    def load(self, feature_extractor_class: Type[BaseFeaturesExtractor] = None, filename: str = "best_model.zip") -> None:
        """
        Загрузка обученной модели и метаданных.
        
        Args:
            feature_extractor_class: Класс экстрактора признаков.
            filename: Имя файла модели для загрузки.
        """
        meta_path = os.path.join(self.model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("metadata.json не найден в директории модели")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.env_config = EnvConfig(**self.metadata["env_config"])
        self.risk_profile = RiskProfile(**self.metadata["risk_profile"])
        self.training_cfg = TrainingConfig(**self.metadata["training_cfg"])
        self.policy_cfg = PolicyConfig(**self.metadata["policy_cfg"])
        if feature_extractor_class:
            self.extractor_class = feature_extractor_class
        elif self.extractor_class is None:
            raise ValueError("feature_extractor_class должен быть передан при загрузке")
        n_assets = len(self.env_config.tickers)
        n_temporal_total = n_assets * len(self.env_config.feature_columns)
        static_dim = n_assets + 4
        obs_dim = (self.env_config.window_size * n_temporal_total) + static_dim

        class DummyEnv(gym.Env):
            def __init__(self, obs_shape: int, n_actions: int):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape,), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    -1.0, 1.0, shape=(n_actions,), dtype=np.float32
                )

            def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
                return np.zeros(obs_shape, dtype=np.float32), {}

            def step(self, a: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
                return np.zeros(obs_shape, dtype=np.float32), 0.0, True, False, {}

        dummy_vec = make_vec_env(
            lambda: DummyEnv(obs_dim, n_assets), n_envs=1
        )
        norm_filename = (
            "vec_normalize_best.pkl"
            if "best" in filename
            else "vec_normalize.pkl"
        )
        stats_path = os.path.join(self.model_dir, norm_filename)
        if not os.path.exists(stats_path):
            stats_path = os.path.join(self.model_dir, "vec_normalize.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Файл статистики нормализации не найден: {stats_path}")
        self.train_env = VecNormalize.load(stats_path, dummy_vec)
        self.train_env.training = False
        self.train_env.norm_reward = False
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{filename} не найден в {self.model_dir}")
        self.model = SAC(
            "MlpPolicy",
            self.train_env,
            policy_kwargs=self._build_policy_kwargs(),
            device=self.device
        )
        self.model.set_parameters(model_path)
        print(f"Модель загружена из {model_path} с статистикой из {stats_path}")

    def predict(self, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Предсказание траектории портфеля на тестовых данных.
        
        Args:
            test_df: DataFrame с тестовыми данными.
            
        Returns:
            Словарь с траекторией значений портфеля, весов, доходностей и полезностью.
        """
        if self.model is None:
            raise RuntimeError("Загрузите модель перед предсказанием")

        self._validate_inference_data(test_df)

        raw_env = BSACEnv(test_df, self.env_config, self.risk_profile, self.training_cfg)
        obs, _ = raw_env.reset()

        trajectory = {
            "portfolio_values": [raw_env.portfolio_value],
            "weights": [],
            "returns": [],
            "utilities": [],
            "raw_utilities": [],
            "z_values": []
        }

        while True:
            norm_obs = self.train_env.normalize_obs(obs.reshape(1, -1))[0]
            action, _ = self.model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = raw_env.step(action)

            trajectory["portfolio_values"].append(raw_env.portfolio_value)
            trajectory["weights"].append(raw_env.portfolio_weights.copy())
            trajectory["returns"].append(info.get("portfolio_return", 0.0))
            trajectory["utilities"].append(reward)
            trajectory["raw_utilities"].append(info.get("raw_utility", 0.0))
            trajectory["z_values"].append(raw_env.z)

            if terminated or truncated:
                break

        return {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in trajectory.items()
        }

    def finetune(self, new_train_df: pd.DataFrame, new_val_df: pd.DataFrame, additional_timesteps: int = 10000, new_learning_rate: float = None) -> None:
        """
        Дообучение модели на новых данных.
        
        Args:
            new_train_df: DataFrame с новыми обучающими данными.
            new_val_df: DataFrame с новыми валидационными данными.
            additional_timesteps: Количество дополнительных шагов обучения.
            new_learning_rate: Новый learning rate.
        """
        if self.model is None or self.train_env is None:
            raise RuntimeError("Загрузите модель перед дообучением")
        lr = (
            new_learning_rate
            if new_learning_rate is not None
            else self.training_cfg.learning_rate * 0.1
        )
        self.model.learning_rate = lr
        self.training_cfg.learning_rate = lr
        self.train_env = self._make_env(new_train_df, training=True)
        self.eval_env = self._make_env(new_val_df, training=False)
        self.model.set_env(self.train_env)
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=self.training_cfg.max_no_improvement_evals,
            min_evals=self.training_cfg.min_evals,
            verbose=self.training_cfg.verbose
        )
        save_best_norm_cb = SaveVecNormalizeCallback(
            self.train_env,
            os.path.join(self.model_dir, "vec_normalize_best.pkl")
        )
        eval_cb = EvalCallback(
            self.eval_env,
            best_model_save_path=self.model_dir,
            log_path=os.path.join(self.model_dir, "finetune_logs"),
            eval_freq=self.training_cfg.eval_freq,
            n_eval_episodes=self.training_cfg.n_eval_episodes,
            deterministic=True,
            callback_after_eval=stop_cb,
            callback_on_new_best=save_best_norm_cb,
            verbose=self.training_cfg.verbose
        )
        self.model.learn(
            total_timesteps=additional_timesteps,
            callback=eval_cb,
            reset_num_timesteps=False,
            progress_bar=True
        )
        self.metadata["finetuned_steps"] = (
            self.metadata.get("finetuned_steps", 0) + additional_timesteps
        )
        self.metadata["finetuned_lr"] = lr
