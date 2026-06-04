"""Коллбэки для обучения BSAC агента."""
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class SaveVecNormalizeCallback(BaseCallback):
    """
    Коллбэк для сохранения статистики нормализации VecNormalize.
    
    Сохраняет параметры нормализации наблюдений и наград в файл.
    """

    def __init__(self, vec_norm: VecNormalize, save_path: str, verbose: int = 0) -> None:
        """
        Инициализация коллбэка.
        
        Args:
            vec_norm: Объект VecNormalize для сохранения.
            save_path: Путь к файлу для сохранения статистики.
            verbose: Уровень детализации логирования.
        """
        super().__init__(verbose)
        self.vec_norm = vec_norm
        self.save_path = save_path

    def _on_step(self) -> bool:
        """
        Сохранение статистики нормализации на каждом шаге.
        
        Returns:
            True для продолжения обучения, False для остановки
        """
        self.vec_norm.save(self.save_path)
        return True
