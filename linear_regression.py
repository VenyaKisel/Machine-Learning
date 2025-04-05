from __future__ import annotations

from typing import List
import numpy as np
from descents import BaseDescent, get_descent


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Fitting descent weights for x and y dataset
        :param x: features array
        :param y: targets array
        :return: self
        """

        # Сохраняем начальный лосс
        self.loss_history.append(self.calc_loss(x, y))

        for iteration in range(self.max_iter):
            old_weights = self.descent.w.copy()

            # Проверка перед обновлением весов
            if np.isnan(self.descent.w).any() or np.isinf(self.descent.w).any():
                raise ValueError("NaN/Inf detected in weights before update! Training has diverged.")

            # Ограничение весов перед шагом
            self.descent.w = np.clip(self.descent.w, -1e3, 1e3)

            delta_w = self.descent.step(x, y)

            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)

            if np.linalg.norm(delta_w) ** 2 < self.tolerance:
                break

            if np.isnan(delta_w).any():
                break

            if np.isnan(self.descent.w).any():
                break 

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        # Проверка входных данных
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("NaN/Inf detected in input features!")

        if x.shape[1] != self.descent.w.shape[0]:
            raise ValueError(f"Feature mismatch: x.shape[1] = {x.shape[1]}, w.shape[0] = {self.descent.w.shape[0]}")

        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        return self.descent.calc_loss(x, y)
