from __future__ import annotations

from typing import List
import numpy as np
from descents import BaseDescent, get_descent

class LinearRegression:
    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 500):
        self.descent = get_descent(descent_config)
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_history = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        self.loss_history = [self.calc_loss(x, y)]
        
        for iteration in range(self.max_iter):
            delta_w = self.descent.step(x, y)
            current_loss = self.calc_loss(x, y)
            self.loss_history.append(current_loss)
            
            if np.linalg.norm(delta_w) < self.tolerance:
                break
                
            if (np.isnan(delta_w).any() or np.isinf(delta_w).any() or 
                np.isnan(self.descent.w).any() or np.isinf(self.descent.w).any()):
                print(f"Предупреждение: расходимость на итерации {iteration}")
                break
                
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.descent.calc_loss(x, y)
