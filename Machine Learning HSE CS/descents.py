from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5
    iteration: int = 0

    def __call__(self):
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        y_pred = self.predict(x)
        
        if self.loss_function is LossFunction.MSE:
            return (2 / n) * (x.T @ (y_pred - y))
        elif self.loss_function is LossFunction.MAE:
            return (x.T @ np.sign(y_pred - y)) / n
        elif self.loss_function is LossFunction.LogCosh:
            return (x.T @ np.tanh(y_pred - y)) / n
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0
            error = y_pred - y
            is_small_error = np.abs(error) <= delta
            gradient = np.where(is_small_error, error, delta * np.sign(error))
            return (x.T @ gradient) / n
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        if self.loss_function is LossFunction.MSE:
            return np.mean((y - y_pred) ** 2)
        elif self.loss_function is LossFunction.MAE:
            return np.mean(np.abs(y - y_pred))
        elif self.loss_function is LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(y_pred - y)))
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0
            error = y_pred - y
            is_small_error = np.abs(error) <= delta
            squared_loss = 0.5 * error**2
            linear_loss = delta * (np.abs(error) - 0.5 * delta)
            return np.mean(np.where(is_small_error, squared_loss, linear_loss))
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w


class VanillaGradientDescent(BaseDescent):

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        eta_k = self.lr()
        w_old = self.w.copy()
        self.w = w_old - eta_k * gradient
        self.w = np.clip(self.w, -1e3, 1e3)
        return self.w - w_old


class StochasticDescent(VanillaGradientDescent):
    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        batch_indices = np.random.choice(n, self.batch_size, replace=False)
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        return super().calc_gradient(x_batch, y_batch)

class MomentumDescent(VanillaGradientDescent):
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9
        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        eta_k = self.lr()
        self.h = self.alpha * self.h + eta_k * gradient
        w_old = self.w.copy()
        self.w = w_old - self.h
        self.w = np.clip(self.w, -1e3, 1e3)
        return self.w - w_old


class Adam(VanillaGradientDescent):
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8
        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
        self.beta_1: float = 0.9
        self.beta_2: float = 0.999
        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.iteration += 1
        eta_k = self.lr()
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)
        w_old = self.w.copy()
        self.w = w_old - eta_k * m_hat / (np.sqrt(v_hat) + self.eps)
        self.w = np.clip(self.w, -1e3, 1e3)
        return self.w - w_old


class BaseDescentReg(BaseDescent):
    def __init__(self, *args, mu: float = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        loss_gradient = super().calc_gradient(x, y)
        
        l2_gradient = self.mu * self.w
        if len(l2_gradient) > 0:  
            l2_gradient[-1] = 0 
        
        return loss_gradient + l2_gradient
class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    pass


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    pass


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    pass


class AdamReg(BaseDescentReg, Adam):
    pass


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    descent_class = descent_mapping[descent_name]
    kwargs = descent_config.get('kwargs', {})

    try:
        descent_instance = descent_class(**kwargs)
        
        if not hasattr(descent_instance, 'calc_loss'):
            raise AttributeError(f"{descent_class.__name__} has no calc_loss method")
            
        return descent_instance
        
    except TypeError as e:
        raise TypeError(f"Error creating {descent_class.__name__} with kwargs {kwargs}: {e}")
