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
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass
    
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass
    
    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        if self.loss_function is LossFunction.MSE:
            y_pred = self.predict(x)
            return np.mean((y - y_pred) ** 2)
        
        elif self.loss_function is LossFunction.LogCosh:
            return 


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w

class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент MSE по весам
        :param x: Матрица признаков (размер [N, D])
        :param y: Вектор целевых значений (размер [N,])
        :return: Градиент (размер [D,])
        """
        n = x.shape[0] 
        prediction = x @ self.w  
        gradient = (2 / n) * (x.T @ (prediction - y))  
        return gradient

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновляет веса с использованием градиента и динамического шага eta_k
        :param gradient: Градиент (размер [D,])
        :return: Разница между новым и старым векторами весов (w_{k+1} - w_k)
        """
        eta_k = self.lr()

        w_old = self.w.copy()
        self.w = w_old - eta_k * gradient
        self.w = np.clip(self.w, -1e3, 1e3)

        return self.w - w_old

class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: размер батча (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент MSE по случайному батчу
        :param x: Матрица признаков (размер [N, D])
        :param y: Вектор целевых значений (размер [N,])
        :return: Градиент (размер [D,])
        """
        n = x.shape[0]

        batch_indices = np.random.randint(0, n, self.batch_size)

        x_batch = x[batch_indices]
        y_batch = y[batch_indices]

        prediction = x_batch @ self.w

        gradient = (2 / self.batch_size) * (x_batch.T @ (prediction - y_batch))

        return gradient

class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: число признаков (размерность весов)
        :param lambda_: скорость обучения
        """
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9 

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновляет веса с использованием метода Momentum
        :param gradient: Градиент (размер [D,])
        :return: Разница между новым и старым векторами весов (w_{k+1} - w_k)
        """
        eta_k = self.lr()

        self.h = self.alpha * self.h + eta_k * gradient

        w_old = self.w.copy()
        self.w = w_old - self.h
        self.w = np.clip(self.w, -1e3, 1e3)

        return self.w - w_old

class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: размерность признаков (число параметров модели)
        :param lambda_: скорость обучения
        """
        super().__init__(dimension, lambda_, loss_function)
        
        self.eps: float = 1e-8 
        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension) 

        self.beta_1: float = 0.9  
        self.beta_2: float = 0.999  

        self.iteration: int = 0  

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновляет веса с использованием Adam
        :param gradient: Градиент (размер [D,])
        :return: Разница между новым и старым векторами весов (w_{k+1} - w_k)
        """
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
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of MSE loss + L2 regularization with respect to weights.
        Regularization is NOT applied to w_0 (intercept).
        """
        n = x.shape[0]
        prediction = x @ self.w  
        base_gradient = (2 / n) * (x.T @ (prediction - y))

        l2_gradient = self.mu * self.w
        l2_gradient[-1] = 0
        return base_gradient + l2_gradient

class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
