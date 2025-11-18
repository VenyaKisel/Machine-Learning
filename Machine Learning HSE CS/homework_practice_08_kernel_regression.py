import numpy as np
from typing import Callable
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression, make_friedman1
from sklearn.gaussian_process.kernels import RBF
from catboost import CatBoostClassifier
from homework_practice_08_rff import FeatureCreatorPlaceholder

class KernelRidgeRegression(RegressorMixin):
    def __init__(
        self,
        lr=0.01,
        regularization=1.0,
        tolerance=1e-6,
        max_iter=1000,
        kernel_scale=1.0,
        fit_bias=True
    ):
        self.lr = float(lr)
        self.regularization = float(regularization)
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kernel_scale = kernel_scale
        self.fit_bias = fit_bias

        self.kernel = RBF(kernel_scale)
        self.scaler = StandardScaler()

        self.w = None
        self.X_train = None
        self.b = 0.0
        self.loss_history = []

    def get_params(self, deep=True):
        return {
            "lr": self.lr,
            "regularization": self.regularization,
            "tolerance": self.tolerance,
            "max_iter": self.max_iter,
            "kernel_scale": self.kernel_scale,
            "fit_bias": self.fit_bias,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        if "kernel_scale" in params:
            self.kernel = RBF(self.kernel_scale)
        return self

    def _compute_kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2)

    def calc_loss(self, x, y):

        K = self._compute_kernel_matrix(x, self.X_train)
        y_pred = K @ self.w + self.b

        data_loss = 0.5 * np.mean((y_pred - y) ** 2)
        reg_loss = 0.5 * self.regularization * np.dot(self.w, self.w)

        return data_loss + reg_loss

    def calc_grad(self, x, y):

        K_x = self._compute_kernel_matrix(x, self.X_train)

        y_pred = K_x @ self.w + self.b
        error = y_pred - y

        grad_w = (K_x.T @ error) / len(y) + self.regularization * self.w
        grad_b = np.mean(error) if self.fit_bias else 0.0

        return grad_w, grad_b

    def fit(self, x, y):
        x_scaled = self.scaler.fit_transform(x)
        n_samples = x_scaled.shape[0]

        self.w = np.random.normal(0, 0.01, size=n_samples)
        self.X_train = x_scaled.copy()
        self.b = 0.0

        for i in range(self.max_iter):
            grad_w, grad_b = self.calc_grad(x_scaled, y)

            w_new = self.w - self.lr * grad_w
            b_new = self.b - self.lr * grad_b if self.fit_bias else self.b

            diff_norm_sq = np.sum((w_new - self.w) ** 2)
            if self.fit_bias:
                diff_norm_sq += (b_new - self.b) ** 2

            self.w = w_new
            self.b = b_new

            if i % 100 == 0:
                self.loss_history.append(self.calc_loss(x_scaled, y))

            if diff_norm_sq < self.tolerance:
                break

        final_loss = self.calc_loss(x_scaled, y)
        self.loss_history.append(final_loss)

        return self

    def fit_closed_form(self, x, y):
        x_scaled = self.scaler.fit_transform(x)
        n_samples = x_scaled.shape[0]

        self.X_train = x_scaled.copy()
        K = self._compute_kernel_matrix(x_scaled, x_scaled)
        I = np.eye(n_samples)

        self.w = np.linalg.solve(K + self.regularization * I, y)

        if self.fit_bias:
            self.b = np.mean(y - K @ self.w)
        else:
            self.b = 0.0

        self.loss_history.append(self.calc_loss(x_scaled, y))
        return self

    def predict(self, x):
        x_scaled = self.scaler.transform(x)
        K_test = self._compute_kernel_matrix(x_scaled, self.X_train)
        return K_test @ self.w + self.b



    
class RFFRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_features: int = 1000,
        new_dim: int = 50,
        use_PCA: bool = True,
        feature_creator_class=FeatureCreatorPlaceholder,
        regressor: str = 'ridge',
        regressor_params=None,
        func=np.cos,
        random_state=None
    ):
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.regressor_type = regressor
        self.regressor_params = regressor_params if regressor_params is not None else {}
        self.func = func
        self.random_state = random_state
        self.feature_creator_class = feature_creator_class
        self.pipeline = None
        self._initialize_regressor()

    def _initialize_regressor(self):
        if self.regressor_type == "ridge":
            self.regressor = Ridge(**self.regressor_params)
        elif self.regressor_type == "linear":
            self.regressor = LinearRegression(**self.regressor_params)
        elif self.regressor_type == "svr_linear":
            svr_params = self.regressor_params.copy()
            svr_params['kernel'] = 'linear'
            self.regressor = SVR(**svr_params)
        elif self.regressor_type == "svr_rbf":
            svr_params = self.regressor_params.copy()
            svr_params['kernel'] = 'rbf'
            self.regressor = SVR(**svr_params)
        else:
            raise ValueError(f"Unsupported regressor: {self.regressor_type}")

    def fit(self, X, y):
        pipeline_steps = []
        pipeline_steps.append(('scaler', StandardScaler()))
        
        if self.use_PCA:
            n_components = min(self.new_dim, X.shape[1])
            pca_transformer = PCA(n_components=n_components, random_state=self.random_state)
            pipeline_steps.append(('pca', pca_transformer))
            feature_dim = n_components
        else:
            feature_dim = X.shape[1]
        
        feature_creator = self.feature_creator_class(
            n_features=feature_dim,
            new_dim=self.n_features,
            func=self.func
        )
        if hasattr(feature_creator, 'random_state'):
            feature_creator.random_state = self.random_state
            
        pipeline_steps.append(('feature_creator', feature_creator))
        pipeline_steps.append(('regressor', self.regressor))
        
        self.pipeline = Pipeline(pipeline_steps)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("Model not fitted yet")
        return self.pipeline.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
