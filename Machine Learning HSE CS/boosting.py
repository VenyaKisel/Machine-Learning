from __future__ import annotations

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
from typing import Optional, Tuple, Union, Literal
from sklearn.model_selection import train_test_split
from optuna.storages import RDBStorage

import optuna

def score(clf, x, y):
    return roc_auc_score(y, clf.predict_proba(x)[:, 1])

class Boosting:
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        random_state: int = 42,
        bootstrap_type: str | None = 'Bernoulli',
        bagging_temperature: float = 1.0,
        rsm: float = 1.0,
        quantization_type: str | None = 'Uniform',
        nbins: int = 255,
        goss: bool = False,
        goss_k: float = 0.2,
        goss_subsample: float = 0.3,
        top_importances: Optional[int] = 10,
        print_feature_importances: bool = False,
        enable_optimize: bool = False,
        enable_plot_optimize: bool = False,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
        dart: bool = False,
        dropout_rate: float = 0.1
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = base_model_params or {}
        if max_depth is not None:
            self.base_model_params['max_depth'] = max_depth
        if min_samples_split is not None:
            self.base_model_params['min_samples_split'] = min_samples_split
            
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.init_bias = None 
        self.models: list = []
        self.gammas: list = []
        self.history = defaultdict(list)
        self.best_iter = None
        self.subsample = subsample
        self.rng = np.random.RandomState(random_state)
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.feature_indices = []
        self.quantizers = []
        self.goss = goss
        self.goss_k = goss_k
        self.goss_subsample = goss_subsample
        self.top_importances = top_importances
        self.print_feature_importances = print_feature_importances
        self.enable_optimize = enable_optimize
        self.enable_plot_optimize = enable_plot_optimize
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.dart = dart
        self.dropout_rate = dropout_rate
        
        self.sigmoid = lambda x: np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        
        self.loss_fn = lambda y, z: np.mean(
            np.log(1 + np.exp(-y * z)))
        
        self.loss_derivative = lambda y, z: -y / (1 + np.exp(y * z))
        
    def _get_bootstrap_weights(self, n_samples):
        if self.bootstrap_type == 'Bernoulli':
            weights = np.ones(n_samples)
            if self.subsample < 1.0:
                mask = self.rng.rand(n_samples) < self.subsample
                weights[~mask] = 0
            return weights
        
        elif self.bootstrap_type == 'Bayesian':
            U = self.rng.uniform(0, 1, n_samples)
            weights = (-np.log(U)) ** self.bagging_temperature
            if self.subsample < 1.0:
                mask = self.rng.rand(n_samples) < self.subsample
                weights[~mask] = 0
            return weights
        else:
            return np.ones(n_samples)
        
    def _select_features(self, n_features):
        if self.rsm < 1.0:
            n_selected = max(1, int(self.rsm * n_features))
            return self.rng.choice(n_features, size=n_selected, replace=False)
        return np.arange(n_features)
    
    def _quantize_features(self, X, fit=False):
        if self.quantization_type is None:
            return X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        elif hasattr(X, 'todense'):
            X = X.todense()
        if self.quantization_type in ['Quantile', 'Uniform']:
            strategy = 'quantile' if self.quantization_type == 'Quantile' else 'uniform'
            
            if fit:
                quantizer = KBinsDiscretizer(
                    n_bins=self.nbins, 
                    encode='ordinal', 
                    strategy=strategy,
                    subsample=None
                )
                X_quantized = quantizer.fit_transform(X)
                self.quantizers.append(quantizer)
            else:
                quantizer = self.quantizers[-1]
                X_quantized = quantizer.transform(X)
            return X_quantized
        return X
        
    def _predict(self, X, training=False):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        elif hasattr(X, 'todense'):
            X = X.todense()
        if self.init_bias is None:
            return np.zeros(X.shape[0])
        tree_sum = np.zeros(X.shape[0])
        
        if self.dart and training and len(self.models) > 0:
            n_drop = int(self.dropout_rate * len(self.models))
            if n_drop > 0:
                drop_indices = self.rng.choice(len(self.models), size=n_drop, replace=False)
                kept_indices = [i for i in range(len(self.models)) if i not in drop_indices]
                
                for i in kept_indices:
                    model, gamma = self.models[i], self.gammas[i]
                    X_selected = X[:, self.feature_indices[i]]
                    if self.quantization_type is not None:
                        X_selected = self._quantize_features(X_selected, fit=False)
                    tree_sum += self.learning_rate * gamma * model.predict(X_selected)
                
                if len(kept_indices) > 0:
                    tree_sum *= len(self.models) / len(kept_indices)
            else:
                for i, (model, gamma) in enumerate(zip(self.models, self.gammas)):
                    X_selected = X[:, self.feature_indices[i]]
                    if self.quantization_type is not None:
                        X_selected = self._quantize_features(X_selected, fit=False)
                    tree_sum += self.learning_rate * gamma * model.predict(X_selected)
        else:
            for i, (model, gamma) in enumerate(zip(self.models, self.gammas)):
                X_selected = X[:, self.feature_indices[i]]
                if self.quantization_type is not None:
                    X_selected = self._quantize_features(X_selected, fit=False)
                tree_sum += self.learning_rate * gamma * model.predict(X_selected)

        return self.init_bias + tree_sum

    def partial_fit(self, X, y):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        elif hasattr(X, 'todense'):
            X = X.todense()
        if self.init_bias is None:
            p = np.clip(np.mean(y == 1), 1e-5, 1 - 1e-5)
            self.init_bias = np.log(p / (1 - p))

        pred = self._predict(X, training=True)
        residuals = -self.loss_derivative(y, pred)
        
        n_samples = X.shape[0]
        
        if self.goss:
            abs_gradients = np.abs(residuals)
            sorted_indices = np.argsort(abs_gradients)[::-1]
            
            n_top = int(self.goss_k * n_samples)
            top_indices = sorted_indices[:n_top]
            
            remaining_indices = sorted_indices[n_top:]
            n_remaining = len(remaining_indices)
            n_subsample = int(self.goss_subsample * n_remaining)
            
            if n_subsample > 0:
                subsample_indices = self.rng.choice(
                    remaining_indices, 
                    size=n_subsample, 
                    replace=False
                )
                selected_indices = np.concatenate([top_indices, subsample_indices])
                
                weights = np.ones(n_samples)
                scale_factor = (1 - self.goss_k) / self.goss_subsample
                weights[subsample_indices] = scale_factor
            else:
                selected_indices = top_indices
                weights = np.ones(n_samples)
            
            if self.subsample < 1.0:
                bootstrap_weights = self._get_bootstrap_weights(n_samples)
                weights = weights * bootstrap_weights
            
            X_selected_samples = X[selected_indices]
            residuals_selected = residuals[selected_indices]
            weights_selected = weights[selected_indices]
        else:
            weights = self._get_bootstrap_weights(n_samples)
            selected_indices = np.where(weights > 0)[0]
            
            if len(selected_indices) > 0:
                X_selected_samples = X[selected_indices]
                residuals_selected = residuals[selected_indices]
                weights_selected = weights[selected_indices]
            else:
                X_selected_samples = X
                residuals_selected = residuals
                weights_selected = np.ones(n_samples)
        
        n_features = X.shape[1]
        feature_idx = self._select_features(n_features)
        self.feature_indices.append(feature_idx)
        X_selected = X_selected_samples[:, feature_idx]
        
        if self.quantization_type is not None:
            X_selected = self._quantize_features(X_selected, fit=True)
        
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_selected, residuals_selected, sample_weight=weights_selected)
        
        new_pred = model.predict(X[:, feature_idx])

        gamma = self.find_optimal_gamma(y, pred, new_pred)

        self.models.append(model)
        self.gammas.append(gamma)
        return self

    def fit(
        self, 
        X_train, 
        y_train, 
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        early_stopping_rounds: Optional[int] = None,
        plot: bool = False,
        callback: Optional[callable] = None
    ):
        self.models = []
        self.gammas = []
        self.history = defaultdict(list)
        self.init_bias = None
        self.best_iter = 0
        self.feature_indices = []
        self.quantizers = []
        
        if eval_set is not None:
            X_val, y_val = eval_set
        
        best_val_loss = float('inf')
        rounds_without_improvement = 0
        
        for i in range(self.n_estimators):
            self.partial_fit(X_train, y_train)

            z_tr = self._predict(X_train, training=False)
            train_loss = self.loss_fn(y_train, z_tr)
            self.history["train_loss"].append(train_loss)

            if X_val is not None and y_val is not None:
                z_val = self._predict(X_val, training=False)
                val_loss = self.loss_fn(y_val, z_val)
                val_auc = roc_auc_score(y_val, self.sigmoid(z_val))
                
                self.history["val_loss"].append(val_loss)
                self.history["val_roc_auc"].append(val_auc)
                
                if callback is not None:
                    callback(i, train_loss, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                    self.best_iter = i
                else:
                    rounds_without_improvement += 1
                    
                if early_stopping_rounds is not None and rounds_without_improvement >= early_stopping_rounds:
                    print(f"Early stopping at iteration {i}, best iteration is {self.best_iter}")
                    self.models = self.models[:self.best_iter+1]
                    self.gammas = self.gammas[:self.best_iter+1]
                    self.feature_indices = self.feature_indices[:self.best_iter+1]
                    if self.quantization_type is not None:
                        self.quantizers = self.quantizers[:self.best_iter+1]
                    break
            else:
                if callback is not None:
                    callback(i, train_loss, None)
        
        if self.print_feature_importances:
            print("\n")
            print("="*50)
            importances = self.get_feature_importances(top_k=self.top_importances)
            for feature, importance in importances.items():
                print(f"{feature}: {importance:.4f}")
            print("="*50)
                    
        if plot:
            self.plot_history(X_val, y_val)
            
    def predict_proba(self, X):
        z = self._predict(X, training=False)
        proba_positive = self.sigmoid(z)
        return np.vstack([1 - proba_positive, proba_positive]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=5, num=100)
        losses = [self.loss_fn(y, old_predictions + self.learning_rate * gamma * new_predictions) 
                 for gamma in gammas]
        return gammas[np.argmin(losses)]
    
    def get_feature_importances(self, top_k: Optional[int] = None):
        if not self.models:
            return {}
            
        n_features = self.models[0].feature_importances_.shape[0]
        total_importances = np.zeros(n_features)
            
        for model in self.models:
            importances = model.feature_importances_
            total_importances += importances
                
        total_importances /= len(self.models)
            
        if total_importances.sum() > 0:
            total_importances /= total_importances.sum()
            
        feature_importances = {}
        for i, importance in enumerate(total_importances):
            feature_importances[f'feature_{i}'] = importance
            
        sorted_importances = dict(sorted(
            feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
            
        if top_k is not None:
            return dict(list(sorted_importances.items())[:top_k])
        return sorted_importances

    def score(self, X, y):
        return score(self, X, y)
    
    def optimize(
        self, 
        X, 
        y, 
        n_trials=100,
        test_size=0.2,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        storage_url: Optional[str] = "sqlite:///boosting_optuna.db",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        study_name: str = "boosting_optimization"
    ):
        if eval_set is not None:
            X_train, y_train = X, y
            X_val, y_val = eval_set
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if storage_url:
            storage = RDBStorage(storage_url)
        else:
            storage = None
        
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,  
                interval_steps=5
            )
        
        def objective(trial):
            boosting_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'rsm': trial.suggest_float('rsm', 0.1, 1.0),
            }
            
            base_model_params = {}
            if self.base_model_class == DecisionTreeRegressor:
                base_model_params.update({
                    'max_depth': trial.suggest_int('max_depth', 2, 8),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                })
            
            if self.goss:
                boosting_params.update({
                    'goss_k': trial.suggest_float('goss_k', 0.1, 0.5),
                    'goss_subsample': trial.suggest_float('goss_subsample', 0.1, 0.5),
                })
            
            if self.dart:
                boosting_params.update({
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.5),
                })
            
            model = Boosting(
                base_model_class=self.base_model_class,
                base_model_params=base_model_params, 
                goss=self.goss,
                bootstrap_type=self.bootstrap_type,
                quantization_type=self.quantization_type,
                dart=self.dart,
                random_state=42,
                **boosting_params
            )
            
            history = {'val_loss': [], 'iterations': []}
            
            def report_callback(iteration, train_loss, val_loss):
                if val_loss is not None:
                    trial.report(val_loss, step=iteration)
                    history['val_loss'].append(val_loss)
                    history['iterations'].append(iteration)
                    
                    if trial.should_prune():
                        trial.set_user_attr("history", history)
                        trial.set_user_attr("n_estimators_actual", iteration)
                        raise optuna.TrialPruned()
            
            try:
                model.fit(
                    X_train, y_train, 
                    eval_set=(X_val, y_val),
                    callback=report_callback
                )
            except optuna.TrialPruned:
                if history['val_loss']:
                    return history['val_loss'][-1]
                else:
                    return float('inf')

            
            trial.set_user_attr("history", {
                'val_loss': model.history.get("val_loss", []),
                'train_loss': model.history.get("train_loss", []),
                'val_roc_auc': model.history.get("val_roc_auc", [])
            })
            trial.set_user_attr("n_estimators_actual", len(model.models))
            
            if model.history.get("val_loss"):
                return min(model.history["val_loss"])
            else:
                return float('inf')

        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )
            print(f"Загружено существующее исследование '{study_name}' с {len(study.trials)} trials")
        except:
            study = optuna.create_study(
                direction='minimize',
                storage=storage,
                study_name=study_name,
                pruner=pruner
            )
            print(f"Создано новое исследование '{study_name}'")
        
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params_ = study.best_params
        self.study_ = study
        
        print(f"Лучший trial: {study.best_trial.number}")
        print(f"Лучшая валидационная потеря: {study.best_value:.4f}")
        print(f"Лучшие параметры: {study.best_params}")
        
        if self.enable_plot_optimize:
            self.plot_optimization_results()
        
        return study.best_params
    
    def plot_optimization_results(self):
        if not hasattr(self, 'study_'):
            print("Сначала запустите optimize()")
            return
        
        optuna.visualization.plot_optimization_history(self.study_).show()
        optuna.visualization.plot_param_importances(self.study_).show()
        optuna.visualization.plot_contour(self.study_).show()
        
    def plot_history(self, X=None, y=None):
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        if "val_loss" in self.history:
            plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.title("Loss History")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        if "val_roc_auc" in self.history:
            plt.plot(self.history["val_roc_auc"], label="Validation ROC-AUC", color="green")
        plt.title("ROC-AUC History")
        plt.xlabel("Iteration")
        plt.ylabel("ROC-AUC")
        plt.legend()
        plt.grid(True)
        
        if hasattr(self, 'study_') and self.enable_plot_optimize:
            plt.subplot(1, 3, 3)
            best_trial = self.study_.best_trial
            plt.text(0.1, 0.5, f"Best trial: {best_trial.number}\nBest value: {best_trial.value:.4f}", 
                    fontsize=12, transform=plt.gca().transAxes)
            plt.axis('off')
            plt.title("Optimization Results")
        
        plt.tight_layout()
        plt.show()