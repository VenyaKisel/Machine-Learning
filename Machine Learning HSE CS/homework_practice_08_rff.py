import numpy as np
from sklearn.svm import SVC, SVR
from typing import Callable
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression, make_friedman1
from catboost import CatBoostClassifier


def estimate_sigma_sq(X, n_pairs = 1000000):
    n_samples = X.shape[0]
        
    max_possible_pairs = n_samples * (n_samples - 1) // 2
    n_pairs = min(n_pairs, max_possible_pairs)
        
    indices_i = np.random.randint(0, n_samples, n_pairs)
    indices_j = np.random.randint(0, n_samples, n_pairs)
        
    mask = indices_i != indices_j
    indices_i = indices_i[mask][:n_pairs]
    indices_j = indices_j[mask][:n_pairs]   
    diffs = X[indices_i] - X[indices_j]
    squared_distances = np.sum(diffs**2, axis=1)
    sigma_sq = np.median(squared_distances)
    return sigma_sq

def estimate_sigma_laplacian(X, n_pairs=1000000):
    n_samples = X.shape[0]
    
    max_possible_pairs = n_samples * (n_samples - 1) // 2
    n_pairs = min(n_pairs, max_possible_pairs)
    
    indices_i = np.random.randint(0, n_samples, n_pairs)
    indices_j = np.random.randint(0, n_samples, n_pairs)
    
    mask = indices_i != indices_j
    indices_i = indices_i[mask][:n_pairs]
    indices_j = indices_j[mask][:n_pairs]   
    
    diffs = X[indices_i] - X[indices_j]
    l1_distances = np.sum(np.abs(diffs), axis=1)
    sigma = np.median(l1_distances)
    return sigma


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X
        
class RandomFeatureCreator(FeatureCreatorPlaceholder):

    def fit(self, X, y=None):
        sigma_sq = estimate_sigma_sq(X)
        self.w = np.random.normal(0, 1 / np.sqrt(sigma_sq), size=(self.n_features, self.new_dim))
        self.b = np.random.uniform(-np.pi, np.pi, size=self.new_dim)
        return self
    
    def transform(self, X, y=None):
        projection = X @ self.w + self.b
        return self.func(projection)


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        sigma = np.sqrt(estimate_sigma_sq(X))
        d = X.shape[1]
        D = self.new_dim
        
        blocks_count = int(np.ceil(D / d))
        
        blocks = []
        for i in range(blocks_count):
            G = np.random.normal(0,1,size = (d,d))
            Q, R = np.linalg.qr(G)
            
            sign_diag = np.diag(np.sign(np.diag(R)))
            Q = Q @ sign_diag
            s_values = np.array([np.linalg.norm(np.random.normal(0,1,d)) for _ in range (d)])
            S = np.diag(s_values)
            W_block = (1 / sigma) * S @ Q
            blocks.append(W_block)
            
        W = np.vstack(blocks)
        
        self.w = W[:D, :].T
        self.b = np.random.uniform(-np.pi, np.pi, size=self.new_dim)
        
        return self

class LaplacianRFF(RandomFeatureCreator):
    def __init__(self, n_features, new_dim, sigma=None, random_state=None, func=np.cos):
        super().__init__(n_features, new_dim, func=func)
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_features = X.shape[1]
        
        if self.sigma is None:
            self.sigma_ = estimate_sigma_laplacian(X)
        else:
            self.sigma_ = self.sigma
        
        self.w = np.random.standard_cauchy(size=(n_features, self.new_dim)) / self.sigma_
        self.b = np.random.uniform(0, 2 * np.pi, size=self.new_dim)
        
        return self
    
class RFFPipeline(BaseEstimator):
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier: str = 'logreg',
            classifier_params=None,
            feature_creator_params=None,
            func=np.cos,
    ):
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.feature_creator_class = feature_creator_class
        self.func = func
        
        if classifier_params is None:
            classifier_params = {}
        if classifier == "logreg":
            self.classifier = LogisticRegression(**classifier_params)
        elif classifier == "svm_linear":
            svm_params = classifier_params.copy()
            svm_params['kernel'] = 'linear'
            self.classifier = SVC(**svm_params)
        elif classifier == "svm_rbf":
            svm_params = classifier_params.copy()
            svm_params['kernel'] = 'rbf'
            self.classifier = SVC(**svm_params)
        elif classifier == "catboost":
            self.classifier = CatBoostClassifier(**classifier_params, verbose=False)
        
        if feature_creator_params is None:
            feature_creator_params = {}
        self.feature_creator_params = feature_creator_params
        
        self.pipeline = None

    def fit(self, X, y):
        pipeline_steps = []
        pipeline_steps.append(('scaler', StandardScaler()))
        
        if self.use_PCA:
            pca_transformer = PCA(n_components=min(self.new_dim, X.shape[1]))
            pipeline_steps.append(('pca', pca_transformer))
            rff_input_dim = self.new_dim
        else:
            rff_input_dim = X.shape[1]
        
        feature_creator = self.feature_creator_class(
            n_features=rff_input_dim,
            new_dim=self.n_features,
            func=self.func,
            **self.feature_creator_params
        )
        
        pipeline_steps.append(('feature_creator', feature_creator))
        pipeline_steps.append(('classifier', self.classifier))
        
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self
        
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
    
    def score(self, X, y):
        return self.pipeline.score(X, y)