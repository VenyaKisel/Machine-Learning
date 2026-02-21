from joblib import Parallel, delayed
import numpy as np

def c_factor(n):
    """
    Computes average path length for an unsuccessful search in a binary search tree.
    Params:
        n: int - number of data points for BST
    Returns:
        float - average path length
    """
    if n <= 1:
        return 0
    harmonic = np.log(n - 1) + 0.5772156649 
    return 2 * harmonic - (2 * (n - 1) / n)

def calc_height(X, depth, node):
    """
    Calculates anomaly scores for sample in a recursive manner.
    Params:
        X: np.array - current sample, available to node
        depth: int - path length up to current node
        node: Node - current tree node
    Returns:
        scores: np.array - path lengths for each sample
    """
    if node.kind == 'external':
        return depth + c_factor(node.size)

    side = np.dot(X, node.w) + node.b > 0
    scores = np.zeros(X.shape[0])
    if np.any(side):
        scores[side] = calc_height(X[side], depth + 1, node.right)
    if np.any(~side):
        scores[~side] = calc_height(X[~side], depth + 1, node.left)
    return scores

def _build_tree(X, subsample_size, max_depth, random_state=None):
    """Вспомогательная функция для построения одного дерева на случайной подвыборке."""
    if random_state is not None:
        np.random.seed(random_state) 
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=subsample_size, replace=False)
    X_sub = X[indices]
    return RandomizedTree(X_sub, max_depth)


class Node(object):
    """
    A single node object for each tree. Contains information on height, current data,
    splitting hyperplane and children nodes.
    
    Attributes:
        X: np.array - data available to current node
        size: int - length of available data
        
        depth: int - depth of node

        left: Node - left child
        right: Node - right child

        kind: str - either "internal" or "external", indicates the type of current node

        w: np.array - normal vector for the splitting hyperplane
        b: float - intercept term for the splitting hyperplane
    """
    def __init__(self, X, depth, left, right, kind, w, b):
        """
        Node(h, left, right, kind, w, b)
        Represents the node object.
        
        Params:
            X: np.array - data available to current node
            depth: int - depth of node
            
            left: Node - left child
            right: Node - right child
            
            kind: str - either "internal" or "external", indicates the type of current node
            
            w: np.array - normal vector for the splitting hyperplane
            b: float - intercept term for the splitting hyperplane
            
        """
        self.size = len(X)
        
        self.depth = depth
        
        self.left = left
        self.right = right
        
        self.kind = kind
    
        self.w = w
        self.b = b
    
    def __repr__(self):
        """
        For convenience only.
        """
        return f"Node(size={self.size}, depth={self.depth}, kind={self.kind})"
    
    
class RandomizedTree(object):
    def __init__(self, X, max_depth):
        self.depth = 0
        self.max_depth = max_depth
        self.internal_count = 0
        self.external_count = 0
        self.root = self.grow(X, 0)

    def grow(self, X, depth):
        """
        Grow tree recursively.
        """
        n, d = X.shape
        if depth >= self.max_depth or n <= 1:
            self.external_count += 1
            return Node(X, depth, None, None, 'external', None, None)

        w = np.random.randn(d)
        p = X[np.random.randint(n)]
        b = -np.dot(w, p)

        left_mask = np.dot(X, w) + b <= 0
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            self.external_count += 1
            return Node(X, depth, None, None, 'external', None, None)

        left_child = self.grow(X[left_mask], depth + 1)
        right_child = self.grow(X[right_mask], depth + 1)

        self.internal_count += 1
        return Node(X, depth, left_child, right_child, 'internal', w, b)

    def score_samples(self, X):
        """
        Calculate anomaly scores for given data.
        """
        return calc_height(X, 0, self.root)
        
    
class ExtendedIsolationForest(object):
    def __init__(self, n_trees, subsample_rate, max_depth=None, contamination=0.01):
        self.n_trees = n_trees
        self.subsample_rate = subsample_rate
        self.max_depth = max_depth
        self.contamination = contamination
        self.trees = []
        self.is_fit = False

    def fit(self, X, n_jobs=-1):
            """
            Fit EIF to new data.
            Params:
                X: np.array
                n_jobs: int - количество параллельных процессов (-1 означает все доступные ядра)
            """
            n_samples = X.shape[0]
            subsample_size = int(self.subsample_rate * n_samples)
            if subsample_size < 1:
                raise ValueError("subsample_rate слишком мала: выборка пуста")
            self.subsample_size_ = subsample_size

            if self.max_depth is None:
                self.max_depth = int(np.ceil(np.log2(subsample_size)))
            seeds = [np.random.randint(0, 2**31) for _ in range(self.n_trees)]

            self.trees = Parallel(n_jobs=n_jobs)(
                delayed(_build_tree)(X, subsample_size, self.max_depth, seed) 
                for seed in seeds
            )

            self.is_fit = True
            return self

    def score_samples(self, X):
        """
        Estimate (normalized) anomaly score for each given sample.
        """
        if not self.is_fit:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit.")

        depths = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            depths[:, i] = tree.score_samples(X)

        avg_depth = np.mean(depths, axis=1)

        c = c_factor(int(self.subsample_rate * X.shape[0]))
        scores = 2 ** (-avg_depth / c)
        return scores

    def predict(self, X):
        """
        Predict if given samples are outliers (1) or inliers (0).
        """
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return (scores >= threshold).astype(int)