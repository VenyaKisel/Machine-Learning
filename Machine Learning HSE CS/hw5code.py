import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from collections import Counter


def find_best_split(feature_vector, target_vector):

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]
    
    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    
    unique_threshold_indices = np.where(np.diff(sorted_features) != 0)[0]
    thresholds = thresholds[unique_threshold_indices]
    
    cumsum_left = np.cumsum(sorted_targets[:-1])[unique_threshold_indices] 
    count_left = unique_threshold_indices + 1
    p1_left = cumsum_left / count_left
    p0_left = 1 - p1_left
    H_left = 1 - p1_left**2 - p0_left**2
    
    total_1 = np.sum(sorted_targets)
    total_0 = len(sorted_targets) - total_1
    
    cumsum_right = total_1 - cumsum_left
    count_right = len(sorted_targets) - count_left
    
    p1_right = cumsum_right / count_right
    p0_right = 1 - p1_right
    H_right = 1 - p1_right**2 - p0_right**2
    
    ginis = -(count_left/len(target_vector))*H_left - (count_right/len(target_vector))*H_right
    
    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        
        # тут у нас хранятся дерево и параметры
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth = 0):
        # здесь критерий остановы и была ошибка sub_y != sub_y[0], проверка что все элементы одинаковы
        if (self._max_depth is not None and depth >= self._max_depth) or \
           (self._min_samples_split is not None and len(sub_y) < self._min_samples_split) or \
           (self._min_samples_leaf is not None and len(sub_y) < 2 * self._min_samples_leaf) or \
           np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        # выбор признака для разбиения
        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real": # проверка является ли признак непрерывным
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical": # или категориальным
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = (current_click + 1e-6) / (current_count + 1e-6)
                sorted_categories = [k for k, _ in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                    raise ValueError
            if self._min_samples_leaf is not None and len(feature_vector) < 2 * self._min_samples_leaf:
                continue
            if np.unique(feature_vector).size < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini_best < gini: # здесь тоже была ошибка, джинибест был больше, а это неверно
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if self._min_samples_leaf is not None:
                    left_count = np.sum(split)
                    right_count = len(sub_y) - left_count
                    if left_count < self._min_samples_leaf or right_count < self._min_samples_leaf:
                        continue
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat in categories_map if categories_map[cat] < threshold]

                else:
                    raise ValueError
        # если лучшего разбиения не нашлось, выдаем наиболее частый класс
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        # иначе: 
        # нашли лучший узел и рекурсивно делаем подузлы и обучаем
        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best if threshold_best is not None else []
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

        
    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature = node["feature_split"]
        feature_type = self._feature_types[feature]
        
        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        
        elif feature_type == "categorical":
            # Добавляем проверку на None и делаем categories_split по умолчанию пустым списком
            categories = node.get("categories_split", [])
            if x[feature] in categories:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError
    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


class LinearRegressionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None, n_quantiles=10):
        if np.any([ft not in ["real", "categorical"] for ft in feature_types]):
            raise ValueError("Unknown feature type")
            
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self.n_quantiles = n_quantiles

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        # Условия остановки рекурсии
        stop_condition = (
            (self._max_depth is not None and depth >= self._max_depth) or
            (self._min_samples_split is not None and len(sub_y) < self._min_samples_split) or
            (self._min_samples_leaf is not None and len(sub_y) < self._min_samples_leaf)
        )
        
        if stop_condition:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return
            
        best_feature, best_threshold, best_loss, best_split = None, None, float('inf'), None
        
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                # Преобразование категориальных признаков в числовые (по среднему значению)
                categories = {}
                for cat in np.unique(sub_X[:, feature]):
                    mask = (sub_X[:, feature] == cat)
                    if mask.sum() > 0:
                        categories[cat] = np.mean(sub_y[mask])
                feature_vector = np.array([categories[x] for x in sub_X[:, feature]])
            
            # Пропуск признаков с недостаточной вариативностью
            if np.unique(feature_vector).size < 2:
                continue
                
            # Выбор порогов по квантилям
            quantiles = np.linspace(0, 1, self.n_quantiles+2)[1:-1]
            thresholds = np.unique(np.quantile(feature_vector, quantiles))
            
            for threshold in thresholds:
                split = feature_vector < threshold
                n_left = np.sum(split)
                n_right = len(sub_y) - n_left
                
                # Проверка min_samples_leaf
                if self._min_samples_leaf is not None:
                    if n_left < self._min_samples_leaf or n_right < self._min_samples_leaf:
                        continue
                
                X_left, y_left = sub_X[split], sub_y[split]
                X_right, y_right = sub_X[~split], sub_y[~split]
                
                # Обучение моделей для левой и правой частей
                model_left = LinearRegression().fit(X_left, y_left)
                model_right = LinearRegression().fit(X_right, y_right)
                
                # Расчет ошибки для частей
                mse_left = mean_squared_error(y_left, model_left.predict(X_left))
                mse_right = mean_squared_error(y_right, model_right.predict(X_right))
                
                # Взвешенная ошибка
                total_loss = (n_left * mse_left + n_right * mse_right) / len(sub_y)
                
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature
                    best_threshold = threshold
                    best_split = split
                    best_feature_type = feature_type
        
        # Если не найдено подходящее разбиение
        if best_feature is None:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return
        
        # Сохранение лучшего разбиения
        node["type"] = "nonterminal"
        node["feature_split"] = best_feature
        node["threshold"] = best_threshold
        node["feature_type"] = best_feature_type
        
        # Для категориальных признаков сохраняем карту преобразования
        if best_feature_type == "categorical":
            categories = {}
            for cat in np.unique(sub_X[:, best_feature]):
                mask = (sub_X[:, best_feature] == cat)
                if mask.sum() > 0:
                    categories[cat] = np.mean(sub_y[mask])
            node["categories_map"] = categories
        
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[best_split], sub_y[best_split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~best_split], sub_y[~best_split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict([x])[0]
        
        feature = node["feature_split"]
        feature_type = node["feature_type"]
        
        if feature_type == "real":
            value = x[feature]
        elif feature_type == "categorical":
            value = node["categories_map"].get(x[feature], 0)
        
        if value < node["threshold"]:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])
    
    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree, 0)
    
    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])