import numpy as np

class TreeNode:
    def __init__(self, feature: int = None, threshold: float = None, left: 'TreeNode' = None, 
                 right: 'TreeNode' = None, value: int = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, X, y, max_depth=10, min_samples_split=5, min_samples_leaf = 1):
        if len(X.shape) == 1:
            self.X = X = X.reshape(-1, 1)
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.min_samples_leaf = min_samples_leaf
        self.feature_importance = np.zeros(self.X.shape[1])
    def _Entropy(self, y):
        # Calculates the entropy of the labels.
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        entropy = -sum([p_i*np.log2(p_i) for p_i in p if p_i != 0])
        return entropy   
    def _IG(self, X, y, H, threshold):
        # Calculates the information gain given a threshold.
        left_mask = X <= threshold
        right_mask = ~left_mask

        left_size = len(y[left_mask])
        right_size = len(y[right_mask])

        if left_size == 0 or right_size == 0:
            return -np.inf

        H_left = self._Entropy(y[left_mask])
        H_right = self._Entropy(y[right_mask])
        H_w = (left_size / len(y)) * H_left + (right_size / len(y)) * H_right
        return H - H_w
    def _find_best_split(self, X, y):
        # Finds the best feature and threshold to split the data.
        H = self._Entropy(y)

        best_feature, best_threshold, best_ig = None, None, -np.inf
        for feature in range(X.shape[1]):
            possible_thresholds = np.unique(np.linspace(
                                    X[:, feature].min(), X[:, feature].max(), num=min(20, len(np.unique(X[:, feature])))))
            for threshold in possible_thresholds:
                ig = self._IG(X[:,feature], y, H, threshold)
                if ig > best_ig:
                    best_feature, best_threshold, best_ig = feature, threshold, ig    
        return {"feature_number": best_feature, "threshold": best_threshold, "best_ig": best_ig}
    def _build_tree(self, X, y, depth=0):
         # Recursively builds the decision tree.
        if len(np.unique(y)) == 1:
            return TreeNode(value=np.unique(y)[0])
        
        if depth >= self.max_depth or len(y) <= self.min_samples_split:
            return TreeNode(value=np.bincount(y).argmax())
        
        best_feature, best_threshold, best_ig = self._find_best_split(X, y).values()
        if best_ig == -np.inf:
            return TreeNode(value=np.bincount(y).argmax()) 
        
        self.feature_importance[best_feature] += best_ig / (depth + 1)
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if len(y[left_mask]) < self.min_samples_leaf or len(y[right_mask]) < self.min_samples_leaf:
            return TreeNode(value=np.bincount(y).argmax())

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(best_feature, best_threshold, left_node, right_node)
    def fit(self):
        # Trains the decision tree.
        self.root = self._build_tree(self.X, self.y)
        total_importance = np.sum(self.feature_importance)
        if total_importance > 0:
            self.feature_importance /= total_importance
    def _predict(self, X):
         # Predicts a single sample.
        tmp = self.root
        while tmp.value is None:
            if X[tmp.feature] <= tmp.threshold:
                tmp = tmp.left
            else:
                tmp = tmp.right
        return tmp.value
    def predict(self, X):
        # Predicts a list of samples.
        if len(X.shape) == 1:
            return self._predict(X)
        else:
            return np.array([self._predict(i) for i in X])