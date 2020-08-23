import numpy as np

from .XgboostBase import Boost_DTree

class XgboostReg():
    def __init__(self, method='bdtree'):
        super().__init__()
        self.trees = None
        self.lr = None
        self.method = method

    def _get_g(self, y, y_hat):
        return (y_hat - y)
    
    def _get_h(self, y, y_hat):
        return np.ones(y.shape[0])

    def fit(self, X, y, n_estimators=100, lr=0.1, gamma=0.1, lam=0.1, max_depth=8, min_sample_split=2, subsample=False):
        assert isinstance(X, np.ndarray), len(X.shape)==2
        assert type(y) == np.ndarray and y.shape[0] == X.shape[0] and len(y.shape) <= 2
        y = y.flatten()

        y_hat = np.zeros(y.shape[0])
        self.trees = []
        self.lr = lr
        for n_esti in range(n_estimators):
            idx = np.arange(X.shape[0])
            if subsample:
                k = int(subsample * X.shape[0])
                np.random.shuffle(idx)
                idx = idx[: k]    
            
            X_train = X[idx]
            y_train = y[idx]
            g = self._get_g(y_train, y_hat[idx])
            h = self._get_h(y_train, y_hat[idx])

            bdtree = Boost_DTree()
            bdtree.fit(X_train, y_train, g, h, lam, gamma, max_depth, min_sample_split)
            self.trees.append(bdtree)

            y_hat = lr * bdtree.predict(X)

    def _predict_row(self, x):
        return self.lr * np.sum(tree._predict_row(x) for tree in self.trees)

    def predict(self, X):
        assert type(X) == np.ndarray and len(X.shape) == 2
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred[i] = self._predict_row(X[i])
        return pred

