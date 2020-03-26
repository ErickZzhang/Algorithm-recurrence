import numpy as np

from .reg_tree import Regression_Tree


class GradientBoostingBase():
    def __init__(self):
        super().__init__()
        self.trees = None
        self.lr = None
        self.initval = None
        self.fn = lambda x: x

    def _cal_initval(self, y):
        return sum(y) / y.shape[0]

    def _cal_residuals(self, y, y_hat):
        return y - y_hat

    def fit(self, X, y, n_estimators=50, lr=0.1, max_depth=5, min_samples_split=2, subsample=False, initial=None):
        assert isinstance(X, np.ndarray), len(X.shape)==2
        assert type(y) == np.ndarray and y.shape[0] == X.shape[0] and len(y.shape) <= 2
        y = y.flatten()
        
        if initial:
            if initial == 'zeros':
                self.initval = 0.
        else:
            # default avg initial
            self.initval = self._cal_initval(y)

        y_hat = np.array([self.initval]).repeat(X.shape[0])

        self.trees = []
        self.lr = lr
        for n_est in range(n_estimators):
            idx = np.arange(X.shape[0])
            if subsample:
                k = int(subsample * X.shape[0])
                np.random.shuffle(idx)
                idx = idx[: k]    

            X_train = X[idx]
            y_train = y[idx]
            y_hat_train = y_hat[idx]
            residuals_train = self._cal_residuals(y_train, y_hat_train)    

            reg_tree = Regression_Tree()
            reg_tree.fit(X_train, residuals_train, max_depth=max_depth, min_samples_split=min_samples_split)
            self.trees.append(reg_tree)

            y_hat += lr * reg_tree.predict(X)

    def _predict_row(self, row):
        return self.fn(self.initval + self.lr * np.sum(tree._predict_row(row) for tree in self.trees))

    def predict(self, X):
        assert type(X) == np.ndarray and len(X.shape) == 2
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred[i] = self._predict_row(X[i])
        return pred
