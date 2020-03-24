import numpy as np

def sigmoid(x):
    if isinstance(x, (float, np.ndarray)):
        return 1 / (1 + np.exp(-x))
    if isinstance(x, list):
        return 1 / (1 + np.exp(-np.array(x)))
    raise ValueError("x")

class Log_reg():
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def fit(self, X, y):
        assert type(X) == np.ndarray and len(X.shape) == 2
        assert type(y) == np.ndarray and y.shape[0] == X.shape[0] and len(y.shape) <= 2
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        if len(y.shape) == 1:
            y = np.reshape(y, (y.shape[0], 1))
        self.W = np.random.rand(X.shape[1], 1)
        n = X.shape[0]

        for i in range(1000):
            self.W -= X.T.dot(sigmoid(X.dot(self.W)) - y) / n * self.lr

    def predict(self, X):
        assert type(X) == np.ndarray and len(X.shape) == 2
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        assert X.shape[1] == self.W.shape[0]
        return sigmoid(X.dot(self.W))

if __name__ == "__main__":
    postive_point = np.zeros((300, 2))
    negative_point = np.zeros((300, 2))
    postive_point[:, 0] = np.random.normal(4, 1, 300)
    negative_point[:, 0] = np.random.normal(8, 1, 300)
    postive_point[:, 1] = np.random.normal(8, scale=2, size=300)
    negative_point[:, 1] = np.random.normal(12,1, 300)
    y = np.concatenate((np.ones((300)), np.zeros((300))))
    X = np.concatenate((postive_point, negative_point))
    log_reg = Log_reg()
    log_reg.fit(X, y)
    predict_X = log_reg.predict(X)
    print("acc: {}".format((sum(predict_X[:300]>=0.5) + sum(predict_X[300:]<0.5)) / 600))
