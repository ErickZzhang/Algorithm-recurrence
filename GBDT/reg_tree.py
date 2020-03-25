import numpy as np

class Node():
    def __init__(self, score=None):
        super().__init__()
        self.score = score
        self.left = None
        self.right = None
        self.split = None
        self.feature = None

class Regression_Tree():
    def __init__(self):
        super().__init__()
        self.root = Node()
        self.height = 0

    def _cal_split_mse(self, X, y, idx, split, feature):
        split_sum = [0., 0.]
        split_sqr_sum = [0., 0.]
        split_count = [0., 0.]
        for i in idx:
            if X[i, feature] < split:
                split_count[0] += 1
                split_sum[0] += y[i]
                split_sqr_sum[0] += y[i]**2
            else:
                split_count[1] += 1
                split_sum[1] += y[i]
                split_sqr_sum[1] += y[i]**2
        split_avg = [split_sum[0]/split_count[0], split_sum[1]/split_count[1]]
        split_mse = [np.sum(split_sqr_sum[0]) - split_sum[0]*split_avg[0],
                     np.sum(split_sqr_sum[1]) - split_sum[1]*split_avg[1]]
        mse = split_mse[0] + split_mse[1]
        return mse, split, split_avg

    def _choose_split_point(self, X, y, idx, feature):
        unique = set(X[idx, feature])
        if len(unique) == 1:
            return None

        unique.remove(min(unique)) # 按值最小的点划分等于不划分

        mse, split, split_avg = min((self._cal_split_mse(X, y, idx, split, feature) for split in unique), key=lambda x: x[0])
        return mse, split, feature, split_avg

    def _choose_split_feature(self, X, y, idx):
        choose_set = [m for m in map(lambda x: self._choose_split_point(X, y, idx, x), range(X.shape[1])) if m != None]
        if choose_set == []:
            return None
        mse, split, feature, split_avg = min(choose_set, key=lambda x: x[0])
        idx_split = [[], []]
        for id in idx:
            if X[id, feature] < split:
                idx_split[0].append(id)
            else:
                idx_split[1].append(id)
        return feature, split, idx_split, split_avg

    def fit(self, X, y, max_depth=5, min_samples_split=2):
        assert isinstance(X, np.ndarray), len(X.shape)==2
        assert type(y) == np.ndarray and y.shape[0] == X.shape[0] and len(y.shape) <= 2
        y = y.flatten()
        queue = [[0, self.root, np.arange(X.shape[0])]]
        while queue:
            depth, root_node, idx = queue.pop(0)

            if depth == max_depth: # 说明之后的depth都==max_depth
                break

            if len(idx) < min_samples_split or set(y[idx]) == 1:
                # 小于最小分裂个数或者y值一样
                continue

            split_rule = self._choose_split_feature(X, y, idx)
            # return feature, split, idx_split, split_avg
            if split_rule == None:
                continue
            root_node.feature = split_rule[0]
            root_node.split = split_rule[1]
            root_node.left = Node(split_rule[3][0])
            root_node.right = Node(split_rule[3][1])
            queue.append([depth+1, root_node.left, split_rule[2][0]])
            queue.append([depth+1, root_node.right, split_rule[2][1]])

        self.height = depth

    def _predcit_row(self, x):
        temp_node = self.root
        while temp_node.feature != None:
            if x[temp_node.feature] < temp_node.split:
                temp_node = temp_node.left
            else:
                temp_node = temp_node.right
        return temp_node.score

    def predict(self, X):
        assert type(X) == np.ndarray and len(X.shape) == 2
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred[i] = self._predcit_row(x[i])
        return pred

if __name__ == "__main__":
    x1 = np.arange(8,108)
    x2 = np.arange(30,130)*2
    x3 = np.arange(-33,100-33)*3
    y = 8*x1-4*x2+16*x3 + np.random.normal(0, 10, 100)
    x1 = x1 + np.random.normal(0, scale=3, size=100)
    x2 = x2 + np.random.normal(0, scale=3, size=100)
    x3 = x3 + np.random.normal(0, scale=2, size=100)
    x1, x2, x3 = x1.reshape((100,1)), x2.reshape((100,1)), x3.reshape((100,1))
    x = np.concatenate((x1, x2, x3), axis=1)
    reg_tree = Regression_Tree()
    reg_tree.fit(x, y)
    pred = reg_tree.predict(x)

    # cal R^2拟合优度 goodness of fit
    y_mean = np.mean(y)
    R_2 = np.sum((pred-y_mean)**2) / np.sum((y-y_mean)**2)

    print("goodness of fit: {}".format(R_2))


