import numpy as np

class Node():
    def __init__(self, score=None):
        super().__init__()
        self.score = score
        self.left = None
        self.right = None
        self.split = None
        self.feature = None

class Boost_DTree():
    def __init__(self):
        super().__init__()
        self.root = Node()
        self.height = 0

    def _cal_split_gain(self, X, y, idx, split, feature):
        """计算当前分割的gain 
        
        Parameters
        ----------
        X : ndarray
            train set
        y : ndarray
            target set
        idx : ndarray or listt
            当前要分割的index集合
        split : float
            当前分割的点
        feature : int
            当前分割的feature

        Returns
        -------
        float   
            在idx子集上，按照split, feature进行分割后得到的增益gain
        """

        GL, GR, HL, HR = 0., 0., 0., 0.
        split_sum = [0., 0.]
        split_count = [0, 0]
        for i in idx:
            if X[i, feature] < split:
                GL += self.g[i]
                HL += self.h[i]
                split_sum[0] += y[i]
                split_count[0] += 1
            else:
                GR += self.g[i]
                HR += self.h[i]
                split_sum[1] += y[i]
                split_count[1] += 1
        gain = (GL**2/(HL+self.lam) + GR**2/(HR+self.lam) - (GL+GR)**2/(HR+HL+self.lam))/2 - self.gamma
        split_avg = [split_sum[0]/split_count[0], split_sum[1]/split_count[1]]
        return gain, split, split_avg

    def _choose_split_point(self, X, y, idx, feature):
        """ 计算按照feature进行分割的最佳分割点 
        
        Parameters
        ----------
        X : ndarray
            train set
        y : ndarray
            target set
        idx : ndarray or listt
            当前要分割的index集合
        feature : int
            当前分割的feature
        
        Returns
        -------
           
            gain, split, split_avg or None
            如果无法分割返回None
            如果可以分割，最佳分割的gain, split_point, 分割后左右节点的平均值
        """
        unique = set(X[idx, feature])
        if len(unique) == 1:
            return None

        unique.remove(min(unique)) # 按值最小的点划分等于不划分

        gain, split, split_avg = max((self._cal_split_gain(X, y, idx, split, feature) for split in unique),
                                      key=lambda x: x[0])
        return gain, split, feature, split_avg

    def _choose_split_feature(self, X, y, idx):
        """ 选择最佳分割feature
        
        Parameters
        ----------
        X : ndarray
            train set
        y : ndarray
            target set
        idx : ndarray or listt
            当前要分割的index集合
        
        Returns
        -------
         feature, split, idx-split, split-avg
            最佳分割的feature, 该feature上的最佳分割点，分割后的左右idx, 左右子树的均值
        """
        choose_set = [m for m in map(lambda x: self._choose_split_point(X, y, idx, x),
                                     range(X.shape[1])) if m != None]
        if choose_set == []:
            return None
        mse, split, feature, split_avg = max(choose_set, key=lambda x: x[0])
        idx_split = [[], []]
        for id in idx:
            if X[id, feature] < split:
                idx_split[0].append(id)
            else:
                idx_split[1].append(id)
        return feature, split, idx_split, split_avg

    def fit(self, X, y, g, h, lam, gamma, max_depth=8, min_samples_split=2):
        """[summary]
        
        Parameters
        ----------
        X : ndarray
            train set
        y : ndarray
            target set
        g : ndarray
            一阶导数
        h : ndarray
            二阶导数
        lam : float
            l2正则化系数
        gamma : float
            叶子节点惩罚项
        max_depth : int, optional
            [description], by default 5
        min_samples_split : int, optional
            [description], by default 2
        
        Returns
        -------
        [type]
            [description]
        """

        assert isinstance(X, np.ndarray), len(X.shape)==2
        assert type(y) == np.ndarray and y.shape[0] == X.shape[0] and len(y.shape) <= 2
        y = y.flatten()
        queue = [[0, self.root, np.arange(X.shape[0])]]

        self.g, self.h, self.lam, self.gamma = g, h, lam, gamma

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

    def _predict_row(self, x):
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
            pred[i] = self._predict_row(X[i])
        return pred
