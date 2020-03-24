import numpy as np


class SVC():
    def __init__(self, C=0.1, tolar=0.001, max_iter=40, kernel='linear'):
        super().__init__()
        self.C = C
        self.tolar = tolar
        self.maxiter = max_iter
        self.kernel = kernel

    def __kernel_func(self):
        if self.kernel == 'linear':
            pass

    def fit(self, X, y):
        assert isinstance(X, np.ndarray), len(X.shape)==2
        assert type(y) == np.ndarray and y.shape[0] == X.shape[0] and len(y.shape) <= 2
        y = y.flatten()

        n, k = X.shape
        self.alphas = np.zeros((X.shape[0]))
        self.b = 0.
        iter = 0
        total_set = True
        alphas_pair_changed = 0
        gx = np.zeros((X.shape[0]))
        while (iter < self.maxiter) and ((alphas_pair_changed > 0) or total_set):
            alphas_pair_changed = 0
            if total_set:
                for i in range(n):
                    Ei = gx[i] + self.b - y[i] # array
                    if ((y[i]*Ei<-self.tolar) and (self.alphas[i]<self.C)) or ((y[i]*Ei>self.tolar) and (self.alphas[i]>0)):

                        # find j argmax |Ei - Ej|
                        j = np.argmax(np.abs(gx + self.b - y - Ei))
                        Ej = gx[j] + self.b - y[j]

                        # 判断a2new的范围
                        if y[i] != y[j]:
                            L = max(0, self.alphas[j]-self.alphas[i])
                            H = min(self.C , self.C+self.alphas[j]-self.alphas[i])
                        else:
                            L=max(0, self.alphas[j]+self.alphas[i]-self.C) 
                            H=min(self.C, self.alphas[j]+self.alphas[i])
                        if L==H:
                            continue
                            
                        eta = X[i].dot(X[i].T) + X[j].dot(X[j].T) - 2. * X[i].dot(X[j].T)  # kernel!!
                        if eta == 0: # 此项为分母
                            continue

                        a_i_old, a_j_old = self.alphas[i], self.alphas[j]

                        self.alphas[j] += y[j]*(Ei-Ej)/eta # 更新a2
                        # 结合给定的范围更新a2
                        if self.alphas[j] > H: 
                            self.alphas[j] = H
                        elif self.alphas[j] < L:
                            self.alphas[j] = L

                        if np.abs(self.alphas[j] - a_j_old) < 0.00001: # j更新的步长过少
                            continue
                        self.alphas[i] += y[i]*y[j]*(a_j_old - self.alphas[j])

                        if 0 < self.alphas[i] < self.C:
                            self.b = self.b - Ei - y[i]*(self.alphas[i]-a_i_old)*X[i].dot(X[i].T) - y[j]*(self.alphas[j]-a_j_old)*X[i].dot(X[j].T)
                        elif 0 < self.alphas[j] < self.C:
                            self.b = self.b - Ej - y[i]*(self.alphas[i]-a_i_old)*X[i].dot(X[j].T) - y[j]*(self.alphas[j]-a_j_old)*X[j].dot(X[j].T)
                        else:
                            b1 = self.b - Ei - y[i]*(self.alphas[i]-a_i_old)*X[i].dot(X[i].T) - y[j]*(self.alphas[j]-a_j_old)*X[i].dot(X[j].T)
                            b2 = self.b - Ej - y[i]*(self.alphas[i]-a_i_old)*X[i].dot(X[j].T) - y[j]*(self.alphas[j]-a_j_old)*X[j].dot(X[j].T)
                            self.b = (b1 + b2) / 2.

                        # 更新g(x)
                        gx += self.alphas[i]*y[i]*X.dot(X[i].T) + self.alphas[j]*y[j]*X.dot(X[j].T)
                        
                        alphas_pair_changed += 1
                iter += 1
            else:
                bound0_C = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                for i in bound0_C:
                    Ei = gx[i] + self.b - y[i] # array
                    if ((y[i]*Ei<-self.tolar) and (self.alphas[i]<self.C)) or ((y[i]*Ei>self.tolar) and (self.alphas[i]>0)):

                        # find j argmax |Ei - Ej|
                        j = np.argmax(np.abs(gx + self.b - y - Ei))
                        Ej = gx[j] + self.b - y[j]

                        # 判断a2new的范围
                        if y[i] != y[j]:
                            L = max(0, self.alphas[j]-self.alphas[i])
                            H = min(self.C , self.C+self.alphas[j]-self.alphas[i])
                        else:
                            L=max(0, self.alphas[j]+self.alphas[i]-C) 
                            H=min(C, self.alphas[j]+self.alphas[i])
                        if L==H:
                            continue
                            
                        eta = X[i].dot(X[i].T) + X[j].dot(X[j].T) - 2. * X[i].dot(X[j].T)  # kernel!!
                        if eta == 0: # 此项为分母
                            continue

                        a_i_old, a_j_old = self.alphas[i], self.alphas[j]

                        self.alphas[j] += y[j]*(Ei-Ej)/eta # 更新a2
                        # 结合给定的范围更新a2
                        if self.alphas[j] > H: 
                            self.alphas[j] = H
                        elif self.alphas[j] < L:
                            self.alphas[j] = L

                        if np.abs(self.alphas[j] - a_j_old) < 0.00001: # j更新的步长过少
                            continue
                        self.alphas[i] += y[i]*y[j]*(a_j_old - self.alphas[j])

                        if 0 < self.alphas[i] < self.C:
                            self.b = self.b - Ei - y[i]*(self.alphas[i]-a_i_old)*X[i].dot(X[i].T) - y[j]*(self.alphas[j]-a_j_old)*X[i].dot(X[j].T)
                        elif 0 < self.alphas[j] < self.C:
                            self.b = self.b - Ej - y[i]*(self.alphas[i]-a_i_old)*X[i].dot(X[j].T) - y[j]*(self.alphas[j]-a_j_old)*X[j].dot(X[j].T)
                        else:
                            b1 = self.b - Ei - y[i]*(self.alphas[i]-a_i_old)*X[i].dot(X[i].T) - y[j]*(self.alphas[j]-a_j_old)*X[i].dot(X[j].T)
                            b2 = self.b - Ej - y[i]*(self.alphas[i]-a_i_old)*X[i].dot(X[j].T) - y[j]*(self.alphas[j]-a_j_old)*X[j].dot(X[j].T)
                            self.b = (b1 + b2) / 2.

                        # 更新g(x)
                        gx += self.alphas[i]*y[i]*X.dot(X[i].T) + self.alphas[j]*y[j]*X.dot(X[j].T)
                        
                        alphas_pair_changed += 1
                iter += 1
            if total_set:
                total_set = False
            elif alphas_pair_changed == 0:
                total_set = True

        self.w = (self.alphas * y).reshape((1, -1)).dot(X).T # shape k*1

    def predict(self, X):
        assert type(X) == np.ndarray and len(X.shape) == 2
        out = X.dot(self.w) + self.b
        out[out >= 0] = 1
        out[out < 0] = -1
        return out


if __name__ == "__main__":
    postive_point = np.zeros((300, 2))
    negative_point = np.zeros((300, 2))
    postive_point[:, 0] = np.random.normal(4, 1, 300)
    negative_point[:, 0] = np.random.normal(8, 1, 300)
    postive_point[:, 1] = np.random.normal(8, scale=2, size=300)
    negative_point[:, 1] = np.random.normal(12,1, 300)
    y = np.concatenate((np.ones((300)), -np.ones((300))))
    X = np.concatenate((postive_point, negative_point))
    svc = SVC()
    svc.fit(X, y)
    predict_X = svc.predict(X)
    print(sum(predict_X==1))
    print("acc: {}".format((sum(predict_X[:300]==1) + sum(predict_X[300:]==-1)) / 600))