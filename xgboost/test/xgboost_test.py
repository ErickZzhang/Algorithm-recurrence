import numpy as np

from ..XgboostReg import XgboostReg


def xgb_reg_test():
    x1 = np.arange(8,108)
    x2 = np.arange(30,130)*2
    x3 = np.arange(-33,100-33)*3
    y = 8*x1-4*x2+16*x3 + np.random.normal(0, 20, 100)
    x1 = x1 + np.random.normal(0, scale=3, size=100)
    x2 = x2 + np.random.normal(0, scale=3, size=100)
    x3 = x3 + np.random.normal(0, scale=2, size=100)
    x1, x2, x3 = x1.reshape((100,1)), x2.reshape((100,1)), x3.reshape((100,1))
    x = np.concatenate((x1, x2, x3), axis=1)
    reg = XgboostReg()
    reg.fit(x, y)
    pred = reg.predict(x)

    temp = reg.trees[0].predict(x)
    # cal R^2拟合优度 goodness of fit
    y_mean = np.mean(y)
    R_2 = np.sum((pred-y_mean)**2) / np.sum((y-y_mean)**2)

    print("goodness of fit: {}".format(R_2))
