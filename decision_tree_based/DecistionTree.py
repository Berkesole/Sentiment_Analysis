import numpy as np
from collections import Counter
from math import log2


def entropy(y_label):
    counter = Counter(y_label)
    ent = 0.0
    for num in counter.values():
        p = num / len(y_label)
        ent += -p * log2(p)
    return ent


def conditional_entropy(attr, y_label):  # 计算某一列特征的条件熵
    enti = 0.0
    # 计算条件熵
    for d in set(attr):  # 对于在这个特征下的每个取值
        sub_ent = len(y_label[attr == d]) / len(y_label) * entropy(y_label[attr == d])
        enti += sub_ent
    return enti


class DecisionTree:
    def __init__(self):
        self.tree = {}

    # 训练决策树
    def fit(self, X, y):
        cols = list(range(X.shape[1]))  # cols: [0 ~ 特征的维度)
        # 对X得每一列数据，计算分割后得信息熵
        self.tree = self._genTree(cols, X, y)

    # 递归生成决策树
    def _genTree(self, cols, X, y):
        base_entropy = entropy(y)  # 只看结果，基础熵
        best_gain = 0.0  # 最大信息增益

        # 初始化
        best_feature = -1  # 最优特征的索引值
        for i in cols:
            coli = X[:, i]  # 拿到第 i 个特征的数据
            # 计算条件熵
            enti = conditional_entropy(coli, y)
            # 信息增益 = base_entropy - 条件熵
            info_gain = base_entropy - enti
            if info_gain > best_gain:
                best_feature = i
                best_gain = info_gain

        # 根据最小熵特征有几个值，就生成几个新的子树分支
        newtree = {}
        mincol = X[:, best_feature]
        cols.remove(best_feature)  # 剩余的特征

        # 针对最优特征的每个值，进一步划分树
        for d in set(mincol):
            entd = entropy(y[mincol == d])  # 计算信息熵
            if entd < 1e-10:  # 已经完全分开
                newtree[d] = y[mincol == d][0]
            else:  # 还需要进一步细分
                # X[mincol == d, :] 指 mincol 列为 d 的所在的前几行
                newtree[d] = self._genTree(cols.copy(), X[mincol == d, :], y[mincol == d])
        return {best_feature: newtree}  # 将列号作为索引，返回新生成的树

    # 预测新样本
    def predict(self, X):
        X = X.tolist()
        y = [None for i in range(len(X))]
        for i in range(len(X)):  # 对于每一个样本
            predictDict = self.tree
            while predictDict != 'Yes' and predictDict != 'No':  # 还未到达叶子结点
                col = list(predictDict.keys())[0]  # 当前所在的特征（一般这个 list 也就1个值）

                predictDict = predictDict[col]

                predictDict = predictDict[X[i][col]]

            else:
                y[i] = predictDict
        return y


if __name__ == '__main__':
    X = np.array([['Yes985', '本科', 'C++'],
                  ['Yes985', '本科', 'Java'],
                  ['No985', '硕士', 'Java'],
                  ['No985', '硕士', 'C++'],
                  ['Yes985', '本科', 'Java'],
                  ['No985', '硕士', 'C++'],
                  ['Yes985', '硕士', 'Java'],
                  ['Yes985', '博士', 'C++'],
                  ['No985', '博士', 'Java'],
                  ['No985', '本科', 'Java']])
    y = np.array(['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No'])

    dt = DecisionTree()
    dt.fit(X, y)
    print(dt.tree)
    print(dt.predict(X))
