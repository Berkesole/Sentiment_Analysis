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


def conditional_entropy(attr, y_label, split):  # 计算某一列特征的条件熵
    # 只分为 小于等于 和 大于
    # (-inf, split], (split, inf)
    enti = len(y_label[attr <= split]) / len(y_label) * entropy(y_label[attr <= split]) \
           + len(y_label[attr > split]) / len(y_label) * entropy(y_label[attr > split])
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
        best_feature = cols[0]  # 最优特征的索引值，初始化为cols[0]（因为可能出现 剩下的都一样，导致熵都一样，不能初始化为-1）
        # best_split = None
        init_coli = sorted(X[:, cols[0]])
        best_split = init_coli[int(len(init_coli) / 2)] if len(init_coli) % 2 == 1 \
            else (init_coli[int(len(init_coli) / 2) - 1] + init_coli[int(len(init_coli) / 2)])/2

        for i in cols:
            coli = X[:, i]  # 拿到第 i 个特征的数据
            sorted_coli = sorted(coli)  # 排序一下，以每个节点作为分割点
            l = len(sorted_coli)

            # 取中位数作为分割点
            split = sorted_coli[int(l / 2)] if l % 2 == 1 \
                else (sorted_coli[int(l / 2) - 1] + sorted_coli[int(l / 2)])/2

            # split = sorted_coli[int(len(sorted_coli) / 2) - 1]
            # 对于每个 split 点，可以划分为 (-inf, split], (split, inf)
            # 计算条件熵
            enti = conditional_entropy(coli, y, split)

            # 信息增益 = base_entropy - 条件熵
            info_gain = base_entropy - enti

            if info_gain > best_gain:
                best_feature = i
                best_gain = info_gain
                best_split = split

        # 根据最小熵特征有几个值，就生成几个新的子树分支
        newtree = {}
        mincol = X[:, best_feature]
        cols.remove(best_feature)  # 剩余的特征

        # !!!!!!!!!!!!!!!!!!!!!!!!
        # 这里说明，X完全一样，但y不一样，已经完全分不开了
        # 即有一边已经是空的了
        if len(y[mincol <= best_split]) == 0 or len(y[mincol > best_split]) == 0:
            counter = Counter(y[mincol <= best_split]).most_common()

            newtree['leq_' + str(best_split)] = counter[0][0]
            newtree['g_' + str(best_split)] = counter[1][0]
            return {best_feature: newtree}

        # 针对最优特征的每个值，进一步划分树
        # (-inf, split] 部分
        entd = entropy(y[mincol <= best_split])  # 计算信息熵
        if entd < 1e-10:  # 已经完全分开
            newtree['leq_' + str(best_split)] = y[mincol <= best_split][0]
        else:  # 还需要进一步细分
            newtree['leq_' + str(best_split)] = self._genTree(cols.copy(), X[mincol <= best_split, :],
                                                              y[mincol <= best_split])

        # (split, inf) 部分
        entd = entropy(y[mincol > best_split])  # 计算信息熵
        if entd < 1e-10:  # 已经完全分开
            newtree['g_' + str(best_split)] = y[mincol > best_split][0]
        else:  # 还需要进一步细分
            newtree['g_' + str(best_split)] = self._genTree(cols.copy(), X[mincol > best_split, :],
                                                            y[mincol > best_split])

        return {best_feature: newtree}  # 将列号作为索引，返回新生成的树

    # 预测新样本
    def predict(self, X):
        X = X.tolist()
        y = [None for i in range(len(X))]
        for i in range(len(X)):  # 对于每一个样本
            predictDict = self.tree
            while predictDict != 1 and predictDict != 0:  # 还未到达叶子结点
                col = list(predictDict.keys())[0]  # 当前所在的特征（一般这个 list 也就1个值）
                predictDict = predictDict[col]

                options = list(predictDict.keys())  # 只有两个，一个是 leq_xxx, 一个是 g_xxx
                split = float(options[0].split('_')[1])
                if X[i][col] <= split:
                    predictDict = predictDict[options[0]]
                else:
                    predictDict = predictDict[options[1]]

            else:
                y[i] = predictDict
        return y


if __name__ == '__main__':
    X = np.array([[0.25163243, 0.36360169, 1.26892475, 0.02173185, 0.22489555, -0.97398363, 0.95891133, 1.43049426,
                   -1.22867934, 0.29061654],
                  [-0.0100559, 0.58408389, 1.11175102, -0.54805864, 0.44584558, -0.5110477, 1.40617181, 0.74466494,
                   -0.9892136, -0.6554191],
                  [0.09104865, 0.67684912, 0.42635433, -0.33562513, 0.75172201, -0.84384867, 1.41826594, 1.03892271,
                   -1.29144398, 0.00988601],
                  [0.17826754, -0.51559839, 0.85877631, -1.31270313, -0.23317759, 0.299095, 0.35394064, 1.88326949,
                   -1.37625839, -0.34602718],
                  [0.59814797, -0.79825217, -0.20303472, -0.30252622, -0.08623843, 0.74611709, 1.34464787, 0.97553973,
                   -1.36859827, -0.17111981],
                  [-0.24847772, -0.08508921, -0.0667594, 0.02909157, 0.20265033, 0.08229019, 1.36755766, 1.27237682,
                   -0.97243795, -0.25036014],
                  [-0.18382561, 0.62702304, 0.19292908, -0.25554506, 0.02711287, -0.59721688, 2.013507, 0.72037058,
                   -1.11669703, -0.17413173],
                  [0.27186938, -0.51699083, -0.70873227, -0.7568591, -0.29114804, 0.39320196, 2.27653427, 0.47748754,
                   -1.31001918, -0.99783882],
                  [-0.16771908, -0.79952519, 1.71429803, -0.78922997, 0.62139708, -1.52931258, 0.98340571, 1.73059296,
                   -1.44220974, 0.22348308],
                  [-0.1565649, -0.07535878, 0.13212344, -0.09579068, 0.36581285, -0.36910951, 1.34813892, 1.38337072,
                   -0.99153516, -0.23920818]])
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1])

    dt = DecisionTree()
    dt.fit(X, y)
    print(dt.tree)
    print(dt.predict(X))
