from collections import defaultdict
import numpy as np
import jieba

class MaxEntClassifier:
    def __init__(self, max_iter=500):
        self.feats = defaultdict(int)
        self.labels = {0, 1}
        self.weight = []
        self.max_iter = max_iter

    def prob_weight(self, features, label):
        weight = 0.0
        for feature in features:
            if (label, feature) in self.feats:
                weight += self.weight[self.feats[(label, feature)]]
        return np.exp(weight)

    def calculate_probability(self, features):
        weights = [(self.prob_weight(features, label), label) for label in self.labels]
        try:
            z = sum([weight for weight, label in weights])
            prob = [(weight / z, label) for weight, label in weights]
        except ZeroDivisionError:
            return "collapse"
        return prob

    def convergence(self, last_weight):
        for w1, w2 in zip(last_weight, self.weight):
            if abs(w1 - w2) >= 0.001:
                return False
        return True

    def train(self, train_data, train_data_labels, best_words=None):
        print("MaxEntClassifier is training ...... ")

        # init the parameters
        train_data_length = len(train_data_labels)
        if best_words is None:
            for i in range(train_data_length):
                for word in set(train_data[i]):
                    self.feats[(train_data_labels[i], word)] += 1
        else:
            for i in range(train_data_length):
                for word in set(train_data[i]):
                    if word in best_words:
                        self.feats[(train_data_labels[i], word)] += 1

        the_max = max([len(record) - 1 for record in train_data])  # the_max param for GIS training algorithm
        self.weight = [0.0] * len(self.feats)  # init weight for each feature
        ep_empirical = [0.0] * len(self.feats)  # init the feature expectation on empirical distribution
        for i, f in enumerate(self.feats):
            ep_empirical[i] = self.feats[f] / train_data_length  # feature expectation on empirical distribution
            self.feats[f] = i  # each feature function correspond to id

        for i in range(self.max_iter):
            ep_model = [0.0] * len(self.feats)  # feature expectation on model distribution
            for doc in train_data:
                prob = self.calculate_probability(doc)  # calculate p(y|x)
                if prob == "collapse":
                    print("The program collapse. The iter number: %d." % (i + 1))
                    return
                for feature in doc:
                    for weight, label in prob:
                        if (label, feature) in self.feats:  # only focus on features from training data.
                            idx = self.feats[(label, feature)]  # get feature id
                            ep_model[idx] += weight * (1.0 / train_data_length)  # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N

            last_weight = self.weight[:]
            for j, win in enumerate(self.weight):
                delta = 1.0 / the_max * np.log(ep_empirical[j] / ep_model[j])
                self.weight[j] += delta  # update weight

            # test if the algorithm is convergence
            if self.convergence(last_weight):
                print("The program convergence. The iter number: %d." % (i + 1))
                break

        print("MaxEntClassifier trains over!")

    def test(self, train_data, train_labels, best_words, test_data):
        classify_results = []

        # init the parameters
        train_data_length = len(train_labels)
        if best_words is None:
            for i in range(train_data_length):
                for word in set(train_data[i]):
                    self.feats[(train_labels[i], word)] += 1
        else:
            for i in range(train_data_length):
                for word in set(train_data[i]):
                    if word in best_words:
                        self.feats[(train_labels[i], word)] += 1

        the_max = max([len(record) - 1 for record in train_data])  # the_max param for GIS training algorithm
        self.weight = [0.0] * len(self.feats)  # init weight for each feature
        ep_empirical = [0.0] * len(self.feats)  # init the feature expectation on empirical distribution
        for i, f in enumerate(self.feats):
            ep_empirical[i] = self.feats[f] / train_data_length  # feature expectation on empirical distribution
            self.feats[f] = i  # each feature function correspond to id

        for i in range(self.max_iter):
            print("MaxEntClassifier is training ...... ")

            ep_model = [0.0] * len(self.feats)  # feature expectation on model distribution
            for doc in train_data:
                prob = self.calculate_probability(doc)  # calculate p(y|x)
                if prob == "collapse":
                    print("The program collapse. The iter number: %d." % (i + 1))
                    return
                for feature in doc:
                    for weight, label in prob:
                        if (label, feature) in self.feats:  # only focus on features from training data.
                            idx = self.feats[(label, feature)]  # get feature id
                            ep_model[idx] += weight * (1.0 / train_data_length)  # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N

            last_weight = self.weight[:]
            for j, win in enumerate(self.weight):
                delta = 1.0 / the_max * np.log(ep_empirical[j] / ep_model[j])
                self.weight[j] += delta  # update weight

            print("MaxEntClassifier is testing ...")
            classify_labels = []
            for data in test_data:
                classify_labels.append(self.classify(data))
            classify_results.append(classify_labels)

            # test if the algorithm is convergence
            if self.convergence(last_weight):
                print("The program convergence. The iter number: %d." % (i + 1))
                break

        print("MaxEntClassifier trains over!")

        return classify_results

    def classify(self, the_input_features):
        prob = self.calculate_probability(the_input_features)
        prob.sort(reverse=True)
        if prob[0][0] > prob[1][0]:
            return prob[0][1]
        else:
            return prob[1][1]
