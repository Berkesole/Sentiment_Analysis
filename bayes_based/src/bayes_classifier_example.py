import numpy as np

class BayesClassifier:
    def __init__(self, train_data, train_data_labels, best_words):
        self._pos_word_p = {}
        self._neg_word_p = {}
        self._pos_p = 0.
        self._neg_p = 1.
        self._train(train_data, train_data_labels, best_words)

    def _train(self, train_data, train_data_labels, best_words=None):
        """
        this method is different from the method self.train()
        we use the training data, do some feature selection, then train,
        get some import values
        :param train_data:
        :param train_data_labels:
        :param best_words:
        """
        print("BayesClassifier is training ...... ")

        # get the frequency information of each word
        total_pos_data, total_neg_data = {}, {}
        total_pos_length, total_neg_length = 0, 0
        total_word = set()
        for i, doc in enumerate(train_data):  # doc 是一个句子经过分词后的words list
            if train_data_labels[i] == 1:
                for word in doc:
                    if best_words is None or word in best_words:
                        total_pos_data[word] = total_pos_data.get(word, 0) + 1  # （如果没查到，缺省值为0）
                        total_pos_length += 1
                        total_word.add(word)
            else:
                for word in doc:
                    if best_words is None or word in best_words:
                        total_neg_data[word] = total_neg_data.get(word, 0) + 1
                        total_neg_length += 1
                        total_word.add(word)
        self._pos_p = total_pos_length / (total_pos_length + total_neg_length)
        self._neg_p = total_neg_length / (total_pos_length + total_neg_length)

        # get each word's probability
        for word in total_word:
            self._pos_word_p[word] = np.log(total_pos_data.get(word, 1e-100) / total_pos_length)
            self._neg_word_p[word] = np.log(total_neg_data.get(word, 1e-100) / total_neg_length)

        print("BayesClassifier trains over!")

    def classify(self, input_data):
        """
        according to the input data, calculate the probability of the each class
        :param input_data:
        """
        pos_score = 0.
        for word in input_data:
            pos_score += self._pos_word_p.get(word, 0.)
        pos_score += np.log(self._pos_p)

        neg_score = 0.
        for word in input_data:
            neg_score += self._neg_word_p.get(word, 0.)
        neg_score += np.log(self._neg_p)

        if pos_score > neg_score:
            return 1
        else:
            return 0
