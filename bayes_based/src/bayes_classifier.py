import numpy as np

class BayesClassifier:
    def __init__(self):
        self._pos_word_p = {}
        self._neg_word_p = {}
        self._pos_p = 0.
        self._neg_p = 1.

    def train(self, train_data, train_data_labels, best_words=None):
        # print(best_words[:20])
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

    def evaluate(self, test_data, labels):
        predicted_labels = list()
        for sent in test_data:
            predicted_labels.append(self.classify(sent))

        # 计算 accuracy
        res = {}
        for index, label in enumerate(predicted_labels):
            if label == 1 and labels[index] == 1:
                res['true_positive'] = res.get('true_positive', 0) + 1
            elif label == 0 and labels[index] == 0:
                res['true_negative'] = res.get('true_negative', 0) + 1
            elif label == 1 and labels[index] == 0:
                res['false_positive'] = res.get('false_positive', 0) + 1
            elif label == 0 and labels[index] == 1:
                res['false_negative'] = res.get('false_negative', 0) + 1
        
        print(res)

        acc = (res.get('true_positive',0) + res.get('true_negative',0))/sum(res.values())
        
        print('Test Accuracy: %.5f' % (acc*100))	