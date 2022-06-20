import codecs
import re
import jieba
from gensim.models.word2vec import Word2Vec
import numpy as np
import joblib
import os
from DecistionTree4Vec import DecisionTree
import pickle


class DTClassifier:
    def __init__(self, n_dim):
        self.dt_data_dir = './output'
        if not os.path.exists(self.dt_data_dir):
            os.mkdir(self.dt_data_dir)

        self.train_labels = []
        self.train_sents = []
        self.train_words_list = []
        self.train_vecs = None

        self.dev_labels = []
        self.dev_sents = []
        self.dev_words_list = []
        self.dev_vecs = None

        self.n_dim = n_dim  # word2vec的向量维度

        self.dt = DecisionTree()

    def read_data(self, path, isTrain=False):
        with codecs.open(path, 'r', encoding='utf-8') as pf:
            lines = pf.readlines()[1:]  # 去掉表头
            pattern = re.compile(r"\s+")

            labels = []
            sents = []
            for line in lines:
                if isTrain:
                    label, sent = pattern.split(line.strip())[:2]  # 去掉末尾的空字符
                else:
                    label, sent = pattern.split(line.strip())[1:3]
                labels.append(int(label))
                sents.append(sent)

            words = [jieba.lcut(sent) for sent in sents]

        return np.array(labels), sents, words

    # 对每个句子的所有词向量取均值，来生成一个句子的vector
    def build_sentence_vector(self, text, size, model):
        vec = np.zeros(size)
        count = 0.
        for word in text:
            try:
                vec += model.wv[word]
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    # 批量得到句子向量
    def get_vectors(self, words_list, word2vec_model_path=None):
        print(word2vec_model_path)
        if word2vec_model_path is None:
            model = Word2Vec(words_list, min_count=10, vector_size=self.n_dim)
            model.save(os.path.join(self.dt_data_dir, 'w2v_model.pkl'))
        else:
            model = Word2Vec.load(word2vec_model_path)

        vecs = []
        for words in words_list:
            sent_vec = self.build_sentence_vector(words, self.n_dim, model)
            vecs.append(sent_vec)
        vecs = np.array(vecs)
        return vecs

    # 得到单句向量
    def get_single_vec(self, words, word2vec_model_path):
        assert word2vec_model_path is not None

        model = Word2Vec.load(word2vec_model_path)
        vec = self.build_sentence_vector(words, self.n_dim, model)
        return vec

    def train(self, train_path):
        self.train_labels, self.train_sents, self.train_words_list = self.read_data(train_path, isTrain=True)
        # 得到句子向量
        self.train_vecs = self.get_vectors(self.train_words_list)
        self.dt.fit(self.train_vecs, np.array(self.train_labels))
        print("Training complete.")

        with open(os.path.join(self.dt_data_dir, 'dt_model.pkl'), 'wb') as f:
            pickle.dump(self.dt, f)

    def evaluate(self, dev_path, word2vec_model_name, dt_model_name):
        self.dev_labels, self.dev_sents, self.dev_words_list = self.read_data(dev_path, isTrain=False)

        # 得到句子向量
        word2vec_model_path = os.path.join(self.dt_data_dir, word2vec_model_name)
        self.dev_vecs = self.get_vectors(self.dev_words_list, word2vec_model_path)

        dt_model_path = os.path.join(self.dt_data_dir, dt_model_name)
        self.dt = joblib.load(dt_model_path)

        predict_results = self.dt.predict(self.dev_vecs)

        # print(predict_results)

        # print(self.dev_labels)

        same = 0
        for i in range(len(predict_results)):
            if predict_results[i] == self.dev_labels[i]:
                same += 1
        print(same/len(predict_results))

    def predict(self, sent, word2vec_model_name, dt_model_name):
        words = jieba.lcut(sent)

        word2vec_model_path = os.path.join(self.dt_data_dir, word2vec_model_name)
        words_vec = self.get_single_vec(words, word2vec_model_path).reshape(1, -1)

        dt_model_path = os.path.join(self.dt_data_dir, dt_model_name)
        self.dt = joblib.load(dt_model_path)

        predict_result = self.dt.predict(words_vec)[0]

        return predict_result


if __name__ == "__main__":
    dt_classifier = DTClassifier(200)

    dt_classifier.train("../dataset/ChnSentiCorp/train.tsv")

    print(dt_classifier.dt.tree)

    dt_classifier.evaluate(
        dev_path="../dataset/ChnSentiCorp/dev.tsv",
        word2vec_model_name='w2v_model.pkl',
        dt_model_name='dt_model.pkl')

    text = "这本书也许你不会一气读完，也许它不够多精彩，但确实是一本值得用心去看的书。活在当下，所谓的悲伤和恐惧都是人脑脱离当下自己瞎想出来的。书里的每句话每个理论都需要用心去体会，你会受益匪浅，这是真的！做个简单快乐的人，也不过如此。看了这本书，如果你用心去看了的话，会觉得豁然轻松了，一下子看开了，不会因为生活中的琐碎而成天担忧，惶恐不安。这是一本教你放下压力的值得一买的好书。"
    result = dt_classifier.predict(
        text,
        word2vec_model_name='w2v_model.pkl',
        dt_model_name='dt_model.pkl'
    )
    print(result)