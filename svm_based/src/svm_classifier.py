import codecs
import re
import jieba
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.svm import SVC
import joblib
import os


class SVMClassifier:
    def __init__(self, n_dim):
        self.svm_data_dir = '../output'
        if not os.path.exists(self.svm_data_dir):
            os.mkdir(self.svm_data_dir)

        self.train_labels = []
        self.train_sents = []
        self.train_words_list = []
        self.train_vecs = None

        self.dev_labels = []
        self.dev_sents = []
        self.dev_words_list = []
        self.dev_vecs = None

        self.n_dim = n_dim  # word2vec的向量维度

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
            model.save(os.path.join(self.svm_data_dir, 'w2v_model.pkl'))
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
        # print(self.train_vecs[: 10])
        # print(self.train_labels[:10])

        clf = SVC(C=0.5, kernel='rbf')
        clf.fit(self.train_vecs, self.train_labels)
        joblib.dump(clf, os.path.join(self.svm_data_dir, 'svm_model.pkl'))
        print("Training complete.")

    def evaluate(self, dev_path, word2vec_model_name, svm_model_name):
        self.dev_labels, self.dev_sents, self.dev_words_list = self.read_data(dev_path, isTrain=False)

        # 得到句子向量
        word2vec_model_path = os.path.join(self.svm_data_dir, word2vec_model_name)
        self.dev_vecs = self.get_vectors(self.dev_words_list, word2vec_model_path)

        svm_model_path = os.path.join(self.svm_data_dir, svm_model_name)
        clf = joblib.load(svm_model_path)
        print(clf.score(self.dev_vecs, self.dev_labels))

    # 对单个句子进行情感判断
    def predict(self, sent, word2vec_model_name, svm_model_name):
        words = jieba.lcut(sent)

        word2vec_model_path = os.path.join(self.svm_data_dir, word2vec_model_name)
        words_vec = self.get_single_vec(words, word2vec_model_path).reshape(1, -1)

        svm_model_path = os.path.join(self.svm_data_dir, svm_model_name)
        clf = joblib.load(svm_model_path)

        result = clf.predict(words_vec)
        return result


if __name__ == '__main__':
    svm_classifier = SVMClassifier(n_dim=10)
    svm_classifier.train("../../dataset/ChnSentiCorp/train.tsv")
    # svm_classifier.evaluate(
    #     dev_path="../../dataset/ChnSentiCorp/dev.tsv",
    #     word2vec_model_name='w2v_model.pkl',
    #     svm_model_name='svm_model.pkl')
    #
    # svm_classifier.predict(
    #     sent="在当当上买了很多书，都懒于评论。但这套书真的很好，3册都非常精彩。我家小一的女儿，认字多，非常喜爱，每天睡前必读。她还告诉我，学校的语文课本中也有相同的文章。我还借给我的同事的女儿，我同事一直头疼她女儿不爱看书，但这套书，她女儿非常喜欢。两周就看完了。建议买。很少写评论，但忍不住为这套书写下。也给别的读者参考下。",
    #     word2vec_model_name='w2v_model.pkl',
    #     svm_model_name='svm_model.pkl')
