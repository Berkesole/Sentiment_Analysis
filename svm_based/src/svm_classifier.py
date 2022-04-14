# -*- coding: utf-8 -*-
import codecs
import re
import jieba
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.svm import SVC
import joblib
import os

n_dim = 200
svm_data_dir = '../svm_data'
if not os.path.exists(svm_data_dir):
    os.mkdir(svm_data_dir)


def read_data(path='../../dataset/ChnSentiCorp/train.tsv', isTrain=False):
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
            labels.append(label)
            sents.append(sent)

        words = [jieba.lcut(sent) for sent in sents]

    return labels, sents, words


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, model):
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


def get_vecs(train_data, dev_data):
    # 初始化模型和词表
    model = Word2Vec(train_data, min_count=10, vector_size=n_dim)

    # 得到句子向量
    train_vecs = []
    for words in train_data:
        sent_vec = build_sentence_vector(words, n_dim, model)
        train_vecs.append(sent_vec)
    train_vecs = np.array(train_vecs)

    # train_vecs = scale(train_vecs)

    np.save(os.path.join(svm_data_dir, 'train_vecs.npy'), train_vecs)
    # 在开发集上训练
    model.train(dev_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    model.save(os.path.join(svm_data_dir, 'w2v_model.pkl'))
    # Build test tweet vectors then scale
    dev_vecs = []
    for words in dev_data:
        sent_vec = build_sentence_vector(words, n_dim, model)
        dev_vecs.append(sent_vec)
    dev_vecs = np.array(dev_vecs)

    # dev_vecs = scale(dev_vecs)
    np.save(os.path.join(svm_data_dir, 'dev_vecs.npy'), dev_vecs)
    # print(dev_vecs.shape)


def get_data():
    train_vecs = np.load(os.path.join(svm_data_dir, 'train_vecs.npy'))
    train_labels = np.load(os.path.join(svm_data_dir, 'train_labels.npy'))
    dev_vecs = np.load(os.path.join(svm_data_dir, 'dev_vecs.npy'))
    dev_labels = np.load(os.path.join(svm_data_dir, 'dev_labels.npy'))
    return train_vecs, train_labels, dev_vecs, dev_labels


def svm_train(train_vecs, train_labels, dev_vecs, dev_labels):
    clf = SVC(C=0.5, kernel='rbf')
    clf.fit(train_vecs, train_labels)
    joblib.dump(clf, os.path.join(svm_data_dir, 'svm_model.pkl'))
    print(clf.score(dev_vecs, dev_labels))


def get_predict_vecs(words):
    model = Word2Vec.load(os.path.join(svm_data_dir, 'w2v_model.pkl'))
    vecs = build_sentence_vector(words, n_dim, model)
    # print(vecs.shape)
    return vecs


# 对单个句子进行情感判断
def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words).reshape(1, -1)
    print(words_vecs.shape)
    clf = joblib.load(os.path.join(svm_data_dir, 'svm_model.pkl'))

    result = clf.predict(words_vecs)
    return result


if __name__ == '__main__':
    # train_labels, train_sents, train_words_list = read_data('../../dataset/ChnSentiCorp/train.tsv', isTrain=True)
    # dev_labels, dev_sents, dev_words_list = read_data('../../dataset/ChnSentiCorp/dev.tsv')
    #
    # np.save(os.path.join(svm_data_dir, 'train_labels.npy', train_labels)
    # np.save(os.path.join(svm_data_dir, 'dev_labels.npy', dev_labels)
    #
    # get_vecs(train_words_list, dev_words_list)

    # train_vecs, train_labels, dev_vecs, dev_labels = get_data()
    #
    # svm_train(train_vecs, train_labels, dev_vecs, dev_labels)

    text = "在当当上买了很多书，都懒于评论。但这套书真的很好，3册都非常精彩。我家小一的女儿，认字多，非常喜爱，每天睡前必读。她还告诉我，学校的语文课本中也有相同的文章。我还借给我的同事的女儿，我同事一直头疼她女儿不爱看书，但这套书，她女儿非常喜欢。两周就看完了。建议买。很少写评论，但忍不住为这套书写下。也给别的读者参考下。"
    res = svm_predict(text)
    print(res)

