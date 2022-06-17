import pandas as pd
import jieba
import csv
from pickle import dump
# from string import punctuation

def data_builder(path, is_train, save_dir):

    data = pd.read_csv(path, delimiter ='\t', header=0, quoting=csv.QUOTE_NONE)
    labels = list()
    sents = list()
    for row in data.itertuples():
        sents.append(getattr(row,'text_a'))
        if is_train:
            labels.append(getattr(row,'label'))
        else:
            labels.append(getattr(row,'qid'))

    sents = cleaner(sents)
    save_dataset([sents, labels], save_dir)
    return sents, labels

def cleaner(sents):
    sents_filtered = list()
    # 停用词过滤
    sents = [jieba.lcut(sent) for sent in sents]
    stopwords_dir = '../dataset/stopwords.txt'
    stop_words = []
    with open(stopwords_dir,'r',encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            lline = line.strip()
            stop_words.append(lline)
    stop_words = set(stop_words)
    
    for sent in sents:
        words = [word for word in sent if not word in stop_words]
        # 过滤 short tokens
        words = [word for word in sent if len(word) > 1]
        words = ' '.join(words)
        sents_filtered.append(words)

    return sents_filtered

def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

