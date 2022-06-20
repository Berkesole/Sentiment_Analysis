import pandas as pd
import jieba
import csv
from ngram import create_ngram_set, add_ngram
from keras.preprocessing import sequence
# from pickle import dump
# from string import punctuation

def data_builder(path):

    data = pd.read_csv(path, delimiter ='\t', header=0, quoting=csv.QUOTE_NONE)
    labels = list()
    sents = list()
    for row in data.itertuples():
        sents.append(getattr(row,'text_a'))
        labels.append(getattr(row,'label'))

    sents = cleaner(sents)
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

def text2seq(train_data, test_data, max_index, ngram_index=2):
    ngram_set = set()
    for text in train_data:
        for index in range(2, ngram_index + 1):
            ngram_of_text = create_ngram_set(text, index)
            ngram_set.update(ngram_of_text)

    # print(ngram_set)
    #映射n-gram字符为整数，整数的数值大于最大特征数以免冲突  
    max_features = max_index
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}  
    
    #最大特征数是数据集中数值最大的整数
    max_features = len(token_indice) + start_index 
    #用n-grams特征加强（augment）训练集和测试集
    train_data = add_ngram(train_data, token_indice, 3)
    test_data = add_ngram(test_data, token_indice, 3)

    train_data = sequence.pad_sequences(train_data, maxlen=300, padding='post')
    test_data = sequence.pad_sequences(test_data, maxlen=300, padding='post')

    return train_data, test_data, max_features

# def save_dataset(dataset, filename):
# 	dump(dataset, open(filename, 'wb'))
# 	print('Saved: %s' % filename)