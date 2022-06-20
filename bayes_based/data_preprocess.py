import csv
import pandas as pd
import jieba

def data_clean(doc_dir):
    sents_list = list()
    labels = list()
    sents_list_filtered = list()

    data = pd.read_csv(doc_dir, delimiter ='\t', header=0, quoting=csv.QUOTE_NONE)
    for row in data.itertuples():
        sents_list.append(getattr(row,'text_a'))
        labels.append(getattr(row,'label'))
    
    # 停用词过滤
    sents_list = [jieba.lcut(sent) for sent in sents_list]
    stopwords_dir = '../dataset/stopwords.txt'
    stop_words = []
    with open(stopwords_dir,'r',encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            lline = line.strip()
            stop_words.append(lline)
    stop_words = set(stop_words)
    
    for sent in sents_list:
        words = [word for word in sent if not word in stop_words]
        # 过滤 short tokens
        # words = [word for word in sent if len(word) > 1]
        # 过滤空格
        for i in words:
            if  i == ' ':
                words.remove(i)
        sents_list_filtered.append(words)

    return sents_list_filtered, labels

class Feature:
    def __init__(self, doc_dir):
        self.labels = list()
        self.sents_list = list()
        self.total_words, self.total_pos_words, self.total_neg_words = {}, {}, {}
        
        self.sents_list, self.labels = data_clean(doc_dir)

        for index, sent in enumerate(self.sents_list):
            if self.labels[index] == 1:
                for word in sent:
                    self.total_pos_words[word] = self.total_pos_words.get(word, 0) + 1
                    self.total_words[word] = self.total_words.get(word,0) + 1
            else:
                for word in sent:
                    self.total_neg_words[word] = self.total_neg_words.get(word, 0) + 1
                    self.total_words[word] = self.total_words.get(word,0) + 1

        self.words = {}
        for word, freq in self.total_words.items():
            pos_score = self.__words_pos_score(self.total_pos_words.get(word, 0), 
                                                self.total_neg_words.get(word, 0), freq, 
                                                 sum(self.total_pos_words.values()), 
                                                    sum(self.total_neg_words.values()),
                                                        sum(self.total_words.values()))
            self.words[word] = pos_score * 2
            self.num = len(self.words)

    @staticmethod
    def __words_pos_score(pos_word_freq, neg_word_freq, word_freq, total_pos_words_freq, total_neg_words_freq, total_words_freq):
        """
        calculate 单词pos分数
        Args:
            pos_word_freq: 该单词出现pos的频率
            neg_word_freq: 该单词出现neg的频率
            word_freq: 该单词出现的总频率
            total_pos_words_freq: 所有单词出现pos的频率之和
            total_neg_words_freq: 所有单词出现neg的频率之和
            total_words_freq: 所有单词出现的总频率
        """
        other_pos_words_freq = total_pos_words_freq - pos_word_freq
        other_neg_words_freq = total_neg_words_freq - neg_word_freq
        score = total_words_freq * ((pos_word_freq * other_neg_words_freq - neg_word_freq * other_pos_words_freq)**2) / \
                                    (word_freq * (total_words_freq - word_freq) * total_pos_words_freq * total_neg_words_freq)
        
        return score

    def best_words(self, num, return_score=False):

        words = sorted(self.words.items(), key=lambda word_pair: word_pair[1], reverse=True)  # 得分大的放前面
        if return_score:
            return self.sents_list, self.labels, [word for word in words[:num]]
        else:
            return self.sents_list, self.labels, [word[0] for word in words[:num]]


