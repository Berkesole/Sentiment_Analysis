import re

from jieba import posseg
import json
import codecs
import numpy as np
import os


class DictClassifier:
    def __init__(self):
        self.__root_filepath = "./dictionary"

        # jieba.load_userdict("./dictionary/user.dict")  # 准备分词词典

        # 准备情感词典词典
        self.__positive_dict = self.__get_dict(os.path.join(self.__root_filepath, "positive_dict.txt"))
        self.__negative_dict = self.__get_dict(os.path.join(self.__root_filepath, "negative_dict.txt"))
        self.__conjunction_dict = self.__get_dict(os.path.join(self.__root_filepath, "conjunction_dict.txt"))
        self.__punctuation_dict = self.__get_dict(os.path.join(self.__root_filepath, "punctuation_dict.txt"))
        self.__adverb_dict = self.__get_dict(os.path.join(self.__root_filepath, "adverb_dict.txt"))
        self.__denial_dict = self.__get_dict(os.path.join(self.__root_filepath, "denial_dict.txt"))

    def classify(self, sentence, print_show=False):
        return self.analyse_sentence(sentence, print_show)

    def evaluate(self, dev_path):
        with codecs.open(dev_path, 'r', encoding='utf-8') as pf:
            lines = pf.readlines()[1:]  # 去掉表头
            pattern = re.compile(r"\s+")

            labels = []
            sents = []
            for line in lines:
                label, sent = pattern.split(line.strip())[1:3]
                labels.append(int(label))
                sents.append(sent)

            preds = []

            for sent in sents:
                preds.append(self.analyse_sentence(sent))

            labels = np.array(labels)
            preds = np.array(preds)

            same = 0
            for i in range(len(preds)):
                if preds[i] == labels[i]:
                    same += 1
            accuracy = same / len(preds)

            tp = np.sum((labels + preds) > 1)  # label: 1, pred: 1
            fp = np.sum(labels < preds)  # label: 0, pred: 1
            fn = np.sum(labels > preds)  # label: 1, pred: 0

            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)

            print("accuracy: %f\n precision: %f\n"
                  "recall: %f"
                  % (accuracy, precision, recall))

    def analyse_sentence(self, sentence, print_show=False):
        # 情感分析整体数据结构
        comment_analysis = {"score": 0}
        # 将评论分句
        the_clauses = self.__split_sentence(sentence + "%")

        # 对每分句进行情感分析
        for i, clause in enumerate(the_clauses):
            # 情感分析子句的数据结构
            sub_clause = self.analyse_clause(clause.replace("。", "."), print_show)  # 连续的句号也表示省略号

            # 将子句分析的数据结果添加到整体数据结构中
            comment_analysis["sub-clause" + str(i)] = sub_clause
            comment_analysis['score'] += sub_clause['score']

        if print_show:
            print("\n" + sentence)
            self.__output_analysis(comment_analysis)
            print(comment_analysis, end="\n\n\n")

        # for key, value in comment_analysis.items():
        #     print(key, value)

        if comment_analysis["score"] > 0:
            return 1
        else:
            return 0

    def analyse_clause(self, the_clause, print_show):
        sub_clause = {"score": 0, "positive": [], "negative": [], "conjunction": [], "punctuation": [], "pattern": []}
        seg_result = posseg.lcut(the_clause)  # 分词，产生list

        if print_show:
            print(the_clause)
            print(seg_result)

        # 逐个分析分词
        # sign用来标记一个clause中的 conjunction, punctuation, positive, negative
        pos_sign = [0 for n in range(len(seg_result))]
        for i, _item in enumerate(seg_result):
            mark, result = self.analyse_word(_item.word, seg_result, i, pos_sign)
            if mark == 0:
                continue
            elif mark == 1:
                sub_clause["conjunction"].append(result)
            elif mark == 2:
                sub_clause["punctuation"].append(result)
            elif mark == 3:
                sub_clause["positive"].append(result)
                sub_clause["score"] += result["score"]
            elif mark == 4:
                sub_clause["negative"].append(result)
                sub_clause["score"] -= result["score"]
            pos_sign[i] = 1

        # 综合连词的情感值
        for a_conjunction in sub_clause["conjunction"]:
            sub_clause["score"] *= a_conjunction["value"]

        # 综合标点符号的情感值
        for a_punctuation in sub_clause["punctuation"]:
            sub_clause["score"] *= a_punctuation["value"]

        return sub_clause

    def analyse_word(self, the_word, seg_result=None, index=-1, pos_sign=None):
        # 判断是否属于以下某类词，如果是，则返回 (类别序列, {"key": 这个单词, "value": 这个单词在词典中的得分})
        # 如果都不是，类别序列就是0

        if the_word in self.__conjunction_dict:  # 判断是否是连词
            conjunction = {"key": the_word, "value": self.__conjunction_dict[the_word]}
            return 1, conjunction
        elif the_word in self.__punctuation_dict:  # 判断是否是标点符号
            punctuation = {"key": the_word, "value": self.__punctuation_dict[the_word]}
            return 2, punctuation
        elif the_word in self.__positive_dict:  # 判断是否是正向情感词
            # 从情感词往前搜索，判断程度
            return 3, self.emotional_word_analysis(the_word, self.__positive_dict[the_word],
                                                   [x for x, y in seg_result], index, pos_sign)
        elif the_word in self.__negative_dict:  # 判断是否是负向情感词
            return 4, self.emotional_word_analysis(the_word, self.__negative_dict[the_word],
                                                   [x for x, y in seg_result], index, pos_sign)
        else:
            return 0, ""

    def emotional_word_analysis(self, core_word, value, segments, index, pos_sign):
        # 在情感词典内，则构建一个以情感词为中心的字典数据结构
        orientation = {"key": core_word, "adverb": [], "denial": [], "value": value}
        orientation_score = orientation["value"]

        for pos in range(index - 1, -1, -1):  # 往前寻找副词和否定词
            if pos_sign[pos] == 1:  # 遇到了连接词/标点符号/情感词，停止
                break

            if segments[pos] in self.__adverb_dict:  # 副词
                # 构建副词字典数据结构
                adverb = {"key": segments[pos], "position": pos,
                          "value": self.__adverb_dict[segments[pos]]}
                orientation["adverb"].append(adverb)
                orientation_score *= self.__adverb_dict[segments[pos]]
            elif segments[pos] in self.__denial_dict:
                # 构建否定词字典数据结构
                denial = {"key": segments[pos], "position": pos,
                          "value": self.__denial_dict[segments[pos]]}
                orientation["denial"].append(denial)
                orientation_score *= -1

        # 当否定词个数为奇数，且含有副词时，进行惩罚。判断是否是“不是很好”的结构（区别于“很不好”）
        if len(orientation["adverb"]) > 0 and len(orientation["denial"]) % 2 == 1:
            # 引入调节阈值，0.3
            orientation_score *= 0.3
        # 添加情感分析值。
        orientation['score'] = orientation_score
        # print(orientation)
        return orientation  # 这里的分数都是正的，即使是负面词，会在后面做减法

    # 输出comment_analysis分析的数据结构结果
    def __output_analysis(self, comment_analysis):
        output = "Score:" + str(comment_analysis["score"]) + "\n"

        for i in range(len(comment_analysis) - 1):
            output += "Sub-clause" + str(i) + ": "
            clause = comment_analysis["sub-clause" + str(i)]
            if len(clause["conjunction"]) > 0:
                output += "conjunction:"
                for punctuation in clause["conjunction"]:
                    output += punctuation["key"] + " "
            if len(clause["positive"]) > 0:
                output += "positive:"
                for positive in clause["positive"]:
                    if len(positive["denial"]) > 0:
                        for denial in positive["denial"]:
                            # output += denial["key"] + str(denial["position"]) + "-
                            output += denial["key"] + "-"
                    if len(positive["adverb"]) > 0:
                        for adverb in positive["adverb"]:
                            # output += adverb["key"] + str(adverb["position"]) + "-"
                            output += adverb["key"] + "-"
                    output += positive["key"] + " "
            if len(clause["negative"]) > 0:
                output += "negative:"
                for negative in clause["negative"]:
                    if len(negative["denial"]) > 0:
                        for denial in negative["denial"]:
                            # output += denial["key"] + str(denial["position"]) + "-"
                            output += denial["key"] + "-"
                    if len(negative["adverb"]) > 0:
                        for adverb in negative["adverb"]:
                            # output += adverb["key"] + str(adverb["position"]) + "-
                            output += adverb["key"] + "-"
                    output += negative["key"] + " "
            if len(clause["punctuation"]) > 0:
                output += "punctuation:"
                for punctuation in clause["punctuation"]:
                    output += punctuation["key"] + " "
            if len(clause["pattern"]) > 0:
                output += "pattern:"
                for pattern in clause["pattern"]:
                    output += pattern["key"] + " "
            # if clause["pattern"] is not None:
            #     output += "pattern:" + clause["pattern"]["key"] + " "
            output += "\n"

        print(output)

    @staticmethod
    def __split_sentence(sentence):
        pattern = re.compile("[，。%、！!？?,；～~.… ]+")

        split_clauses = pattern.split(sentence.strip())
        punctuations = pattern.findall(sentence.strip())
        try:
            split_clauses.remove("")
        except ValueError:
            pass
        punctuations.append("")

        clauses = [''.join(x) for x in zip(split_clauses, punctuations)]

        clauses[-1] = clauses[-1][:-1]  # 去掉最后一个子句后额外加的百分号

        return clauses

    # 情感词典的构建
    @staticmethod
    def __get_dict(path, encoding="utf-8"):
        sentiment_dict = {}
        pattern = re.compile(r"\s+")
        with open(path, encoding=encoding) as f:
            for line in f:
                result = pattern.split(line.strip())
                if len(result) == 2:
                    sentiment_dict[result[0]] = float(result[1])
        return sentiment_dict


if __name__ == '__main__':
    ds = DictClassifier()
    a_sentence = "这本书也许你不会一气读完，也许它不够多精彩，但确实是一本值得用心去看的书。活在当下，所谓的悲伤和恐惧都是人脑脱离当下自己瞎想出来的。书里的每句话每个理论都需要用心去体会，你会受益匪浅，这是真的！做个简单快乐的人，也不过如此。看了这本书，如果你用心去看了的话，会觉得豁然轻松了，一下子看开了，不会因为生活中的琐碎而成天担忧，惶恐不安。这是一本教你放下压力的值得一买的好书。"
    result = ds.analyse_sentence(a_sentence, True)
