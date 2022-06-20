# Sentiment_Analysis

本项目基于 9 种方法实现了对数据集 [ChnSentiCorp]() 的情感分析。

分别为：CNN、LSTM、朴素贝叶斯、bert、决策树、字典、fastText、MCCNN 和 SVM。



## 文件目录

Sentiment_Analysis

├── CNN_based

│  └── TextCNN.py

├── LSTM_based

│  ├── LSTM_classifier.py

│  └── output

│    ├── LSTM_Attention_model.pkl

│    ├── w2v_model.pkl

│    └── word_dic.npy

├── Readme.md

├── bayes_based

│  ├── bayes_classifier.py

│  ├── data_preprocess.py

│  └── test.py

├── bert_based

│  ├── bert-based-chinese

│  ├── bert_classifier.py

│  └── output

├── dataset

│  ├── ChnSentiCorp

│  │  ├── License.pdf

│  │  ├── dev.tsv

│  │  ├── test.tsv

│  │  └── train.tsv

│  └── stopwords.txt

├── decision_tree_based

│  ├── DecistionTree4Vec.py

│  ├── __pycache__

│  │  └── DecistionTree4Vec.cpython-37.pyc

│  └── dt_classifier.py

├── dict_based

│  ├── dict_classifier.py

│  └── dictionary

│    ├── adverb_dict.txt

│    ├── conjunction_dict.txt

│    ├── denial_dict.txt

│    ├── negative_dict.txt

│    ├── positive_dict.txt

│    ├── punctuation_dict.txt

│    └── user.dict

├── fastText_based

│  ├── args.py

│  ├── data_preprocess.py

│  ├── encode.py

│  ├── ftmodel.py

│  ├── ngram.py

│  └── test.py

├── mccnn_based

│  ├── data_preprocess.py

│  └── mccnn.py

└── svm_based

  └── svm_classifier.py  



## 环境

gensim==4.0.1 

jieba==0.42.1

numpy==1.19.2 

pandas==1.1.5 

tqdm==4.64.0

transformers==3.4.0

pytorch===1.11.0

cuda==11.3

matplotlib==3.5.2

joblib==1.1.0

sklearn





## 运行方法

### CNN

```
cd CNN
python TextCNN.py
```

训练完之后，会有两个测试用例的输出结果，和训练集、验证集的准确率曲线图



### LSTM

```
cd LSTM_based
python LSTM_classifier.py
```



### 朴素贝叶斯

```
cd bayes_based
python test.py
```



### bert

从网盘中下载预训练模型 [roberta](https://drive.google.com/drive/folders/1VeAm9_blJTOqwCvBFG0jslaiyU-B6EzT?usp=sharing)，或自行从官网下载，覆盖路径 `bert_based/roberta`

从网盘中下载已经训练好的模型 [bert_model.pkl](https://drive.google.com/file/d/1DAxIsKSxjDysYMBNj3uR06lcG69y52Co/view?usp=sharing) 放置在`bert_based/output/bert_model.pkl`

```
cd bert_based
python bert_classify.py
```



### 决策树

```
cd decision_tree_based
python dt_classify.py
```



### 字典

```
cd dict_based
python dict_classifier.py
```



### fastText

```
cd fastText_based
python test.py
```



### MCCNN

```
cd mccnn_based
python mccnn.py
```



### SVM

```
cd svm_based
python svm_classifier.py
```





## 结果比较

| Models               | Accuracy           |
| -------------------- | ------------------ |
| Dictionary           | 0.72               |
| Decision Tree        | 0.7833             |
| SVM                  | 0.7716             |
| Naive Bayes          | 0.8392             |
| fastText             | 0.8147(10 epochs)  |
| MCCNN(Multi-Channel) | 0.8925(5 epochs)   |
| TextCNN              | 0.8907(20 epochs)  |
| BiLSTM(Attention)    | 0.8858(100 epochs) |
| BERT                 | 0.9258(2 epochs)   |

