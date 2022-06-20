# Sentiment_Analysis

本项目基于 9 种方法实现了对数据集 [ChnSentiCorp]() 的情感分析。

分别为：CNN、LSTM、朴素贝叶斯、bert、决策树、字典、fastText、MCCNN 和 SVM。



## 文件目录

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h3eyebj588j20ns12adj6.jpg" alt="截屏2022-06-20 19.53.33" style="zoom:50%;" />  



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

从网盘中下载预训练模型 [bert-based-chinese](https://drive.google.com/drive/folders/1QEHXYpgdPzjP95sTTnXSDyA0Cx8zmaR6?usp=sharing)，或自行从官网下载，路径设置为 `bert_based/bert-based-chinese`

从网盘中下载已经训练好的模型 [span_bert_hide_model1.pkl](https://drive.google.com/drive/folders/1QEHXYpgdPzjP95sTTnXSDyA0Cx8zmaR6?usp=sharing) 放置在`bert_based/output/span_bert_hide_model1.pkl`

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





## 结果分析

