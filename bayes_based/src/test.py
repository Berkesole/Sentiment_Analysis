import data_preprocess
import bayes_classifier 
# from data_preprocess import Feature


if __name__ == '__main__':
    getData = data_preprocess.Feature('../dataset/ChnSentiCorp/train.tsv')

    # 选取best_words feature数量
    feature_ratio = 0.9
    train_X, train_y, best_words = getData.best_words(int(getData.num * feature_ratio))
    test_X, test_y = data_preprocess.data_clean('../dataset/ChnSentiCorp/dev.tsv')
    
    # print(best_words[0:20])
    # print(getData.num)
    # print(len(sents_list))
    # print(len(getData.total_words))

    # 加载bayes模型，训练
    model = bayes_classifier.BayesClassifier()
    model.train(train_X, train_y, best_words)
    model.evaluate(test_X, test_y)
