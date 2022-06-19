import numpy as np
import data_preprocess
from ngram import create_ngram_set, add_ngram
# from tensorflow.keras.utils import np_utils

# def feature(sentslist):
#     max_features = len(char_dic)
#     sentences2id = [[char_dic.get(word) for word in sen] for sen in sentslist]
#     ngram_range = 2
#     if ngram_range > 1:
#         print('Adding {}-gram features'.format(ngram_range))
#         # Create set of unique n-gram from the training set.
#         ngram_set = set()
#         for input_list in sentences2id:
#             for i in range(2, ngram_range + 1):
#                 set_of_ngram = create_ngram_set(input_list, ngram_value=i)
#                 ngram_set.update(set_of_ngram)
#         # Dictionary mapping n-gram token to a unique integer. 将 ngram token 映射到独立整数的词典
#         # Integer values are greater than max_features in order
#         # to avoid collision with existing features.
#         # 整数大小比 max_features 要大，按顺序排列，以避免与已存在的特征冲突
#         start_index = max_features 
#         token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        
#     fea_dict = {**token_indice,**char_dic}
#     # 使用 n-gram 特征增广 X_train 
#     sentences2id= add_ngram(sentences2id,fea_dict, ngram_range)

#     print('Average train sequence length: {}'.format(
#             np.mean(list(map(len, sentences2id)), dtype=int)))

    
#     print('Pad sequences (samples x time)')
#     X_train = sequence.pad_sequences(sentences2id, maxlen=300)
#     labels = np_utils.to_categorical(labels)


if __name__ == '__main__':
    train_doc_dir = '../dataset/ChnSentiCorp/train.tsv'
    test_doc_dir = '../dataset/ChnSentiCorp/dev.tsv'
    train_X, train_y = data_preprocess.data_builder(train_doc_dir)
    test_X, test_y = data_preprocess.data_builder(test_doc_dir)

    print(len(train_X),len(test_X))
    print('Average train sequence length: {}'.format(np.mean(list(map(len, train_X)), dtype=int)))
    print('Average train sequence length: {}'.format(np.mean(list(map(len, test_X)), dtype=int)))

    # print(train_X[0])
    # print(set(zip(*[train_X[0][i:] for i in range(3)])))
    train_X, max_features = data_preprocess.text2seq(train_X, 3)
    print(train_X)
