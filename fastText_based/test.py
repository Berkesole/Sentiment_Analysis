import numpy as np
import data_preprocess
import encode
import ftmodel
import args
# from tensorflow.keras.utils import np_utils

def evaluate(FTmodel,test_X, test_y):
	# evaluate model on test dataset dataset
	loss, acc = FTmodel.model.evaluate(test_X, np.array(test_y), verbose=0)
	print('Test Accuracy: %.5f' % (acc*100))	

if __name__ == '__main__':
    train_doc_dir = '../dataset/ChnSentiCorp/train.tsv'
    test_doc_dir = '../dataset/ChnSentiCorp/dev.tsv'
    train_X, train_y = data_preprocess.data_builder(train_doc_dir)
    test_X, test_y = data_preprocess.data_builder(test_doc_dir)

    print(len(train_X),len(test_X))
    print('Average train sequence length: {}'.format(np.mean(list(map(len, train_X)), dtype=int)))
    print('Average train sequence length: {}'.format(np.mean(list(map(len, test_X)), dtype=int)))

    train_X, test_X, max_index = encode.encode_text(train_X, test_X)
    train_X, test_X, max_features = data_preprocess.text2seq(train_X, test_X, max_index, args.ngram_value)
    # test_X, test_max_features = data_preprocess.text2seq(test_X, max_index, 3)

    FTmodel = ftmodel.FTModel(max_features)
    FTmodel.model.fit(train_X, train_y, 
        batch_size=args.batch_size, 
        epochs=args.epcoh,
        # callbacks=callbackslist,
        shuffle =True)
    evaluate(FTmodel, test_X, test_y)
    # print(train_X[0:5])