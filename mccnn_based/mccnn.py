from more_itertools import take
from data_preprocess import data_builder
from pickle import load
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate

def Trainer(model, train_X, train_y):
	model.fit([train_X,train_X,train_X], array(train_y), epochs=10, batch_size=16)
	model.save('model.h5')

def evaluate(model, train_X, train_y):
	# evaluate model on training dataset
	loss, acc = model.evaluate([train_X,train_X,train_X], array(train_y), verbose=0)
	print('Train Accuracy: %f' % (acc*100))
	# evaluate model on test dataset dataset
	loss, acc = model.evaluate([train_X,train_X,train_X],array(train_y), verbose=0)
	print('Test Accuracy: %f' % (acc*100))	

def model_builder():
	# 加载数据
	trainSents, trainLabels = load_data('../dataset/ChnSentiCorp/train.pkl')
	# tokenizer
	tokenizer = create_tokenizer(trainSents)
	# 计算最大文档长度和总词汇量
	length = max_length(trainSents)
	vocab_size = len(tokenizer.word_index) + 1
	print('Max document length: %d' % length)
	print('Vocabulary size: %d' % vocab_size)
	# encode
	trainX = encode_text(tokenizer, trainSents, length)
	print(trainX.shape)
	# 定义模型
	model = MCCNN(length, vocab_size)

	return trainX, trainLabels, model

def load_data(path):
    return load(open(path, 'rb'))

def max_length(lines):
	return max([len(s.split()) for s in lines])

def create_tokenizer(Sents):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(Sents)
	return tokenizer

def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

def MCCNN(length, vocab_size):
	# channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	# print(model.summary())
	# plot_model(model, show_shapes=True, to_file='multichannel.png')
	return model


# if __name__ == '__main__':
train_dir = '../dataset/ChnSentiCorp/train.tsv'
test_dir = '../dataset/ChnSentiCorp/test.tsv'

print("processing data...")
data_builder(train_dir, True, '../dataset/ChnSentiCorp/train.pkl')
data_builder(test_dir, False, '../dataset/ChnSentiCorp/test.pkl')
train_X, train_y, model = model_builder()
Trainer(model, train_X, train_y)
evaluate(model, train_X, train_y)