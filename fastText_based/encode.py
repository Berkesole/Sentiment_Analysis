from tensorflow.keras.preprocessing.text import Tokenizer

def create_tokenizer(Sents):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(Sents)
	return tokenizer

def max_length(lines):
	return max([len(s.split()) for s in lines])


def encode_text(trainSents,testSents):
	# tokenizer
	tokenizer = create_tokenizer(trainSents+testSents)
	# tokenizer_test = create_tokenizer(testSents)
	# 计算最大文档长度和总词汇量
	length = max_length(trainSents+testSents)
	vocab_size = len(tokenizer.word_index) + 1
	print('Max document length: %d' % length)
	print('Vocabulary size: %d' % vocab_size)
	# encode

	train_X = tokenizer.texts_to_sequences(trainSents)
	test_X = tokenizer.texts_to_sequences(testSents)
	
	return train_X, test_X, vocab_size