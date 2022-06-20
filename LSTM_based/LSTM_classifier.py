from functools import total_ordering
from genericpath import exists
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import codecs
import re
import jieba
from gensim.models.word2vec import Word2Vec
import os

dtype = torch.FloatTensor


class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(1 * 2, len(X), n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, len(X), n_hidden))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention


def load_data(path, isTrain=False):
    with codecs.open(path, 'r', encoding='utf-8') as pf:
        lines = pf.readlines()[1:]
        pattern = re.compile(r"\s+")

        text_list = []
        labels = []
        for line in lines:
            if isTrain:
                label, text = pattern.split(line.strip())[:2]  # 去掉末尾的空字符
            else:
                label, text = pattern.split(line.strip())[1:3]
            text_list.append(text)
            labels.append(label)
        words = [jieba.lcut(text) for text in text_list]
        if not os.path.exists("./output/w2v_model.pkl"):
            model_w2v = Word2Vec(words, vector_size=256, min_count=3)
            model_w2v.save(os.path.join('./output/w2v_model.pkl'))
        else:
            model_w2v = Word2Vec.load('./output/w2v_model.pkl')
        return text_list, labels, words


def train_model(model, input_batch, target_batch, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.train()

    optimizer.zero_grad()
    predictions, attention = model(input_batch)
    loss = criterion(predictions, target_batch)
    epoch_loss += loss.item()
    loss.backward()  # 反向传播
    optimizer.step()  # 梯度下降
    epoch_acc += ((predictions.argmax(axis=1)) == target_batch).sum().item()
    total_len += len(target_batch)
    return epoch_loss / total_len, epoch_acc / total_len


def evaluate(model, input_dev, target_dev, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.eval()
    with torch.no_grad():
        predictions, _ = model(input_dev)
        loss = criterion(predictions, target_dev)
        epoch_loss += loss.item()
        epoch_acc += ((predictions.argmax(axis=1)) == target_dev).sum().item()
        total_len += len(target_dev)
    model.train()
    return epoch_loss / total_len, epoch_acc / total_len


def train(model, input_batch, target_batch):
    bestacc = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training
    for epoch in range(100):
        total_epoch = 100
        train_loss, train_acc = train_model(model, input_batch, target_batch, optimizer, criterion)
        print("Traing Epoch:[{}/{}]".format(epoch + 1, total_epoch))
        print(f'Train Loss: {train_loss * 100:.6f} | Train Acc: {train_acc * 100:.2f}%')
        path = './output/LSTM_Attention_model.pkl'
        if bestacc < train_acc:
            bestacc = train_acc
            print("model saved!")
            torch.save(model, path)


def embedding(word_dict):
    model = Word2Vec.load('./output/w2v_model.pkl')
    embedding_dic = dict(zip(model.wv.index_to_key, model.wv.vectors))
    embedding_matrix = np.zeros((len(word_dict), 256))
    for word in word_dict:
        embedding_vec = embedding_dic.get(word)
        if embedding_vec is not None:
            embedding_matrix[word_dict[word]] = embedding_vec
    return embedding_matrix


def input_batch(sentences, max_length, word_dict):
    inputs = []
    for sen in sentences:
        sp = jieba.lcut(sen)
        s = len(sp)
        if len(sp) >= max_length:
            sp1 = sp[0:max_length]
        else:
            sp1 = sp
        list = []
        for n in sp1:
            if n in word_dict:
                list.append(word_dict[n])
            else:
                list.append(0)
        for i in range(max_length - len(sp1)):
            list.append(0)
        inputs.append(np.asarray(list))
    input_batch = Variable(torch.LongTensor(inputs))
    return input_batch


def target_batch(labels):
    targets = []
    for out in labels:
        targets.append(int(out))
    target_batch = Variable(torch.LongTensor(targets))
    return target_batch


def dic_set(words):
    if not os.path.exists('./output/word_dic.npy'):
        word_list = []
        [word_list.extend(i) for i in words]

        word_list = list(set(word_list))
        word_dict = {w: i for i, w in enumerate(word_list)}
        np.save('./output/word_dic.npy', word_dict)  # 注意带上后缀名
    else:
        word_dict = np.load('./output/word_dic.npy', allow_pickle=True).item()
    vocab_size = len(word_dict)
    return word_dict, vocab_size


if __name__ == '__main__':
    max_length = 128
    embedding_dim = 256
    n_hidden = 128
    num_classes = 2  # 0 or 1
    sentences, labels, words = load_data("../dataset/ChnSentiCorp/train.tsv", isTrain=True)
    word_dict, vocab_size = dic_set(words)
    embedding_matrix = embedding(word_dict)
    train_input_batch = input_batch(sentences, max_length, word_dict)
    train_target_batch = target_batch(labels)
    # train
    if not os.path.exists('./output/LSTM_Attention_model.pkl'):
        model = BiLSTM_Attention()
        model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))[2:10]
        train(model, train_input_batch, train_target_batch)

    model = torch.load('./output/LSTM_Attention_model.pkl')
    # Test
    dev_sentences, dev_labels, dev_words = load_data("../dataset/ChnSentiCorp/dev.tsv", isTrain=False)
    dev_input_batch = input_batch(dev_sentences, max_length, word_dict)
    dev_target_batch = target_batch(dev_labels)
    criterion = nn.CrossEntropyLoss()
    valid_loss, valid_acc = evaluate(model, dev_input_batch, dev_target_batch, criterion)
    print(f' Val. Loss: {valid_loss * 100:.6f} |  Val. Acc: {valid_acc * 100:.2f}%')
