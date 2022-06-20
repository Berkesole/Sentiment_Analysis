import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv(r"..\dataset\ChnSentiCorp\train.tsv", sep='\t', header=None)
dev_data = pd.read_csv(r"..\dataset\ChnSentiCorp\dev.tsv", sep='\t', header=None)
del dev_data[0]
# train_data=train_data.rename(columns={0:'label'})
# train_data=train_data.rename(columns={1:'text'})
dev_data = dev_data.rename(columns={1: 0})
dev_data = dev_data.rename(columns={2: 1})
# print(list(train_data))
# print(list(dev_data))
train_data = pd.concat([train_data, dev_data])
from collections import Counter

train_data["sentence"] = train_data.loc[:, 1]
train_data["sentence_len"] = train_data.loc[:, 1].apply(lambda x: len(str(x)))
train_data = train_data.rename(columns={0: 'label'})
train_data = train_data.rename(columns={1: 'text'})
print(list(train_data))
print(train_data["sentence_len"].head())
print(train_data["sentence_len"].describe(percentiles=[.5, .95]))
with open("vocab.txt", 'w', encoding='utf-8') as fout:
    fout.write("<unk>\n")
    fout.write("<pad>\n")
    vocab = [word for word, freq in Counter(j for i in train_data["sentence"] for j in i).most_common() if freq > 1]
    for i in vocab:
        fout.write(i + "\n")
# 初始化vocab
with open("vocab.txt", encoding='utf-8') as fin:
    vocab = [i.strip() for i in fin]
char2idx = {i: index for index, i in enumerate(vocab)}
idx2char = {index: i for index, i in enumerate(vocab)}
vocab_size = len(vocab)
pad_id = char2idx["<pad>"]
unk_id = char2idx["<unk>"]

sequence_length = 247


# 对输入数据进行预处理
def tokenizer():
    inputs = []
    sentence_char = [[j for j in i] for i in train_data["sentence"]]
    # 将输入文本进行padding
    for index, i in enumerate(sentence_char):
        temp = [char2idx.get(j, unk_id) for j in i]
        if len(temp) < sequence_length:
            for _ in range(sequence_length - len(temp)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs


data_input = tokenizer()
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Embedding_size = 100
Batch_Size = 36
Kernel = 3
Filter_num = 10
Epoch = 60
Dropout = 0.5
Learning_rate = 1e-3


class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


TextCNNDataSet = TextCNNDataSet(data_input, list(train_data["label"]))
train_size = int(len(data_input) * (8 / 9))
dev_size = len(data_input) - train_size
train_dataset, dev_dataset = torch.utils.data.random_split(TextCNNDataSet, [train_size, dev_size])

TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
DevDataLoader = Data.DataLoader(dev_dataset, batch_size=Batch_Size, shuffle=True)
# nn.Conv2d(in_channels,#输入通道数 out_channels,#输出通道数 kernel_size#卷积核大小 )
num_classs = 2


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_dim=Embedding_size)
        out_channel = Filter_num
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channel, (2, Embedding_size)),  # 卷积核大小为2*Embedding_size
            nn.ReLU(),
            nn.MaxPool2d((sequence_length - 1, 1)),
        )
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(out_channel, num_classs)

    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X)  # [batch_size, output_channel, 1, 1]
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)  # [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output


model = TextCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=Learning_rate)


def binary_acc(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return acc.item()


def train():
    avg_acc = []
    model.train()
    for index, (batch_x, batch_y) in enumerate(TrainDataLoader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        acc = binary_acc(torch.max(pred, dim=1)[1], batch_y)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


def evaluate():
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in DevDataLoader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
    return np.array(avg_acc).mean()


def test(model, vocab, sentence):
    """
    模型测试
    """
    device = list(model.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(model(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'
    # with torch.no_grad():


# Training cycle
model_train_acc, model_dev_acc = [], []

for epoch in range(Epoch):
    train_acc = train()
    dev_acc = evaluate()
    if epoch % 10 == 9:
        print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))
        print("epoch = {}, 验证准确率={}".format(epoch + 1, dev_acc))
    model_train_acc.append(train_acc)
    model_dev_acc.append(dev_acc)
test(model, vocab, ['这', '电', '影', '真', '好', '看'])
plt.plot(model_train_acc)
plt.plot(model_dev_acc)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of textCNN model")
plt.legend(['train', 'dev'])
plt.show()
