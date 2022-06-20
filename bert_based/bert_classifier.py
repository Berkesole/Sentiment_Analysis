import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AdamW
from tqdm import tqdm
import torch
import torch.nn.functional as F
import transformers
import codecs
import re
import os


class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        # 加载预训练模型
        pretrained_weights = "./bert-based-chinese/"
        self.bert = transformers.BertModel.from_pretrained(pretrained_weights)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义线性函数
        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 得到bert_output
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # 获得预训练模型的输出
        bert_cls_hidden_state = bert_output[1]
        # 将768维的向量输入到线性层映射为二维向量
        linear_output = self.dense(bert_cls_hidden_state)
        return linear_output


def encoder(max_len, vocab_path, text_list):
    # 将text_list embedding成bert模型可用的输入形式
    # 加载分词模型
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    tokenizer = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'  # 返回的类型为pytorch tensor
    )
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']
    return input_ids, token_type_ids, attention_mask


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
            labels.append(int(label))

    # 调用encoder函数，获得预训练模型的三种输入形式
    input_ids, token_type_ids, attention_mask = encoder(max_len=150, vocab_path="./bert-based-chinese/vocab.txt",
                                                        text_list=text_list)
    labels = torch.tensor(labels)
    # 将encoder的返回值以及label封装为Tensor的形式
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    return data


def dev(model, dev_loader):
    # 设定模式为验证模式
    model.eval()
    # 设定不会有梯度的改变仅作验证
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(dev_loader),
                                                                              desc='Dev Itreation:'):
            input_ids, token_type_ids, attention_mask, labels = input_ids, token_type_ids, attention_mask, labels
            out_put = model(input_ids, token_type_ids, attention_mask)
            _, predict = torch.max(out_put.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
        res = correct / total
        return res


def train(model, train_loader, dev_loader):
    # 设定模型的模式为训练模式
    model.train()
    # 定义模型的损失函数
    criterion = nn.CrossEntropyLoss()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 学习率的设置
    optimizer_params = {'lr': 1e-5, 'eps': 1e-6, 'correct_bias': False}
    # 使用AdamW 主流优化器
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
    # 学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
                                  threshold=0.0001, eps=1e-08)
    t_total = len(train_loader)
    # 设定训练轮次
    total_epochs = 2
    bestAcc = 0
    correct = 0
    total = 0
    print('Training and verification begin!')
    for epoch in range(total_epochs):
        for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):
            # 从实例化的DataLoader中取出数据
            input_ids, token_type_ids, attention_mask, labels = input_ids, token_type_ids, attention_mask, labels
            # 梯度清零
            optimizer.zero_grad()
            # 将数据输入到模型中获得输出
            out_put = model(input_ids, token_type_ids, attention_mask)
            # 计算损失
            loss = criterion(out_put, labels)
            _, predict = torch.max(out_put.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            # 每两步进行一次打印
            if (step + 1) % 2 == 0:
                train_acc = correct / total
                print("Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs,
                                                                                          step + 1, len(train_loader),
                                                                                          train_acc * 100, loss.item()))
            # 每五十次进行一次验证
            if (step + 1) % 50 == 0:
                train_acc = correct / total
                # 调用验证函数dev对模型进行验证，并将有效果提升的模型进行保存
                acc = dev(model, dev_loader)
                if bestAcc < acc:
                    bestAcc = acc
                    # 模型保存路径
                    path = './output/span_bert_hide_model1.pkl'
                    torch.save(model, path)
                print("DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,dev_acc{:.6f} %,loss:{:.6f}".format(
                    epoch + 1, total_epochs, step + 1, len(train_loader), train_acc * 100, bestAcc * 100, acc * 100,
                    loss.item()))
        scheduler.step(bestAcc)


def predict(model, test_loader):
    model.eval()
    predicts = []
    predict_probs = []
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(test_loader):
            input_ids, token_type_ids, attention_mask, labels = input_ids, token_type_ids, attention_mask, labels
            out_put = model(input_ids, token_type_ids, attention_mask)

            _, predict = torch.max(out_put.data, 1)

            pre_numpy = predict.cpu().numpy().tolist()
            predicts.extend(pre_numpy)
            probs = F.softmax(out_put).detach().cpu().numpy().tolist()
            predict_probs.extend(probs)

            correct += (predict == labels).sum().item()
            total += labels.size(0)
        res = correct / total
        print('predict_Accuracy : {} %'.format(100 * res))
        # 返回预测结果和预测的概率
        return predicts, predict_probs


if __name__ == '__main__':
    # 设定batch_size
    batch_size = 16
    # 调用load_data函数，将数据加载为Tensor形式
    train_data = load_data("../dataset/ChnSentiCorp/train.tsv", isTrain=True)
    dev_data = load_data("../dataset/ChnSentiCorp/dev.tsv", isTrain=False)
    test_data = load_data("../dataset/ChnSentiCorp/test.tsv", isTrain=True)
    # 将训练数据和测试数据进行DataLoader实例化
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    if not os.path.exists("./output/span_bert_hide_model1.pkl"):
        model = BertClassificationModel()
        # 调用训练函数进行训练与验证
        train(model, train_loader, dev_loader)
    else:
        # 引进训练好的模型进行测试
        Trained_model = torch.load("./output/span_bert_hide_model1.pkl")
        # predicts是预测的（0   1），predict_probs是概率值
        predicts, predict_probs = predict(Trained_model, dev_loader)
