# ============================================================
# IMDB 电影评论情感分类：Embedding + LSTM 完整示例
# 任务：判断一条影评是正面（1）还是负面（0）
# 运行前请确保已安装：pip install datasets torch
# 如果下载数据集很慢，先设置环境变量：HF_ENDPOINT=https://hf-mirror.com
# ============================================================

import re                                          # 正则表达式，用来清洗文本中的标点
import torch                                       # PyTorch 主库
import torch.nn as nn                              # 神经网络层：Embedding、LSTM、Linear 等
from torch.utils.data import Dataset, DataLoader   # 自定义数据集类和批次加载器
from datasets import load_dataset                  # Hugging Face 的数据集加载工具
from collections import Counter                    # 计数器，用来统计词频、构建词表


# ============================================================
# 第 1 步：加载 IMDB 数据集
# ============================================================
# dataset = load_dataset("imdb")                     # 从 Hugging Face 下载/加载 IMDB，返回 DatasetDict
# dataset = load_dataset("imdb", cache_dir="./imdb_data")
dataset = load_dataset("stanfordnlp/imdb", cache_dir="./imdb_data")

train_raw = dataset["train"]                       # 训练集，原始有 25000 条
test_raw = dataset["test"]                         # 测试集，原始有 25000 条

train_raw = train_raw.shuffle(seed=42).select(range(5000))
test_raw = test_raw.shuffle(seed=42).select(range(1000))

print("训练样本数:", len(train_raw))               # 打印训练集大小
print("测试样本数:", len(test_raw))                # 打印测试集大小
print("第一条样本:", train_raw[0])                 # 查看一条样本结构：{'text': ..., 'label': 0/1}


# ============================================================
# 第 2 步：简单分词（把句子切成单词列表）
# ============================================================
def tokenize(text):
    """把一段英文文本变成小写单词列表"""
    text = text.lower()                            # 全部转小写，减少词表大小
    text = re.sub(r"[^a-z0-9']+", " ", text)       # 把非字母数字的字符替换成空格
    return text.split()                            # 按空格切分成单词列表

print("分词测试:", tokenize("I loved this movie! It's great."))   # 输出 ['i','loved','this','movie',"it's",'great']


# ============================================================
# 第 3 步：统计词频，构建词表
# ============================================================
counter = Counter()                                # 创建计数器，统计每个单词出现次数

for example in train_raw:                          # 遍历训练集每一条样本
    counter.update(tokenize(example["text"]))      # 对每条评论分词，并更新所有单词的计数

vocab = {                                          # 构建“单词 -> 整数索引”的词表
    "<pad>": 0,                                    # 0 号留给填充符 PAD
    "<unk>": 1,                                    # 1 号留给未登录词 UNK
}

for word, freq in counter.most_common():           # 按词频从高到低遍历所有单词
    if freq >= 2:                                  # 只保留出现至少 2 次的词，过滤生僻词
        vocab[word] = len(vocab)                   # 当前词表大小就是新单词的索引

print("词表大小:", len(vocab))                     # 打印最终词表大小


# ============================================================
# 第 4 步：把文本编码成固定长度的整数序列
# ============================================================
max_len = 250                                      # 每条评论统一截断/填充到 200 个 token

def encode(text):
    """把一条文本转成长度为 max_len 的整数列表"""
    tokens = tokenize(text)                        # 先分词
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]   # 每个词查词表，查不到就用 <unk>
    ids = ids[:max_len]                            # 超过 200 个词就截断
    actual_len = len(ids)
    if len(ids) < max_len:                         # 不足 200 个词就补 PAD
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))   # 补 0
    return ids,actual_len                                     # 返回长度为 200 的整数列表

test_encode = encode("this movie is great")
print(f"编码测试: {test_encode[0][:20]} actual_len{test_encode[1]}")   # 查看前 10 个 token 的索引


# ============================================================
# 第 5 步：自定义 PyTorch Dataset
# ============================================================
class IMDBDataset(Dataset):
    """包装 Hugging Face 的原始数据，让 PyTorch 的 DataLoader 能用"""

    def __init__(self, raw_data):
        self.data = raw_data                       # 保存原始 Dataset 对象

    def __len__(self):
        return len(self.data)                      # 返回样本总数，DataLoader 需要

    def __getitem__(self, idx):
        item = self.data[idx]

        tokens = tokenize(item['text'])
        length = min(len(tokens),max_len)
        input_ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
        input_ids = input_ids[:max_len]

        if len(input_ids) < max_len:
            input_ids +=  [vocab["<pad>"]] * (max_len - len(input_ids))   # 补 0


                                                  # 取第 idx 条样本
        input_ids = torch.tensor(input_ids, dtype=torch.long)   # 文本 -> 整数张量
        label = torch.tensor(item["label"], dtype=torch.long)              # 标签 -> 张量
        return input_ids, length,label                                        # 返回 (输入, 标签)

train_dataset = IMDBDataset(train_raw)             # 创建训练集对象
test_dataset = IMDBDataset(test_raw)               # 创建测试集对象

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)    # 训练集分 batch 并打乱
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)     # 测试集分 batch 不打乱

for input_ids, lengths,labels in train_loader:             # 检查一个 batch 的形状
    print(f"input_ids shape:{input_ids.shape} actual len{lengths}")     # 应为 torch.Size([32, 200])
    print("labels shape:", labels.shape)           # 应为 torch.Size([32])
    break                                          # 只看第一个 batch


# ============================================================
# 第 6 步：定义 Embedding + LSTM 模型
# ============================================================
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()                         # 初始化父类 nn.Module

        self.embedding = nn.Embedding(             # Embedding 层：整数索引 -> 稠密向量
            vocab_size,                            # 词表大小，决定 Embedding 的行数
            embed_dim,                             # 每个词的向量维度
            padding_idx=0                          # 索引 0（<pad>）固定为全零，且不更新梯度
        )
        self.lstm = nn.LSTM(                       # LSTM 层：在 Embedding 向量序列上建模时序
            embed_dim,                             # 每个时间步输入维度 = 词向量维度
            hidden_dim,                            # 隐藏状态维度
            batch_first=True                       # 输入形状为 (batch, seq_len, feature)
        )

        self.fc = nn.Linear(hidden_dim, num_classes)   # 全连接层：隐藏状态 -> 类别 logits

    def forward(self, x,len):
        # x 形状：(batch, seq_len)，里面是整数索引
        emb = self.embedding(x)                    # (batch, seq_len, embed_dim)
        emb = nn.utils.rnn.pack_padded_sequence(emb,len.cpu(),batch_first=True, enforce_sorted=False)

        output, (h, c) = self.lstm(emb)            # h[-1]：每条评论真实序列结束时的隐藏状态。 out[:, -1, :]：不能直接这样访问，因为 packed_output 是 PackedSequence，不是普通的三维 Tensor。
        last_hidden = h[-1]                        # 取最后一层最后一步的隐藏状态 hsize (numlayer,batch, hidden_dim)，即使numlayer为1也会返回1
        logits = self.fc(last_hidden)              # 全连接分类，输出 (batch, num_classes)
        return logits                              # 返回 logits，CrossEntropyLoss 内部会做 softmax

model = TextClassifier(                            # 实例化模型
    vocab_size=len(vocab),                         # 词表大小
    embed_dim=64,                                  # 词向量 64 维
    hidden_dim=128,                                # LSTM 隐藏状态 128 维
    num_classes=2                                  # 二分类：负面/正面
)

print(model)                                       # 打印模型结构，确认各层


# ============================================================
# 第 7 步：选择设备、损失函数、优化器
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 有 GPU 用 GPU，否则 CPU
print("使用设备:", device)

model = model.to(device)                           # 把模型参数搬到对应设备
criterion = nn.CrossEntropyLoss()                  # 交叉熵损失，适合分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   # Adam 优化器，学习率 0.001


# ============================================================
# 第 8 步：训练循环
# ============================================================
num_epochs = 3                                     # 训练 3 轮

for epoch in range(num_epochs):                    # 逐轮训练
    model.train()                                  # 切换到训练模式
    total_loss = 0.0                               # 累计本轮损失

    for input_ids, lengths,labels in train_loader:         # 按 batch 遍历训练集
        input_ids = input_ids.to(device)           # 输入搬到设备
        labels = labels.to(device)                 # 标签搬到设备
        lengths =lengths.to(device)
        optimizer.zero_grad()                      # 1. 清空上一步梯度
        logits = model(input_ids,lengths)                  # 2. 前向传播，得到 logits
        loss = criterion(logits, labels)           # 3. 计算损失
        loss.backward()                            # 4. 反向传播，计算梯度
        optimizer.step()                           # 5. 更新参数

        total_loss += loss.item()                  # 累加损失，用于打印平均损失

    avg_loss = total_loss / len(train_loader)      # 本轮平均损失
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")   # 打印训练进度


# ============================================================
# 第 9 步：在测试集上评估
# ============================================================
model.eval()                                       # 切换到评估模式
correct = 0                                        # 预测正确的样本数
total = 0                                          # 总样本数

with torch.no_grad():                              # 评估不需要计算梯度
    for input_ids, lengths,labels in test_loader:          # 按 batch 遍历测试集
        input_ids = input_ids.to(device)           # 输入搬到设备
        labels = labels.to(device)                 # 标签搬到设备
        lengths =lengths.to(device)

        logits = model(input_ids,lengths)                  # 2. 前向传播，得到 logits
        preds = logits.argmax(dim=1)               # 取分数最大的类别作为预测

        correct += (preds == labels).sum().item()  # 统计预测正确的数量
        total += labels.size(0)                    # 统计总数量

print(f"测试集准确率: {correct / total:.4f}")      # 打印准确率


# ============================================================
# 第 10 步：用训练好的模型预测新评论
# ============================================================
def predict(text):
    """输入一段文本，返回 '正面' 或 '负面'"""
    model.eval()
    encode_tx = encode(text)                                   # 评估模式
    ids = torch.tensor(encode_tx[0], dtype=torch.long).unsqueeze(0)   # 加 batch 维 -> (1, 200)
    ids = ids.to(device)
    ids_len = torch.tensor([len(encode_tx)])
                               # 搬到设备
    with torch.no_grad():                          # 不计算梯度
        logits = model(ids,ids_len)                        # 前向传播，(1, 2)
        pred = logits.argmax(dim=1).item()         # 取预测类别
    return "正面" if pred == 1 else "负面"          # 0=负面，1=正面

print(predict("This movie is absolutely wonderful, I love it!"))   # 预期输出：正面
print(predict("Terrible film, a complete waste of time."))         # 预期输出：负面