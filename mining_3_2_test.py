from mining_3_2_dataset import *
from mining_3_2_model import Cifarnet
from torch.utils import data
from torch import nn


# 设定模型的超参数
BATCH_SIZE = 32
LR = 0.01
EPOCH = 1000

# 加载数据
dataset = CIFAR100Dataset()

# 数据集的划分
train_dataset, test_dataset = data.random_split(dataset=dataset, lengths=[7500, 2500])

# 随机梯度下降法
# 包装成一个批次
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 加载模型
net = Cifarnet()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

best_acc = 0
for epoch in range(EPOCH):
    # 训练模型
    for X, Y in train_dataloader:
        X = X.to(torch.float32)
        X = X.permute(0, 3, 1, 2)
        logits = net(X)
        print(Y)
        net.train()  # 仅对特殊的网络层右作用
        loss = loss_fn(logits, Y)
        # 随机梯度下降法
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        optimizer.step()

    # 评估模型
    net.eval()
    valid_correct, valid_total, valid_accuracy = 0, 0, 0
    for X, Y in test_dataloader:
        # 此处不允许反向传播，因为评估模型，不是在训练
        with torch.no_grad():
            logits = net(X)
        preds = torch.max(logits.data, 1)[1]
        valid_correct += (preds == Y).sum().item()
        valid_total += Y.size(0)
    valid_accuracy = valid_correct / valid_total
    print(f"[ Valid | {epoch + 1:03d}/{epoch:03d} ], acc = {valid_accuracy:.5f}\n")

    if valid_accuracy > best_acc:
        best_acc = valid_accuracy
        print(f"best acc [{valid_accuracy:.5f}] in epoch {epoch + 1}\n")

# 持久化模型
from joblib import dump, load
dump(valid_accuracy, 'mining_3_2.joblib')