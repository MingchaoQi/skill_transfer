# 定义训练函数和验证函数，
# 以及利用matplotlib绘制损失图和精度图

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def train_one_epoch(model, train_loader, optimizer, device):
    # 定义使用交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 因为Shufflenet网络中有BN层，需要调用model.train()，启用batch normalization
    # 保证 BN 层能够用到 每一批数据 的均值和方差
    model.train()
    print('Training')
    # 初始化
    train_running_loss = 0.0
    train_running_correct = 0.0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(image)
        # 计算损失
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # 计算准确率
        preds = torch.argmax(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
    # 计算每个epoch的loss和accuracy
    # 训练损失=在训练集错误的图片数/训练的图片总数
    # 训练精度=(训练正确的图片数/划分出的训练集中图片总数)*100%
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc


def validate(model, valid_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():  # 在计算验证集时，不需要计算梯度,也不会进行反向传播
        for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)  # labels采用0-6表示，分别表示不用的类别
            # 前向传播
            outputs = model(image)
            # 计算损失
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # 计算accuracy
            preds = torch.argmax(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
    # 计算每个epoch验证时的loss和accuracy
    # 验证损失=在验证集预测错误的图片数/验证的图片总数
    # 验证精度=(训练正确的图片数/划分出的训练集中图片总数)*100%
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))

    return epoch_loss, epoch_acc


# 定义一个函数，用来保存和生成损失图和精度图
def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    将损失和准确度图保存
    """
    # 生成精度曲线
    plt.figure(figsize=(10, 7))
    # 规定生成图像的大小
    plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
    # 规定训练精度曲线颜色为绿色，曲线样式为'-'，曲线标签为'train accuracy'
    plt.plot(valid_acc,
             color='blue',
             linestyle='-',
             label='validation accuracy')
    # 规定训练精度曲线颜色为蓝色，曲线样式为'-'，曲线标签为'validation accuracy'
    plt.xlabel('Epochs')
    # 规定横坐标为'Epochs'
    plt.ylabel('Accuracy')
    # 规定纵坐标为'Accuracy'
    plt.legend()
    plt.savefig('outputs/accuracy.png')

    # 生成损失曲线
    plt.figure(figsize=(10, 7))
    # 规定生成图像的大小
    plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
    # 规定训练损失曲线颜色为橘黄色，曲线样式为'-'，曲线标签为'train loss'
    plt.plot(valid_loss, color='red', linestyle='-', label='validation loss')
    # 规定训练损失曲线颜色为红色，曲线样式为'-'，曲线标签为'validation loss'
    plt.xlabel('Epochs')
    # 规定横坐标为'Epochs'
    plt.ylabel('Loss')
    # 规定纵坐标为'loss'
    plt.legend()
    plt.savefig('outputs/loss.png')
