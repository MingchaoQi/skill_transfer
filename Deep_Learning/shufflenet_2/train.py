# 通过Shufflenet网络训练出可以识别其中飞机类型的模型

import os
import math
import torch
import argparse

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from model import shufflenet_v2_x2_0
from my_dataset import read_split_data, MyDataSet
from utils import train_one_epoch, validate, save_plots
from torch.utils.tensorboard import SummaryWriter


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 选定用于计算的设备，这里我用的是GPU进行计算
    print(args)
    # 数据可视化，利用tensorboard
    # 初始化 SummaryWriter，无参数，默认将使用 runs/日期时间 路径来保存日志
    # 在命令行输入tensorboard --logdir runs --bind_all调用(调用的时候注意cd到runs所在上一级目录中)
    train_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_path)

    # 这里我只对训练集做出了增强变换处理，而对验证集只是对图片做出了尺寸和位置的调整
    # 故绘制出的训练精度曲线在验证精度曲线的下方
    data_transform = {
        "train":
            transforms.Compose([
                transforms.RandomResizedCrop(
                    224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
                transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
                transforms.ToTensor(),  # 很重要的一步，将图像数据转为Tensor
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # 归一化处理
            ]),
        "val":
            transforms.Compose([
                transforms.Resize(256),  # 重新设定图像大小
                transforms.CenterCrop(224),  # 从图像中心开始裁剪图像，224为裁剪大小
                transforms.ToTensor(),  # 很重要的一步，将图像数据转为Tensor
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # 归一化处理
            ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 用于计算的cpu的数目
    print('Using {} dataloader workers every process'.format(nw))
    # 使用torch.utils.data.DataLoader()方法对训练集和验证集数据进行batch划分
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 在这里我使用官方提供的shufflenetv2_x2_0版本的预训练权重对模型进行加载训练，迁移学习
    model = shufflenet_v2_x2_0(num_classes=args.num_classes).to(device)
    # 刚开始时未使用预训练权重，从头开始训练模型，得到的模型识别准确率不高
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {
                k: v
                for k, v in weights_dict.items()
                if model.state_dict()[k].numel() == v.numel()
            }
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(
                args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    # 优化器选择SGD优化器，即随机梯度下降
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (
            1 - args.lrf) + args.lrf
    # 一般来说，我们希望在训练初期学习率大一些，使得网络收敛迅速，在训练后期学习率小一些，
    # 使得网络在收敛到最优点附近时避免来回震荡，从而更好的收敛到最优解，这里选用余弦衰减
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    # 定义四个空列表，分别用来存放训练损失，训练精度，验证损失和验证精度
    for epoch in range(args.epochs):
        print(f"第{epoch + 1}次训练")
        # epoch是从0开始的，为了更符合我们平常计数规则，输出是显示这是第'epoch+1'次训练
        # 开始训练，这里直接调用在utils.py文件已经定义好的函数'train_one_epoch'
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device)

        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        # 将每一次的训练损失和训练精度存放到列表中
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc(%): {train_epoch_acc:.3f}"
        )
        # 显示本次训练的训练损失和训练精度
        scheduler.step()

        # 开始验证，这里直接调用在utils.py文件已经定义好的函数'validate'
        valid_epoch_loss, valid_epoch_acc = validate(model=model,
                                                     valid_loader=val_loader,
                                                     device=device)
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)
        # 将每一次的验证损失和验证精度存放到列表中
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc(%): {valid_epoch_acc:.3f}"
        )
        # 利用tensorboard实现数据可视化，使用 add_scalar 方法来记录数字常量
        # tag (string): 数据名称，不同名称的数据使用不同曲线展示
        train_writer.add_scalar('train_acc', train_epoch_loss, epoch)
        train_writer.add_scalar('valid_acc', train_epoch_acc, epoch)
        train_writer.add_scalar('valid_loss', valid_epoch_loss, epoch)
        train_writer.add_scalar('valid_acc', valid_epoch_acc, epoch)
        train_writer.add_scalar('optimizer', optimizer.param_groups[0]["lr"],
                                epoch)
        torch.save(model.state_dict(),
                   "./weights/model-{}.pth".format(epoch + 1))
    # 绘制loss和accuracy的曲线
    save_plots(train_acc, valid_acc, train_loss, valid_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    # 需要分类的类别数，这里我需要分类七种战斗机，传入num_classes参数为7
    parser.add_argument('--epochs', type=int, default=30)
    # 需要训练的次数，为了保证有较好的结果，训练30次
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    # 初始学习率，学习率过大则导致模型不收敛，过小则导致模型收敛特别慢或者无法学习，
    # 经过反复的尝试，发现初始学习率设定为0.1学习效果较好
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="/home/qi/Python/Deep_Learning/Plane/plane_1/")

    # 导入预训练权重路径
    parser.add_argument('--weights',
                        type=str,
                        default='./shufflenetv2_x2_0.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    # 规定用于计算的设备，我电脑中有两个GPU，这里我选择用英伟达的GPU进行计算，即'cuda:0'

    opt = parser.parse_args()

    main(opt)
