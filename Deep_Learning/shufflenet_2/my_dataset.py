#数据集预处理

from PIL import Image
import os
import torch
import json
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

#重写pytorch中Dataset类，继承的抽象类Dataset
class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 根据索引item从文件中读取一个数据
        # 对数据预处理
        # 返回数据和对应标签
        img = Image.open(self.images_path[item])
        #使用爬虫从百度爬取下来的图片不一定全部为RGB图片，而我在model里定义的shufflenet网络输入为三通道RGB图片
        # 即input_channels = 3，在第一次运行程序时发生报错
        # 故在这里用一个if判断语句判断数据集中图片是否是RGB图片，如果不是，则对其进行转换
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(
        root)

    # 遍历文件夹，一个文件夹对应一个类别，生成七个飞机类别
    airplane_class = [
        cla for cla in os.listdir(root)
        #os.listdir()传入相应的路径，将会返回相应目录下的所有文件名
        if os.path.isdir(os.path.join(root, cla))
    ]
    # 排序，保证各平台顺序一致
    airplane_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(airplane_class))
    json_str = json.dumps(dict(
        (val, key) for key, val in class_indices.items()),
                          indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in airplane_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [
            os.path.join(root, cla, i) for i in os.listdir(cla_path)
            if os.path.splitext(i)[-1] in supported
        ]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本，这里的val_rate=0.2
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  
            # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  
            # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("数据集中共有{}张图片".format(sum(every_class_num)))
    #打印输出数据集中图片总数
    print("其中{}张图片用于训练".format(len(train_images_path)))
    #打印训练集中图片总数
    print("{}张图片用于验证".format(len(val_images_path)))
    #打印验证集中图片总数
    assert len(train_images_path
               ) > 0, "number of training images must greater than 0."
    assert len(val_images_path
               ) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label