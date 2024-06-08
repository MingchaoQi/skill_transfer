#调用模型对输入图片中飞机类别进行识别预测

import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import shufflenet_v2_x2_0


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #对输入的图片进行一定的处理，如裁剪大小、转换为tensor、归一化等
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #加载指定路径的图片进行图像识别
    img_path = "D:/PYTHON/plane_1/F22/80.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(
        img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # 拓展图片batch维度
    img = torch.unsqueeze(img, dim=0)

    #读取模型训练生成的飞机类别文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(
        json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    
    model = shufflenet_v2_x2_0(num_classes=7).to(device)
    #加载训练好的模型
    model_weight_path = "./weights/model-29.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        #在预测是不需要计算梯度
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}      proability: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}     probability: {:.3}".format(
            class_indict[str(i)], predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
