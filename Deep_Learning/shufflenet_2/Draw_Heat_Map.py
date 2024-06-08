#绘制训练时特征提取的热力图

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from Grad_CAM import GradCAM, show_cam_on_image
from model import shufflenet_v2_x2_0


def main():
    model = shufflenet_v2_x2_0(num_classes=7)
    target_layers = [model.stage4[-1]]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # load image
    img_path = "D:/PYTHON/plane_1/F22/2.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(
        img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    
    target_category = 3
    
    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()