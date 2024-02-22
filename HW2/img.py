import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np
import os

# 加载手写数字数据集
digits = load_digits()

# 指定保存图像的文件夹
save_dir = '../datasets/testing'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 指定保存图像的数量
num_images_to_save = 15
# 你可以根据需要调整这个数量

# 保存图像
for i in range(num_images_to_save):
    # 随机选择一个数字图像
    index = np.random.randint(0, len(digits.images))
    image = digits.images[index]
    digit = digits.target[index]

    # 检查该数字已保存的图像数量，以便生成唯一编号
    existing_files = [filename for filename in os.listdir(save_dir) if filename.startswith(f'digit{digit}_')]
    unique_id = len(existing_files)

    # 图像的文件名和路径
    filename = f'digit{digit}_{unique_id}.png'
    filepath = os.path.join(save_dir, filename)

    # 保存图像
    plt.figure(figsize=(2, 2))
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(filepath)
    plt.close()  # 关闭图像，以便保存下一个图像时不会出现重叠

    print(f'Image saved as {filename} in {save_dir}')
