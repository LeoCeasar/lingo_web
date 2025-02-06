import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def npy_to_2d_image(npy_file: str, output_image_path: str, projection_type='max'):
    """
    将npy格式的体素化数据转换为2D图像。
    
    参数:
    npy_file (str): 输入的npy文件路径，包含体素化的数据。
    output_image_path (str): 输出图像的路径。
    projection_type (str): 投影方式，'max' 或 'average'。默认为 'max'。
    返回:
    None
    """a

    # 加载npy文件
    voxel_data = np.load(npy_file)

    # 验证数据是否是三维
    if voxel_data.ndim != 3:
        raise ValueError(f"输入的npy文件应为三维数组，当前维度为 {voxel_data.ndim}")

    # 根据不同的投影方式生成二维图像
    if projection_type == 'max':
        # 使用最大值投影，布尔类型最大值是True
        projected_data = np.max(voxel_data, axis=0).astype(np.uint8) * 255
    elif projection_type == 'average':
        # 使用平均值投影，布尔类型平均值是True的概率
        projected_data = np.mean(voxel_data, axis=0).astype(np.uint8) * 255
    else:
        raise ValueError("投影方式不支持，仅支持 'max' 或 'average'")


    """
    将数据归一化为 [0, 255] 范围，便于生成图像 (bool类型无需数值归一化)
    projected_data = (projected_data - np.min(projected_data)) / (np.max(projected_data) - np.min(projected_data)) * 255
    projected_data = projected_data.astype(np.uint8)
    """

    # 使用PIL保存为图像
    image = Image.fromarray(projected_data)
    image.save(output_image_path)
    print(f"转换完成，图像保存至 {output_image_path}")

    # 显示图像
    plt.imshow(projected_data, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()





# 路径
npy_file = './cache/livingroom_voxelized.npy'  # 文件路径（需更改）
output_image_path = './outputs/livingroom_projection.png'  # 输出图像路径（需更改）

# 调用函数进行转换
npy_to_2d_image(npy_file, output_image_path, projection_type='max')
