###
#  用来放一些非展示界面的代码
###

from typing import List
import numpy as np
from scipy.ndimage import binary_fill_holes
import warnings
from  trimesh.voxel.creation import voxelize
from datetime import datetime
import os
import pickle
from interfaces import *
from collections import UserDict
##############################################################
#               以下为工具函数/类（并非接口）                   #
##############################################################

class inference_info:
    ###
    # 用来装载准备时推理所需的数据，配有一个格式验证器
    ###
    start_locations:List
    end_locations:List
    actions:List
    scene_path:str
    scene_name:str
    submitted_time:datetime
    _is_submitted=False # 标记数据是否已经进入过队列
    def is_valid(self)->bool:
        msg=""
        if (self.start_locations and self.end_locations and self.actions):
            if not (len(self.start_locations)==len(self.end_locations)==len(self.actions)):
                msg+=f"The length of start_locations, end_locations and actions should be equal, but the input being({len(self.start_locations)},{len(self.end_locations)},{len(self.actions)})\n"
        else:
            msg+="Following information is missing: "
            if not self.start_locations:
                msg+="start_locations, "
            if not self.end_locations:
                msg+="end_locations, "
            if not self.actions:
                msg+="actions, "
            msg=msg[:-1]+".\n"
        if not os.path.exists(self.scene_path):# 如果场景路径不存在
            msg+=f"scene_path is provided as {self.scene_path} yet do not exist.\n"
        else:
            if os.path.isdir(self.scene_path):# 如果路径存在但是个目录
                msg+=f"scene_path expects a file but {self.scene_path} points to a directory."
            elif os.path.basename(self.scene_path)!=self.scene_name:# 如果路径存在，不是目录（是文件），检查路径和记录的场景名是否一致
                msg+="Inconsistent scene_path and scene_name detected: the path is {self.scene_path} yet the name is {self.scene_name}.\n"
        if (self._is_submitted and not self.submitted_time) or (not self._is_submitted and self.submitted_time): # 如果标记为已经上传却没有上传时间，或标记为还未上传却有上传时间
            msg+="Inconsistent submit marker and submitted time detected: the marker is {self._is_submitted} yet the submitted time is {self.submitted_time}\n"
        if len(msg)==0:
            return True
        else:
            print(msg)
            return False

    def set_scene_path(self,path:str)->None:
        #设置scene_path并根据path更新scene_name
        #该方法继承basename的行为，并不检查路径的有效性
        self.scene_path=path
        self.scene_name=os.path.basename(path)

    def __set_submitted_time__(self,overwrite=False)->None:
        if not self._is_submitted or overwrite:
            self._is_submitted=True
            self.submitted_time=datetime.now()
        else:
            warnings(f"Operation aborted, as the submitted status is configured as {self._is_submitted} and submitted time being {self.submitted_time}.")      

def fill_voxel_matrix(original_matrix):
    """
    将体素矩阵的内部区域填充为 True。
    假设 original_matrix 的 True 表示外壳。
    """
    # 使用 binary_fill_holes 对三维矩阵进行内部填充
    filled_matrix = binary_fill_holes(original_matrix)
    
    return filled_matrix


def pad_voxel_matrix_with_y_padding(original_matrix, target_shape):
    """
    将体素矩阵填充到目标尺寸，y轴的padding拼接在原矩阵的末尾，其余维度居中填充。
    """
    # 获取原始矩阵的形状
    original_shape = np.array(original_matrix.shape)
    target_shape = np.array(target_shape)

    # 计算需要填充的数量
    padding = target_shape - original_shape

    # 分别计算 x 和 z 轴（对称填充）和 y 轴（只在末尾填充）的填充量
    x_padding = padding[0] // 2
    z_padding = padding[2] // 2
    y_padding_start = 0
    y_padding_end = padding[1]

    # 构造全False的目标矩阵
    padded_matrix = np.zeros(target_shape, dtype=bool)

    # 构造切片：x 和 z 居中填充，y 在末尾拼接
    slices = (
        slice(x_padding, x_padding + original_shape[0]),  # x 轴填充
        slice(0, original_shape[1]),                      # y 轴不填充开始部分
        slice(z_padding, z_padding + original_shape[2])   # z 轴填充
    )

    # 将原始矩阵放置到目标矩阵中
    padded_matrix[slices] = original_matrix

    return padded_matrix

class PersistentDict(UserDict):
    def __init__(self, id):
        self.id = id
        # 从文件加载现有数据
        if os.path.exists(f'/cache/{self.id}.pickle'):
            with open(f'/cache/{self.id}.pickle', 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = {}

    def save(self):
        # 将数据保存到文件
        try:
            with open(f'{self.id}.pickle', 'r') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.save()  # 每次修改后自动保存

    def __delitem__(self, key):
        super().__delitem__(key)
        self.save()