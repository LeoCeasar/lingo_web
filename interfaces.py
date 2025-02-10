
import numpy as np
import trimesh
from  trimesh.voxel.creation import voxelize
from datetime import datetime
from classes import *
from utils import *
import json
import os
from os.path import isdir,isfile,join as path_join,exists as path_exists
from  shutil import copy2
import warnings
import matplotlib.pyplot as plt
from utils import *

def voxelize_obj(path:str,output:str="./cache/default.npy")->np.ndarray:
    '''
    “体素化”，类似于把矢量图“像素化”，是把一个三维模型采样为一个三维数组，每个元素描述对应位置的“体素”是否与模型重合
    从指定路径下读取场景文件，并将其采样，转化为高维数组，存储至输入数据缓存区（./cache）
    @param path:str 场景文件（.obj）的路径
    @param output:str="./cache/default.npy" 存储.npy文件的路径和文件名
    @return:np.ndarray 高维np数组的转化结果
    '''

    # 读取 .obj 文件
    mesh = trimesh.load_mesh(path)
    x,z,y=mesh.extents
    
    # 目标体素尺寸
    target_shape = (400, 100, 600)
    mesh.fill_holes()

    # 计算模型的长宽高
    bounds_min, bounds_max = mesh.bounds
    model_size = bounds_max - bounds_min  # 长宽高

    # 计算每个维度的缩放因子，保持长宽高比
    scaling_factors = np.array(target_shape) / model_size

    # 取最小的缩放因子，以保证场景尺寸不超过目标尺寸
    scaling_factor = min(scaling_factors)

    # 计算模型边界范围
    bounds_min, bounds_max = mesh.bounds
    model_size = bounds_max - bounds_min

    # 计算缩放比例，使模型正好适配目标体素网格
    scaling_factors = np.array(target_shape) / model_size
    scaling_factor = min(scaling_factors)*0.995  # 保证模型按最小比例等比缩放
    # 缩放模型
    mesh.apply_scale(scaling_factor)
    # 平移模型到体素网格中心
    mesh.apply_translation(-mesh.bounds[0])  # 移动到原点
    mesh.apply_translation([0,0.2,0])  # 移动到原点
    grid_size = np.array(target_shape)
    mesh.apply_translation(grid_size / 2 - model_size * scaling_factor / 2)

    # # 体素化模型
    # 将网格转换为体素网格
    voxel_grid = voxelize(mesh,1) # pitch是体素的大小

    # 获取体素网格的布尔值表示
    voxel_grid = voxel_grid.matrix
    np.save(output, voxel_grid)
    pixel_to_scene_cordinate_ratio=min(400/x,600/y)
    return voxel_grid,pixel_to_scene_cordinate_ratio
        
    
def prep_lingo_job(task:Task)->str:
    run_blender_code("get_input",blend_path="./lingo_model/vis.blend")


def show_voxelized_result(ndarray:np.ndarray,path:str):
    processed_arr_summary=-np.sum(ndarray[:,:70,:],axis=1)
    # 使用 imshow 函数将数组显示为图片
    plt.imshow(processed_arr_summary, cmap='gray') 
    plt.rcParams['figure.autolayout'] = True
    plt.axis('off')
    plt.savefig(path,bbox_inches='tight', pad_inches=0)