
import numpy as np
import trimesh
from datetime import datetime
from utils import *
import json
from os.path import isdir,isfile,join as path_join,exists as path_exists
from  trimesh.voxel.creation import voxelize
from  shutil import copy2
import warnings
import matplotlib.pyplot as plt
def voxelize_obj(path:str,output:str="./cache/default.npy")->np.ndarray:
    '''
    “体素化”，类似于把矢量图“像素化”，是把一个三维模型采样为一个三维数组，每个元素描述对应位置的“体素”是否与模型重合
    从指定路径下读取场景文件，并将其采样，转化为高维数组，存储至输入数据缓存区（./cache）
    @param path:str 场景文件（.obj）的路径
    @param output:str="./cache/default.npy" 存储.npy文件的路径和文件名
    @return:np.ndarray 高维np数组的转化结果
    '''

    # 读取 .obj 文件
    mesh = trimesh.load_mesh(path).to_mesh()
    # 目标体素尺寸
    target_shape = (400, 100, 600)

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
    return voxel_grid
        
    
def prep_lingo_job(inference_info:inference_info,output:str="./queue",overwrite:bool=True,submitted_time:datetime=None)->str:
    '''
    接收模型的起始/终止位置以及行动提示词，并整理模型输入至“./queue”，等待作业队列系统处理。
    假设场景文件为livingroom.obj，上传时间为2025/01/22 14：24，则/queue下目录结构为：
        /livingroom-2025-01-22-14-24
            job_description.json:{
                start_locations:[...],
                end_locations:[...],
                actions:[...],
                submitted_time:2025-01-22 14:41:48.679491,
                scene_name:"livingroom"
            }
            livingroom.obj
            livingroom_voxelized.npy
    
    @param inference_info:inference_info 一个用于封装模型输入和有效性校验逻辑的对象
    @param output:str="./queue" 输出路径，默认为根目录下/queue文件夹
    @param overwrite:bool=True 若输出路径下已存在对应文件，是否覆写
    @param submitted_time:datetime 提交时间；用于对已存在的任务更新，通常不需要赋值
    @param overwrite:bool=True 若对应目录下已存在同名文件，是否覆写(如否则会在发现重名时退出)
    @param submitted_time=None 提交时间,若之前已提交则可使用之前的提交时间，否则按当前时间新建
    @return:bool 操作是否成功（相关输入是否齐全）
    '''
    if not inference_info.is_valid():# 验证数据有效性
        warnings.warn(f"Existing file/directory found at  {output}, as the overwrite is set to {str(overwrite)}, the operation is aborted.", UserWarning)
    if inference_info.submitted_time==None:
        inference_info.submitted_time = datetime.now()
    job_description={
        'start_locations':inference_info.start_locations,
        'end_locations':inference_info.end_locations,
        'actions':inference_info.actions,
        'submitted_time':inference_info.submitted_time,
        'scene_name':inference_info.scene_name,
    }
    output=path_join(output,job_description.scene_name+"-"+str(job_description["submitted_time"]))
    
    JD_NAME="job_description.json"
    JD_PATH=path_join(output,JD_NAME)
    OBJ_NAME=f"{job_description['scene_name']}.obj"
    OBJ_PATH=path_join(output,OBJ_NAME)
    NPY_NAME=f"{job_description.scene_name}_voxelized.npy"
    NPY_PATH=path_join(output,NPY_NAME)
    if not overwrite:
        if path_exists(output) or path_exists(JD_PATH) or path_exists(OBJ_PATH) or path_exists(NPY_PATH):
            warnings.warn(f"Existing file/directory found at  {output}, as the overwrite is set to {str(overwrite)}, the operation is aborted.", UserWarning)
            return False
    os.mkdir(output)
    with open(JD_PATH, 'w', encoding='utf-8') as f:
        json.dump(job_description,f)
    copy2(inference_info.scene_path,OBJ_NAME)
    voxelize_obj(job_description["scene_path"],output=NPY_PATH)
    stat=True
    for each_path in [output,JD_PATH,OBJ_PATH,NPY_PATH]:
        stat=stat and os.path.exists(each_path)
        if not os.path.exists(each_path):
            warnings.warn(f"File/directory should have been made at {each_path}, yet was not found.")
    if stat:
        return True
    else:
        return False

def send_to_queue(path):
    pass

def show_voxelized_result(ndarray:np.ndarray,path:str):
    processed_arr_summary=-np.sum(ndarray[:,:70,:],axis=1)
    # 使用 imshow 函数将数组显示为图片
    plt.imshow(processed_arr_summary, cmap='gray')  
    plt.axis('off')
    plt.savefig(path)