###
#  用来放一些非展示界面的代码
###
from classes import *
from typing import List
import numpy as np
from scipy.ndimage import binary_fill_holes
import warnings
from  trimesh.voxel.creation import voxelize
from datetime import datetime
import os
import pickle as pkl
from interfaces import *
from pathlib import Path
import subprocess
import bpy
import uuid
from os.path import isdir,isfile,join as path_join,exists as path_exists
import numpy as np
import os
import pandas as pd

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




def render_video_in_subprocess(blender_path:str,output:str,device:str="CUDA"):
    """gradio与bpy并发执行时会崩溃，因此需要在子进程中隔离运行渲染视频的代码。代码运行时父进程（gradio）会等待子进程（bpy）结束

    参数:
        blender_path (str): .blend文件的路径，在Windows上请使用绝对路径
        output (str): 输出文件的路径，视频输出格式为mkv，因此命名时请以对应扩展名结尾
        device (str,optional): 渲染使用的加速技术，可以从“CUDA”，“HIP”，“OPTIX”中选择，默认为CUDA
    """
    subprocess.run(["python", "video_renderer.py",blender_path,output,f"-d{device}"])

# def run_blender_code(script_name:str,addon_path:str="./asset/smplx_blender_addon_lh_20241129.zip",blend_path:str="./vis.blend"):
def run_blender_code(script_name:str,blend_path:str="./vis.blend",params:dict={}):
    """ 用以解决blender内部的python代码难以从外部运行的问题。该函数从.blend文件内部提取出代码块，并在当前环境下运行。
        以此法执行的代码依然相当于运行在blender外部，因此部分仅从blender内部才能访问的资源并不可用。
    参数:
        script_name (str): .blend文件内部的脚本名称，例如get_input，vis_output
        addon_path (str, optional): 需要安装的扩展. Defaults to "./asset/smplx_blender_addon_lh_20241129.zip".
        blend_path (str, optional): vis.blend的路径 Defaults to "./vis.blend".
        params (str, optional): 脚本参数，本质上是用于替换代码模板中占位符的字典，用法和python原生的模板-字符串类似. Defaults to {}.
    """
    # bpy.ops.preferences.addon_install(filepath = addon_path)
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    text = bpy.data.texts.get(script_name)
    if text:
        exec(text.as_string())
        bpy.ops.wm.save_mainfile(filepath=blend_path)
    return

def run_lingo_code_in_subprocess(input_path:str,output_path:str):
    """ 调用子进程运行Lingo模型的推理过程，以解决gradio和Lingo模型推理可能存在的兼容性问题，以及gradio并发执行时可能带来的异步错误问题
        此处配置的输入会被传入子进程并配置为环境变量。不同进程之间的环境变量不会互相影响，因此不应带来异步错误
        (需在hydra配置文件中将相应栏目替换成对环境变量的引用，其他可维持不变)
    Args:
        input_path (str): 输入目录路径；若为相对路径则会转换为绝对路径，确保执行过程安全
        output_path (str): 输出目录路径；若为相对路径则会转换为绝对路径，确保执行过程安全
    """
    input_Path=Path(input_path)
    output_Path=Path(output_path)
    input_path_resolved=input_Path.resolve()
    output_path_resolved=output_Path.resolve()
    # 获取当前工作目录
    current_directory = os.getcwd()
    print("当前工作文件夹是:", current_directory)
    os.chdir(".\lingo_model\code")
    print("切换至:", os.getcwd())
    subprocess.run(["python", "sample_lingo.py",input_path_resolved,output_path_resolved])
        
    os.chdir(current_directory)
    print(f"工作完毕，切换回：{current_directory}")


def zip_input_into_pickle(task:Task):
    """将task内的data打包成为模型可识别的输入

    参数:
        task (Task): 需要打包的task对象
    """    
    scene_name = 'demo-21'
    data = []
    ep_num=10
    for _,each_row in task.data.iterrows():
        data.append({'scene_name': scene_name, 
                    'text': each_row["动作"],
                    'start_location': np.array([each_row["起点x1"],1.0,each_row["起点y1"]]),
                    'end_location':  np.array([each_row["终点x2"],1.0,each_row["终点y2"]]),
                    'hand_location':  np.array([each_row["终点x2"],1.0,each_row["终点y2"]]),
                    'episode_num': ep_num
                    })
        ep_num+=10
    seg_num = len(data)
    for seg in data:
        seg['seg_num'] = seg_num
        
    with open(os.path.join(task.output_dir, f'{scene_name}.pkl'), 'wb') as f:
        pkl.dump(data, f)