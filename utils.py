###
#  用来放一些非展示界面的代码
###

from typing import List
import numpy as np


def scene2npy(path:str,output:str="./cache")->np.ndarray:
    '''
    从指定路径下读取场景文件，并将其采样，转化为高维数组，存储至输入数据缓存区（./cache）
    @param path:str 场景文件的路径
    @return:np.ndarray 高维np数组的转化结果
    '''
    
def prep_lingo_job(start_locations:List,end_locations:List,actions:List,output:str="./queue")->bool:
    '''
    接收模型的起始/终止位置以及行动提示词，并整理模型输入至“./queue”，等待作业队列系统处理。
    @param start_locations:List 起始坐标列表
    @param start_locations:List 终止坐标列表
    @param start_locations:List 行动提示词列表
    @return:bool 操作是否成功（相关输入是否齐全）
    '''
    assert len(start_locations)==len(end_locations)==len(actions) # 三项输入应当等长