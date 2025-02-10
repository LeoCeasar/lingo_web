from datetime import datetime
import pandas as pd
import uuid
from collections import UserDict
import os
import pickle as pkl
class Task:
    def __init__(self, obj_path):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.task_id = str(self.generate_uuid_with_timestamp())  # 生成唯一任务ID
        self.status = 'init'  # 初始任务状态为待处理
        self.output_dir=None
        self.obj_path = obj_path
        self.image_path = './images/default.jpg'
        self.video_path = None
        self.result_path = None
        self.npy_path = None
        self.data:pd.DataFrame = None
    def update_status(self, status):
        self.status = status

    def generate_uuid_with_timestamp(self):
        # 使用时间戳和随机数生成UUID
        return uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.timestamp}-{uuid.uuid4()}")    
class PersistentDict(UserDict):
    def __init__(self, id):
        self.id = id
        # 从文件加载现有数据
        if os.path.exists(f'/cache/{self.id}.pkl'):
            with open(f'/cache/{self.id}.pkl', 'rb') as f:
                self.data = pkl.load(f)
        else:
            self.data = {}

    def save(self):
        # 将数据保存到文件
        try:
            with open(f'{self.id}.pkl', 'r') as f:
                return pkl.load(f)
        except FileNotFoundError:
            return None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.save()  # 每次修改后自动保存

    def __delitem__(self, key):
        super().__delitem__(key)
        self.save()