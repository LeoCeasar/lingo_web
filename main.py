import gradio as gr
import uuid
import os
import queue
import threading
import json
import time
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
from interfaces import *
# 定义任务类
class Task:
    def __init__(self, file_path):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.task_id = str(self.generate_uuid_with_timestamp())  # 生成唯一任务ID
        self.status = 'init'  # 初始任务状态为待处理
        self.file_path = file_path
        self.image_path = './images/default.jpg'
        self.video_path = None
        self.result_path = None

    def update_status(self, status):
        self.status = status

    def generate_uuid_with_timestamp(self):
        # 使用时间戳和随机数生成UUID
        return uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.timestamp}-{uuid.uuid4()}")


# 动作可选项
act_ops = ["walk", "run", "pick up", "put down", "lie down"]

# 创建一个简单的任务队列
task_queue = queue.Queue()
is_init = False


# 模拟任务处理的后台线程
def process_task_queue():
    while True:
        # 从队列中取出任务并进行处理
        task = task_queue.get()
        if task is None:
            break

        # 模拟文件处理
        task.update_status('uploaded')
        time.sleep(2)  # 假设这里做一些处理
        task.image_path = f"processed_images/{task.task_id}.png"
        task.update_status('processing')
        
        # 模拟生成视频和结果文件
        time.sleep(3)  # 假设生成视频和结果文件需要一些时间
        task.video_path = f"processed_videos/{task.task_id}_video.mp4"
        task.result_path = f"processed_results/{task.task_id}_result.zip"
        task.update_status('completed')

        # 输出任务信息
        print(f"Task {task.task_id} completed. Image:{task.image_path}")

        task_queue.task_done()

# 启动一个线程来处理任务队列
thread = threading.Thread(target=process_task_queue, daemon=True)
thread.start()


# 初始空表格，假设你有5列，初始化时包含一条数据行
df = pd.DataFrame(columns=["起点x1", "起点y1", "终点x2", "终点y2", "动作"])

# 表格的更新函数
def update_table(selected_option, table):
    lt = len(table)
    global is_init
    if not is_init:
        print("first add")
        is_init = True
        updated_table = pd.DataFrame([["", "", "", "", selected_option]], columns=["起点x1", "起点y1", "终点x2", "终点y2", "动作"])
    elif lt < 5:
        print(f"{lt}th add")
        new_row = pd.DataFrame([["", "", "", "", selected_option]], columns=table.columns)
        updated_table = pd.concat([table, new_row], ignore_index=True)  # 使用pd.concat合并
    return updated_table

# 处理文件上传并创建任务
def process_file(file):
    task = Task(file.name)  # 创建任务对象
    task.update_status('uploaded')  # 更新任务状态为上传成功
    
    os.makedirs(f"./outputs/{task.task_id}", exist_ok=True)
    os.makedirs(f"./processed_images", exist_ok=True)
    global is_init
    is_init = False
    # 体素化图像
    voxelized_result=voxelize_obj(file,output=f"./outputs/{task.task_id}/scene_voxelized.npy")
    show_voxelized_result(voxelized_result,f"./processed_images/{task.task_id}.png")
    # 保存图像路径
    img_path = f"./processed_images/{task.task_id}.png"
    
    task.image_path = img_path  # 更新任务的图片路径
    
    # 将任务信息发送到消息队列
    task_queue.put(task)
    
    return task.task_id, img_path

# 用于提交数据，返回视频和结果文件路径
def submit_task(task_id):
    # 这里可以模拟视频生成与处理
    video_path = f"./outputs/{task.task_id}/processed_videos.mp4"
    result_path = f"./outputs/{task.task_id}/processed_results.zip"
    
    # 更新任务状态
    # 根据 task_id 查找或创建任务实例
    task = None
    for task_item in task_queue.queue:
        if task_item.task_id == task_id:
            task = task_item
            break

    if not task:
        # 如果未找到对应的 task，可以返回错误或创建一个新的 Task
        return "Task not found", None

    task.update_status('completed')  # 更新状态为已完成
    task.video_path = video_path
    task.result_path = result_path
    
    send_to_queue(task)  # 将更新后的任务信息发送到队列

    return video_path, result_path

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align:center;'>3D 场景人物动态交互测试</h1>")  # 标题居中

    # 添加自定义CSS来调整下拉框和按钮的高度
    gr.HTML("""
    <style>
        #dropdown, #add_button {
            height: 80px;
        }
    </style>
    """)
    
    with gr.Column():

        # task_id_output = gr.Textbox(label="任务ID", interactive=False)
        task_id_output = gr.HTML(label="任务ID")


        file_input = gr.File(label="上传场景文件")
        img_output = gr.Image(label="场景图像", interactive=False)
        
        # 创建表格
        table = gr.DataFrame(df, label="动作规划")

        with gr.Row():  # 创建一行，包含下拉框和按钮
            # 创建下拉框
            act_dropdown = gr.Dropdown(choices=act_ops, label="选择动作", elem_id="dropdown")

            # 创建按钮
            add_button = gr.Button("添加动作", elem_id="add_button")



        preview_button = gr.Button("预览")
        submit_button = gr.Button("提交")
        
        video_output = gr.Textbox(label="视频链接", interactive=False)
        download_output = gr.Textbox(label="下载链接", interactive=False)

        file_input.upload(process_file, inputs=file_input, outputs=[task_id_output, img_output])

        # 按钮事件
        add_button.click(update_table, inputs=[act_dropdown, table], outputs=table)  # 更新表格
        # submit_button.click(submit_task, inputs=[file_upload, gr.State()], outputs=[result_display, image_display, image_display])  # 提交时处理任务并显示结果
        # submit_button.click(submit_task, inputs=task_id_output, outputs=[video_output, download_output])
        # 设置自定义CSS来调整高度

demo.launch()
