import gradio as gr
import uuid
import os
import queue
import threading
import json
import time
import shutil
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
from interfaces import *
from npy_to_2d_image import *


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
        self.pixel_to_scene_cordinate_ratio=1
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
image_name_to_p2c_ratio_map={}

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
        print(f"Task {task.task_id} completed.")

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
    print(f"processing {file.name}")

    # 创建输出目录
    task_output_dir = f"./outputs/{task.task_id}"
    os.makedirs(task_output_dir, exist_ok=True)
    global is_init
    is_init = False

    # 文件转存：复制上传的文件到目标目录
    task.file_path = os.path.join(task_output_dir, os.path.basename(file.name))

    shutil.copy(file.name, task.file_path)

    print(f"task file path uploaded: {task.file_path}")
    task.update_status('uploaded')  # 更新任务状态为上传成功

    # 保存图像路径
    img_path = f"./{task_output_dir}/processed_images.png"
    NPY_PATH =f"{task_output_dir}/scene_voxelized.npy"
    # 体素化图像
    voxelized_result,pixel_to_scene_cordinate_ratio=voxelize_obj(task.file_path, output=NPY_PATH)
    image_name_to_p2c_ratio_map[img_path]=pixel_to_scene_cordinate_ratio
    print(img_path)
    show_voxelized_result(voxelized_result, img_path)
    # npy_to_2d_image(NPY_PATH, img_path, projection_type='max')
    
    
    # 更新任务的图片路径
    task.image_path = img_path
    task.update_status('npy')

    # 将任务信息发送到消息队列
    task_queue.put(task)

    return task.task_id, img_path, img_path

# 根据表格内容修改图像的函数

def preview_action(img, table):
    ratio=image_name_to_p2c_ratio_map[img]
    # 确保 img 是一个 PIL 图像对象
    if isinstance(img, str):
        img = Image.open(img)  # 如果 img 是文件路径，加载图像
    draw = ImageDraw.Draw(img)
    width, height = img.size
    r = min(width, height) / 100  # 计算圆的半径

    # 选择字体，这里使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # 确保表格中的坐标列是数字类型，无法转换的会被设置为 NaN，默认值为 0
    table["起点x1"] = pd.to_numeric(table["起点x1"], errors='coerce').fillna(0).astype("float").multiply(ratio).astype("int")
    table["起点y1"] = pd.to_numeric(table["起点y1"], errors='coerce').fillna(0).astype("float").multiply(ratio).astype("int")
    table["终点x2"] = pd.to_numeric(table["终点x2"], errors='coerce').fillna(0).astype("float").multiply(ratio).astype("int")
    table["终点y2"] = pd.to_numeric(table["终点y2"], errors='coerce').fillna(0).astype("float").multiply(ratio).astype("int")

    # 获取图像的中心点坐标
    center_x = width / 2
    center_y = height / 2

    # 遍历表格中的数据，绘制点并标注数字
    for idx, row in table.iterrows():
        try:
            # 获取表格中的坐标
            x1_center = row["起点x1"]
            y1_center = row["起点y1"]

            # 打印坐标进行调试
            print(f"Point {idx + 1}: (center_x={x1_center}, center_y={y1_center})")

            # 将表格中的中心坐标转换为相对于图像左上角的坐标
            start_x = center_x + x1_center  # 从图像中心计算起点x
            start_y = center_y - y1_center  # 从图像中心计算起点y

            # 将中心点坐标转换为圆形绘制范围
            start_x1 = start_x - r  # 圆形的左上角x坐标
            start_y1 = start_y - r  # 圆形的左上角y坐标
            start_x2 = start_x + r  # 圆形的右下角x坐标
            start_y2 = start_y + r  # 圆形的右下角y坐标

            # 绘制点
            draw.ellipse([start_x1, start_y1, start_x2, start_y2], fill="red", outline="black")  # 画点

            # 在点上标注数字
            text_x = start_x + r  # 标注数字的x位置
            text_y = start_y - (2 * r)  # 标注数字的y位置

            # 如果文本位置超出了图像的范围，做一些调整
            text_x = max(0, min(text_x, width - 20))  # 确保文本x坐标在图像范围内
            text_y = max(0, min(text_y, height - 20))  # 确保文本y坐标在图像范围内

            draw.text((text_x, text_y), str(idx + 1), fill="black", font=font)  # 标注数字，idx+1表示从1开始


            # 获取表格中的坐标
            x2_center = row["终点x2"]
            y2_center = row["终点y2"]

            # 打印坐标进行调试
            print(f"Point {idx + 1}: (center_x={x2_center}, center_y={y2_center})")

            # 将表格中的中心坐标转换为相对于图像左上角的坐标
            start_x = center_x + x2_center  # 从图像中心计算起点x
            start_y = center_y - y2_center  # 从图像中心计算起点y

            # 将中心点坐标转换为圆形绘制范围
            start_x1 = start_x - (2*r)  # 圆形的左上角x坐标
            start_y1 = start_y - (2*r)  # 圆形的左上角y坐标
            start_x2 = start_x + (2*r)  # 圆形的右下角x坐标
            start_y2 = start_y + (2*r)  # 圆形的右下角y坐标

            # 绘制点
            draw.ellipse([start_x1, start_y1, start_x2, start_y2], fill="green", outline="black")  # 画点

            # 在点上标注数字
            text_x = start_x + r  # 标注数字的x位置
            text_y = start_y - (2 * r)  # 标注数字的y位置

            # 如果文本位置超出了图像的范围，做一些调整
            text_x = max(0, min(text_x, width - 20))  # 确保文本x坐标在图像范围内
            text_y = max(0, min(text_y, height - 20))  # 确保文本y坐标在图像范围内

            draw.text((text_x, text_y), str(idx + 1), fill="black", font=font)  # 标注数字，idx+1表示从1开始

        except Exception as e:
            print(f"Error drawing row {row}: {e}")

    return img

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
        img_size = gr.HTML(label="图像大小")
        img_path = gr.Textbox(label="img path")
        img_path.visible = False
        
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

        file_input.upload(process_file, inputs=file_input, outputs=[task_id_output, img_output, img_path])

        # 按钮事件
        add_button.click(update_table, inputs=[act_dropdown, table], outputs=table)  # 更新表格
        preview_button.click(preview_action, inputs=[img_path, table], outputs=img_output)  # 点击预览时，根据表格内容生成图像并展示

        # submit_button.click(submit_task, inputs=[file_upload, gr.State()], outputs=[result_display, image_display, image_display])  # 提交时处理任务并显示结果
        # submit_button.click(submit_task, inputs=task_id_output, outputs=[video_output, download_output])
        # 设置自定义CSS来调整高度
demo.launch()
