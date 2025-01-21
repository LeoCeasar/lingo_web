import gradio as gr
import pandas as pd
from utils import *
# 初始空表格，假设你有4列
df = pd.DataFrame(columns=["起点x1", "起点y1", "终点x2", "终点y2", "动作"])

# 表格的更新函数
def update_table(*args):
    global df
    if len(df) < 5:
        new_row = pd.DataFrame([[None, None, None, None, None]], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True) 
    return df

# 提交按钮显示表格内容的函数
def submit_action():
    return df.to_string(index=False)

# 创建Gradio界面
with gr.Blocks() as demo:
    # 页面从上到下居中
    gr.HTML("<h1 style='text-align:center;'>大模型demo</h1>")  # 标题居中
    
    with gr.Row(variant="compact", elem_id="upload-row"):
        file_upload = gr.File(label="上传文件")  # 上传文件按钮
        
    with gr.Row(variant="compact", elem_id="table-row"):
        table = gr.DataFrame(df, label="数据表格", interactive=True) # 设置可编辑表格
    
    with gr.Row(variant="compact", elem_id="add-row"):
        add_button = gr.Button("加号")  # 添加行的按钮

    with gr.Row(variant="compact", elem_id="submit-row"):
        submit_button = gr.Button("提交")  # 提交按钮

    # 按钮事件
    add_button.click(update_table, outputs=table)  # 更新表格
    submit_button.click(submit_action, outputs=gr.HTML())  # 提交时弹出表格内容

# 启动Gradio界面
demo.launch()

