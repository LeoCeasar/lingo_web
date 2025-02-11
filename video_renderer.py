import os
from pprint import pprint
import bpy
import sys
import argparse


def set_gpu_rendering(device="CUDA"):   #目标1：尽量使他运行在GPU上（支持'CUDA', 'HIP', 'OPTIX'）
    """设置 GPU 渲染""" #新增
    assert device in ['CUDA', 'HIP', 'OPTIX', 'NONE']

    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = device  # 'CUDA', 'HIP', 'OPTIX' 根据显卡选择

    # 获取所有计算设备
    preferences.get_devices()
    for d in preferences.devices:
        d.use = device in d.type  # 仅启用指定的 GPU 设备

    # 启用 GPU 计算
    bpy.context.scene.cycles.device = "GPU"
    print(f"GPU 渲染模式已启用 ({device})")
    
    
def auto_detect_animation_range():   # 目标2：自动检测动画中存在动作的帧并自动设置渲染起始与结束帧
    """自动检测动画中存在运动的帧范围"""
    min_frame, max_frame = float('inf'), float('-inf')

    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                keyframes = [kp.co[0] for kp in fcurve.keyframe_points]
                if keyframes:
                    min_frame = min(min_frame, min(keyframes))
                    max_frame = max(max_frame, max(keyframes))

    if min_frame == float('inf') or max_frame == float('-inf'):
        return 1, 250  # 没有关键帧时使用默认范围
    
    return int(min_frame), int(max_frame)


def render_example_video(blend_file_path:str,output:str,device="CUDA"):
    """从一个blender文件（.blend）渲染一段俯瞰视角的示例视频
        !!!重要提示：避免在gradio的上下文内直接调用该代码，否则将导致gradio应用崩溃!!!

    Args:
        blend_file_path (str): blender文件的路径（Windows上运行时请使用绝对路径）
        output (str): 视频输出的路径及文件名（Windows上运行时请使用绝对路径）
        device (str, optional): 使用的设备类型（若不可用会自动回退为CPU）. Defaults to "CUDA".
    """
    assert os.path.exists(blend_file_path)
    
    # 打开.blend文件
    # blend_file_path = "vis.blend"
    print(blend_file_path)
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    print("blender file opened")

    # 设置渲染引擎为 Cycles
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # 使用 GPU 渲染
    set_gpu_rendering(device)

    # 降低采样数量从而加快渲染速度
    scene.cycles.samples = 32  # 降低采样数量
    scene.cycles.preview_samples = 16
    scene.cycles.use_denoising = True  # 启用降噪

    # 降低分辨率
    scene.render.resolution_percentage = 80  # 降低渲染分辨率

    # 启用简化模式
    scene.render.use_simplify = True
    scene.render.simplify_subdivision = 0  # 禁用细分曲面
    scene.render.simplify_child_particles = 0.5  # 减少粒子数量
    scene.render.resolution_x = 640  # 水平分辨率
    scene.render.resolution_y = 480  # 垂直分辨率
    
    # 优化灯光采样
    scene.cycles.light_sampling_threshold = 0.1  # 跳过低影响灯光

    # 关闭体积效果
    scene.cycles.volume_samples = 1  # 降低体积采样
    scene.cycles.use_volumes = False  # 禁用体积

    # 自动检测动画帧范围
    scene.frame_start, scene.frame_end = auto_detect_animation_range()
    print(f"检测到动画帧范围: {scene.frame_start} - {scene.frame_end}")

    # 设置输出路径和帧范围
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.codec="H264"
    scene.render.ffmpeg.format="MPEG4"
    scene.render.filepath =output

    # 开始渲染
    print("开始渲染")
    bpy.ops.render.render(animation=True)
    

if __name__=="__main__":
    print("Running rendering from commandline call")
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='running example video rendering in commandline mode')

    # 添加位置参数
    parser.add_argument('blender_path', help='the path for input blender file')
    # 添加位置参数
    parser.add_argument('output_path', help='the path for output example video file')
    # 添加可选参数
    parser.add_argument('-d', '--device', default='CUDA', help='the acceleration solution to be used, choose from "CUDA", "HIP", "OPTIX", "NONE"')

    # 解析命令行参数
    args = parser.parse_args()

    print(f"blender file path: {args.blender_path}")
    print(f"output video path: {args.output_path}")
    print(f"Using: {args.device}")
    
    render_example_video(args.blender_path,args.output_path,device=args.device)