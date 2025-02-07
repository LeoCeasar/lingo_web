import bpy

# 打开.blend文件
blend_file_path = "./vis.blend"
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

# 设置渲染引擎为 Cycles
bpy.data.scenes[0].render.engine = "CYCLES"

# 使用 GPU 渲染
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "CUDA"  # 'CUDA', 'HIP', 'OPTIX' 根据显卡选择
bpy.context.scene.cycles.device = "GPU"

# 降低采样数量
scene = bpy.context.scene
scene.cycles.samples = 32  # 降低采样数量
scene.cycles.preview_samples = 16
scene.cycles.use_denoising = True  # 启用降噪

# 降低分辨率
scene.render.resolution_percentage = 50  # 渲染分辨率降低到 50%

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

# 设置输出路径和帧范围
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.filepath = "./"
scene.frame_start = 1
scene.frame_end = 250

# 开始渲染
bpy.ops.render.render(animation=True)
