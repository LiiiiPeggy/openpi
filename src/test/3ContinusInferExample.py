import dataclasses
import jax
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import json
from datetime import datetime

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# ========== 配置 ==========
# 获取pi05_droid模型配置
config = _config.get_config("pi05_droid")
# 指定本地checkpoint路径
checkpoint_dir = Path("./checkpoints/pi05_droid")

# 验证checkpoint路径是否存在
assert checkpoint_dir.exists(), f"Checkpoint not found: {checkpoint_dir}"
print(f"Loading checkpoint from: {checkpoint_dir.absolute()}")

# 创建训练好的策略
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# ========== 创建输出目录 ==========
# 创建可视化输出根目录
output_dir = Path("./visualization_output_sequence")
output_dir.mkdir(exist_ok=True)

# 生成时间戳用于文件夹命名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ========== 真实DROID数据 - 遍历所有steps ===========
print("\n=== 真实DROID数据 - 遍历所有steps ===")

import tensorflow_datasets as tfds

# DROID数据集本地路径
droid_data_path = "./droid_examples"
builder_dir = Path(droid_data_path) / "droid_100" / "1.0.0"

# 从本地目录加载TFDS数据集
builder = tfds.builder_from_directory(str(builder_dir))
ds = builder.as_dataset(split='all')

# 获取第一个Episode (Episode 0)
episodes = list(ds.take(1))
episode = episodes[0]
steps = list(episode['steps'])

total_steps = len(steps)
print(f"Episode 0 共有 {total_steps} 个steps")
print(f"开始遍历所有steps并保存结果...\n")

# 创建episode目录
episode_dir = output_dir / f"episode_0_{timestamp}"
episode_dir.mkdir(exist_ok=True)

# 保存episode元信息
episode_summary = {
    "type": "droid_episode",
    "timestamp": timestamp,
    "dataset_name": builder.info.name,
    "dataset_version": str(builder.info.version),
    "total_episodes": builder.info.splits['all'].num_examples,
    "current_episode": 0,
    "total_steps": total_steps,
}

with open(episode_dir / "episode_summary.json", "w") as f:
    json.dump(episode_summary, f, indent=2)

# 遍历所有steps
for step_idx in range(total_steps):
    print(f"处理 Step {step_idx}/{total_steps-1} ...", end=" ")
    
    # 创建当前step的独立目录
    step_dir = episode_dir / f"step_{step_idx:04d}"
    step_dir.mkdir(exist_ok=True)
    
    # 获取当前step数据
    step = steps[step_idx]
    
    # 提取原始图像数据
    exterior_img = step['observation']['exterior_image_1_left'].numpy()
    wrist_img = step['observation']['wrist_image_left'].numpy()
    
    # 调整图像大小到模型输入尺寸224x224
    exterior_img_resized = np.array(Image.fromarray(exterior_img).resize((224, 224)))
    wrist_img_resized = np.array(Image.fromarray(wrist_img).resize((224, 224)))
    
    # 提取机器人状态数据
    joint_pos = step['observation']['joint_position'].numpy()
    gripper_pos = step['observation']['gripper_position'].numpy()
    # 解码语言指令
    instruction = step['language_instruction'].numpy().decode('utf-8')
    
    # 构造模型输入示例
    real_example = {
        "observation/exterior_image_1_left": exterior_img_resized,
        "observation/wrist_image_left": wrist_img_resized,
        "observation/joint_position": joint_pos.astype(np.float32),
        "observation/gripper_position": gripper_pos.astype(np.float32),
        "prompt": instruction,
    }
    
    # 执行推理获取动作预测
    result_real = policy.infer(real_example)
    
    # 保存原始尺寸图像（180x320）
    Image.fromarray(exterior_img).save(step_dir / "exterior_original.png")
    Image.fromarray(wrist_img).save(step_dir / "wrist_original.png")
    
    # 保存调整后尺寸图像（224x224）
    Image.fromarray(exterior_img_resized).save(step_dir / "exterior_resized.png")
    Image.fromarray(wrist_img_resized).save(step_dir / "wrist_resized.png")
    
    # 保存JSON格式的结构化信息
    step_info = {
        "type": "droid_step",
        "timestamp": timestamp,
        "episode": 0,
        "step": step_idx,
        "total_steps_in_episode": total_steps,
        "dataset_name": builder.info.name,
        "dataset_version": str(builder.info.version),
        "actions_shape": result_real["actions"].shape,
        "actions_dtype": str(result_real["actions"].dtype),
        "actions": result_real["actions"].tolist(),
        "original_image_shape": list(exterior_img.shape),
        "resized_image_shape": list(exterior_img_resized.shape),
        "joint_position": joint_pos.tolist(),
        "gripper_position": gripper_pos.tolist(),
        "prompt": instruction,
    }
    
    with open(step_dir / "info.json", "w") as f:
        json.dump(step_info, f, indent=2)
    
    # 保存人类可读的TXT报告
    txt_content = f"""
========================================
DROID Step {step_idx}/{total_steps-1} 推理结果
========================================
时间戳: {timestamp}
数据集: {builder.info.name} v{builder.info.version}
Episode: 0 / {builder.info.splits['all'].num_examples}
Step: {step_idx} / {total_steps-1}

----------------------------------------
输入信息
----------------------------------------
原始图像尺寸: {exterior_img.shape}
调整后图像尺寸: {exterior_img_resized.shape}
关节位置: {joint_pos.tolist()}
夹爪位置: {gripper_pos.tolist()}
语言指令: "{instruction}"

----------------------------------------
输出动作
----------------------------------------
Actions Shape: {result_real["actions"].shape}
Actions Dtype: {result_real["actions"].dtype}

完整动作序列 (共{result_real["actions"].shape[0]}个动作):
说明: 每个动作是8维向量 [7个关节 + 1个夹爪]

"""
    
    # 逐帧写入动作详情
    for i in range(result_real["actions"].shape[0]):
        action = result_real["actions"][i]
        joints = action[:7]
        gripper = action[7]
        txt_content += f"动作 {i:2d}: 关节=[{', '.join([f'{j:8.5f}' for j in joints])}], 夹爪={gripper:.5f}\n"
    
    txt_content += f"""
----------------------------------------
保存文件
----------------------------------------
exterior_original.png  - 外部相机原始图像 ({exterior_img.shape})
wrist_original.png     - 腕部相机原始图像 ({wrist_img.shape})
exterior_resized.png   - 外部相机调整后图像 (224, 224, 3)
wrist_resized.png      - 腕部相机调整后图像 (224, 224, 3)
info.json              - 完整信息(JSON格式)
info.txt               - 本文件

========================================
"""
    
    with open(step_dir / "info.txt", "w") as f:
        f.write(txt_content)
    
    print(f"✓ 已保存到: {step_dir}/")

# 释放策略对象内存
del policy

print(f"\n========================================")
print(f"✓ 全部完成!")
print(f"========================================")
print(f"输出目录: {output_dir.absolute()}/")
print(f"\n文件结构:")
print(f"  {output_dir}/")
print(f"  └── episode_0_{timestamp}/")
print(f"      ├── episode_summary.json")
print(f"      ├── step_0000/")
print(f"      │   ├── exterior_original.png")
print(f"      │   ├── wrist_original.png")
print(f"      │   ├── exterior_resized.png")
print(f"      │   ├── wrist_resized.png")
print(f"      │   ├── info.json")
print(f"      │   └── info.txt")
print(f"      ├── step_0001/")
print(f"      │   └── ...")
print(f"      └── ... (共{total_steps}个steps)")