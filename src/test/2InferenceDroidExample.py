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
config = _config.get_config("pi05_droid")
checkpoint_dir = Path("./checkpoints/pi05_droid")

assert checkpoint_dir.exists(), f"Checkpoint not found: {checkpoint_dir}"
print(f"Loading checkpoint from: {checkpoint_dir.absolute()}")

# 创建训练好的策略
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# ========== 创建输出目录 ==========
output_dir = Path("./visualization_output")
output_dir.mkdir(exist_ok=True)
images_dir = output_dir / "images"
images_dir.mkdir(exist_ok=True)

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ========== 方式1：虚拟示例 ===========
print("\n=== 方式1：虚拟示例 ===")
dummy_example = droid_policy.make_droid_example()
result_dummy = policy.infer(dummy_example)

# 保存虚拟示例的图像
Image.fromarray(dummy_example['observation/exterior_image_1_left']).save(
    images_dir / f"dummy_exterior_{timestamp}.png"
)
Image.fromarray(dummy_example['observation/wrist_image_left']).save(
    images_dir / f"dummy_wrist_{timestamp}.png"
)

# 保存虚拟示例的信息
dummy_info = {
    "type": "dummy_example",
    "timestamp": timestamp,
    "actions_shape": result_dummy["actions"].shape,
    "actions_dtype": str(result_dummy["actions"].dtype),
    "actions_first_5": result_dummy["actions"][:5].tolist(),
    "exterior_image_shape": dummy_example['observation/exterior_image_1_left'].shape,
    "wrist_image_shape": dummy_example['observation/wrist_image_left'].shape,
    "joint_position": dummy_example['observation/joint_position'].tolist(),
    "gripper_position": dummy_example['observation/gripper_position'].tolist(),
    "prompt": dummy_example['prompt'],
}

with open(output_dir / f"dummy_info_{timestamp}.json", "w") as f:
    json.dump(dummy_info, f, indent=2)

print(f"虚拟示例 - Actions shape: {result_dummy['actions'].shape}")
print(f"虚拟示例信息已保存到: {output_dir}/dummy_info_{timestamp}.json")
print(f"虚拟示例图像已保存到: {images_dir}/")

# ========== 方式2：真实DROID数据 ===========
print("\n=== 方式2：真实DROID数据 ===")

import tensorflow_datasets as tfds

droid_data_path = "./droid_examples"
builder_dir = Path(droid_data_path) / "droid_100" / "1.0.0"

# 加载数据集
builder = tfds.builder_from_directory(str(builder_dir))
ds = builder.as_dataset(split='all')

# 获取第一个episode
episodes = list(ds.take(1))
episode = episodes[0]
steps = list(episode['steps'])

# 获取第一个step
step = steps[0]

# 提取数据
exterior_img = step['observation']['exterior_image_1_left'].numpy()
wrist_img = step['observation']['wrist_image_left'].numpy()

# 调整图像大小到224x224
exterior_img_resized = np.array(Image.fromarray(exterior_img).resize((224, 224)))
wrist_img_resized = np.array(Image.fromarray(wrist_img).resize((224, 224)))

# 提取其他数据
joint_pos = step['observation']['joint_position'].numpy()
gripper_pos = step['observation']['gripper_position'].numpy()
instruction = step['language_instruction'].numpy().decode('utf-8')

# 创建example
real_example = {
    "observation/exterior_image_1_left": exterior_img_resized,
    "observation/wrist_image_left": wrist_img_resized,
    "observation/joint_position": joint_pos.astype(np.float32),
    "observation/gripper_position": gripper_pos.astype(np.float32),
    "prompt": instruction,
}

# 运行推理
result_real = policy.infer(real_example)

# 保存原始图像（180x320）
Image.fromarray(exterior_img).save(
    images_dir / f"real_exterior_original_{timestamp}.png"
)
Image.fromarray(wrist_img).save(
    images_dir / f"real_wrist_original_{timestamp}.png"
)

# 保存调整后图像（224x224）
Image.fromarray(exterior_img_resized).save(
    images_dir / f"real_exterior_resized_{timestamp}.png"
)
Image.fromarray(wrist_img_resized).save(
    images_dir / f"real_wrist_resized_{timestamp}.png"
)

# 保存真实数据的信息
real_info = {
    "type": "real_droid_example",
    "timestamp": timestamp,
    "dataset_name": builder.info.name,
    "dataset_version": str(builder.info.version),
    "total_episodes": builder.info.splits['all'].num_examples,
    "current_episode": 0,
    "total_steps_in_episode": len(steps),
    "current_step": 0,
    "actions_shape": result_real["actions"].shape,
    "actions_dtype": str(result_real["actions"].dtype),
    "actions_all": result_real["actions"].tolist(),
    "original_image_shape": list(exterior_img.shape),
    "resized_image_shape": list(exterior_img_resized.shape),
    "joint_position": joint_pos.tolist(),
    "gripper_position": gripper_pos.tolist(),
    "prompt": instruction,
}

with open(output_dir / f"real_info_{timestamp}.json", "w") as f:
    json.dump(real_info, f, indent=2)

# 同时保存一个可读的txt文件
txt_content = f"""
========================================
DROID真实数据推理结果
========================================
时间戳: {timestamp}
数据集: {builder.info.name} v{builder.info.version}
总样本数: {builder.info.splits['all'].num_examples}
当前Episode: 0 (共{len(steps)}个steps)
当前Step: 0

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

for i in range(result_real["actions"].shape[0]):
    action = result_real["actions"][i]
    joints = action[:7]
    gripper = action[7]
    txt_content += f"动作 {i:2d}: 关节=[{', '.join([f'{j:8.5f}' for j in joints])}], 夹爪={gripper:.5f}\n"

txt_content += f"""
----------------------------------------
保存文件
----------------------------------------
原始外部图像: real_exterior_original_{timestamp}.png ({exterior_img.shape})
原始腕部图像: real_wrist_original_{timestamp}.png ({wrist_img.shape})
调整后外部图像: real_exterior_resized_{timestamp}.png (224, 224, 3)
调整后腕部图像: real_wrist_resized_{timestamp}.png (224, 224, 3)
JSON信息: real_info_{timestamp}.json

========================================
"""

with open(output_dir / f"real_info_{timestamp}.txt", "w") as f:
    f.write(txt_content)

print(f"\n真实数据信息已保存到:")
print(f"  - JSON: {output_dir}/real_info_{timestamp}.json")
print(f"  - TXT:  {output_dir}/real_info_{timestamp}.txt")
print(f"\n真实数据图像已保存到: {images_dir}/")
print(f"  - 原始尺寸 (180x320): real_exterior_original_{timestamp}.png, real_wrist_original_{timestamp}.png")
print(f"  - 调整后尺寸 (224x224): real_exterior_resized_{timestamp}.png, real_wrist_resized_{timestamp}.png")

# 释放内存
del policy

print(f"\n✓ 所有结果已保存到: {output_dir.absolute()}/")
print(f"  - 图像文件夹: {images_dir.absolute()}/")