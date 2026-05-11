import dataclasses
import jax
import numpy as np
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def infer_single_image(
    image_path: str,
    prompt: str,
    wrist_image_path: str = None,
    joint_position: np.ndarray = None,
    gripper_position: np.ndarray = None,
    checkpoint_dir: str = "./checkpoints/pi05_droid",
    output_dir: str = "./inference_output"
):
    """
    使用单张图片进行模型推理
    
    Args:
        image_path: 主视角图像路径 (外部相机视角)
        prompt: 语言指令，如 "pick up the red cup"
        wrist_image_path: 腕部视角图像路径（可选，不提供则使用主图像）
        joint_position: 7维关节位置（可选，默认零值）
        gripper_position: 3维夹爪位置（可选，默认零值）
        checkpoint_dir: 模型检查点路径
        output_dir: 输出目录
    """
    
    # ========== 配置与加载模型 ==========
    config = _config.get_config("pi05_droid")
    checkpoint_dir = Path(checkpoint_dir)
    
    assert checkpoint_dir.exists(), f"检查点不存在: {checkpoint_dir}"
    print(f"加载模型: {checkpoint_dir.absolute()}")
    
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    
    # ========== 准备输入数据 ==========
    
    # 1. 加载并预处理主图像 (外部视角)
    exterior_img = Image.open(image_path).convert('RGB')
    exterior_img = exterior_img.resize((224, 224))
    exterior_array = np.array(exterior_img).astype(np.uint8)
    
    # 2. 处理腕部图像（如未提供，复制主图像）
    if wrist_image_path and Path(wrist_image_path).exists():
        wrist_img = Image.open(wrist_image_path).convert('RGB')
        wrist_img = wrist_img.resize((224, 224))
        wrist_array = np.array(wrist_img).astype(np.uint8)
    else:
        # 使用主图像作为腕部图像（或可用黑色图像）
        wrist_array = exterior_array.copy()
        print("警告: 未提供腕部图像，使用主图像替代")
    
    # 3. 设置默认值（如果未提供关节状态）
    if joint_position is None:
        joint_position = np.zeros(7, dtype=np.float32)
        print("警告: 使用默认关节位置 (全零)")
    else:
        joint_position = np.array(joint_position, dtype=np.float32)
        
    if gripper_position is None:
        gripper_position = np.zeros(3, dtype=np.float32)
        print("警告: 使用默认夹爪位置 (全零)")
    else:
        gripper_position = np.array(gripper_position, dtype=np.float32)
    
    # ========== 构建输入示例 ==========
    example = {
        "observation/exterior_image_1_left": exterior_array,
        "observation/wrist_image_left": wrist_array,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }
    
    print(f"\n输入信息:")
    print(f"  图像尺寸: {exterior_array.shape}")
    print(f"  语言指令: '{prompt}'")
    print(f"  关节位置: {joint_position}")
    print(f"  夹爪位置: {gripper_position}")
    
    # ========== 执行推理 ==========
    print("\n执行推理...")
    result = policy.infer(example)
    actions = result["actions"]
    
    print(f"推理完成!")
    print(f"  输出动作形状: {actions.shape}")
    print(f"  动作序列长度: {actions.shape[0]} 步")
    print(f"  每步动作维度: {actions.shape[1]} (7关节 + 1夹爪)")
    
    # ========== 保存结果 ==========
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存输入图像
    exterior_img.save(output_path / f"input_exterior_{timestamp}.png")
    
    # 保存详细结果
    result_data = {
        "timestamp": timestamp,
        "input": {
            "image_path": str(image_path),
            "prompt": prompt,
            "wrist_image_provided": wrist_image_path is not None,
            "joint_position": joint_position.tolist(),
            "gripper_position": gripper_position.tolist(),
        },
        "output": {
            "actions_shape": list(actions.shape),
            "actions": actions.tolist(),  # [time_steps, 8]
        }
    }
    
    with open(output_path / f"result_{timestamp}.json", "w") as f:
        json.dump(result_data, f, indent=2)
    
    # 保存可读文本报告
    report = f"""
========================================
OpenPI π0.5 推理结果
========================================
时间: {timestamp}
图像: {image_path}
指令: "{prompt}"

输入状态:
  关节位置: [{', '.join([f'{x:.3f}' for x in joint_position])}]
  夹爪位置: [{', '.join([f'{x:.3f}' for x in gripper_position])}]

输出动作序列 (共{actions.shape[0]}步):
格式: [关节1, 关节2, ..., 关节7, 夹爪]
"""
    for i, action in enumerate(actions):
        joints = action[:7]
        gripper = action[7]
        report += f"Step {i:2d}: [{', '.join([f'{x:7.4f}' for x in joints])}, {gripper:7.4f}]\n"
    
    report += f"""
========================================
结果已保存:
  - JSON: result_{timestamp}.json
  - 图像: input_exterior_{timestamp}.png
========================================
"""
    
    with open(output_path / f"report_{timestamp}.txt", "w") as f:
        f.write(report)
    
    print(f"\n结果已保存到: {output_path.absolute()}")
    
    # 释放资源
    del policy
    
    return actions, result_data


# ========== 使用示例 ==========
if __name__ == "__main__":
    
    # 示例1: 最简单用法（仅需图片路径和指令）
    actions, info = infer_single_image(
        image_path="./src/test/image.png",      # ← 替换为你的图片路径
        prompt="Pick up the water bottle on the table",        # ← 替换为你的指令
    )
    
    # 示例2: 完整用法（提供所有信息）
    # actions, info = infer_single_image(
    #     image_path="./test_scene.jpg",
    #     prompt="move the bottle to the left",
    #     wrist_image_path="./wrist_view.jpg",  # 可选
    #     joint_position=[0.1, -0.5, 0.3, 0.0, 0.2, -0.1, 0.0],  # 7维，可选
    #     gripper_position=[0.05, 0.02, 0.1],  # 3维，可选
    # )