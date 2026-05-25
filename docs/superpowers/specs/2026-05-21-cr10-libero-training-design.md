# CR10 机械臂 LIBERO 训练与仿真设计

## 概述

在 openpi 项目中，使用现有 LIBERO 公共数据集训练 pi0.5 模型，部署到 CR10 机械臂（6-DOF + 1 夹爪），并在 MuJoCo 仿真环境中可视化推理结果。

**核心策略**：训练时直接使用 LIBERO 的任务空间动作数据（末端执行器增量），推理时通过 IK 求解器将任务空间动作转换为 CR10 关节空间动作。

## 约束

- 设备：CR10 机械臂（来自 `rangerboxcr10lidar_description`），忽略移动底盘
- 数据集：LIBERO（`physical-intelligence/libero`），27 万样本，7D 任务空间动作
- 训练参数：batch_size=8，num_train_steps=30_000，LoRA 微调
- 仿真：MuJoCo（不需要 ROS/Gazebo）
- GPU：RTX A6000 48GB

## 架构

### 训练阶段

```
LIBERO 数据集 (LeRobot 格式)
  → DataConfig (LeRobotLiberoDataConfig, 复用现有)
  → pi0.5 模型 (LoRA, gemma_2b_lora + gemma_300m_lora)
  → 输出: 7D 任务空间动作 (dx, dy, dz, drx, dry, drz, gripper)
```

训练配置 `pi05_cr10_libero` 基于 `pi05_libero_low_mem_finetune`，仅修改 batch_size 和 num_train_steps。

### 推理阶段

```
观测 (图像 + 状态)
  → pi0.5 模型推理
  → 7D 任务空间动作
  → IK 求解器 (CR10IKSolver)
  → 7D 关节空间动作 (j1-j6, gripper)
  → MuJoCo 仿真执行
  → 可视化渲染
```

## 组件详细设计

### 1. 训练配置 (`src/openpi/training/config.py`)

新增 `pi05_cr10_libero` 配置：

```python
TrainConfig(
    name="pi05_cr10_libero",
    model=pi0_config.Pi0Config(
        pi05=True, action_horizon=10, discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=8,
    num_train_steps=30_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True, action_horizon=10, discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
)
```

训练命令：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_cr10_libero --exp-name=cr10_libero_v1 2>&1 | tee logs/cr10_train.log
```

### 2. IK 求解器 (`src/openpi/policies/cr10_ik.py`)

**接口**：
```python
class CR10IKSolver:
    def __init__(self, urdf_path: str, backend: str = "pykdl2"):
        """
        Args:
            urdf_path: CR10 URDF 文件路径
            backend: "pykdl2" (默认) 或 "moveit_kdl"
        """
    
    def solve(self, current_joint_pos: np.ndarray, delta_ee_pose: np.ndarray) -> np.ndarray:
        """
        Args:
            current_joint_pos: 当前关节角 [6]
            delta_ee_pose: 末端执行器增量 [dx, dy, dz, drx, dry, drz]
        Returns:
            目标关节角 [6]
        """
    
    def solve_cartesian(self, current_joint_pos: np.ndarray, target_ee_pose: np.ndarray) -> np.ndarray:
        """绝对位姿 IK（非增量）"""
```

**CR10 运动学参数**（从 URDF 提取）：
- 关节类型：全部 revolute，Z 轴旋转
- 关节限位：j1-j5 ∈ [-π, π]，j3 ∈ [-2.861, 2.861]，j6 ∈ [-2π, 2π]
- 连杆偏移：d1=0.1765, a3=0.607, a4=0.568, d4=0.191, a5=0.125, d6=0.1084

**IK 失败处理**：
1. KDL 求解失败 → 雅可比伪逆增量求解（小步迭代）
2. 仍失败 → 保持当前关节角不变，打印警告

**依赖**：`pykdl2`（`python -m pip install pykdl2`）

**moveit_kdl 后端**：
- 通过 `rospy` 调用 MoveIt 的 KDL 服务
- 需要 ROS 环境和 AGX 项目的 MoveIt 配置
- 配置文件：`/home/ubuntu/locomani/agx/TCP-IP-ROS-6AXis/cr10_moveit/config/kinematics.yaml`

### 3. CR10 策略变换 (`src/openpi/policies/cr10_policy.py`)

```python
class CR10Inputs(DataTransformFn):
    """将 CR10 观测映射到模型输入格式"""
    def __call__(self, data: dict) -> dict:
        # 映射: observation/image → base_0_rgb
        # 映射: observation/wrist_image → left_wrist_0_rgb
        # 映射: observation/state → state (8D, 与 LIBERO 兼容)
        # 映射: prompt → prompt
        ...

class CR10Outputs(DataTransformFn):
    """将模型输出映射到 CR10 动作，附加 IK 转换"""
    def __init__(self, ik_solver: CR10IKSolver):
        self.ik_solver = ik_solver
    
    def __call__(self, data: dict) -> dict:
        # 模型输出 7D 任务空间动作
        # IK 转换为 7D 关节空间动作 (j1-j6, gripper)
        ...
```

### 4. MuJoCo 仿真环境 (`examples/cr10_sim/`)

**`cr10_scene.xml`**：MuJoCo 场景文件
- CR10 固定底座（忽略 Ranger 移动底盘）
- 桌面 + 简单物体（方块、圆柱）
- 第三人称相机 + 腕部相机
- 光照设置

**`env.py`**：Gymnasium 环境封装
```python
class CR10SimEnv(gym.Env):
    def __init__(self, xml_path, cameras, ...):
        # 加载 MuJoCo 模型
        ...
    
    def reset(self) -> dict:
        # 重置环境，返回观测
        # 观测格式与 LIBERO 兼容: image, wrist_image, state
        ...
    
    def step(self, action: np.ndarray) -> tuple:
        # action: 7D 关节空间 [j1-j6, gripper]
        # 返回: obs, reward, done, info
        ...
    
    def render(self) -> np.ndarray:
        # 渲染 RGB 图像
        ...
```

**`main.py`**：推理可视化入口
```python
def main():
    # 1. 加载模型 checkpoint
    # 2. 创建 CR10SimEnv
    # 3. 创建 CR10IKSolver
    # 4. 创建 Policy（带 CR10Inputs/CR10Outputs）
    # 5. 循环推理：
    #    obs → policy.infer(obs) → action → env.step(action)
    # 6. 可视化/录制视频
```

### 5. URDF 转 MuJoCo XML

CR10 URDF 位于 `rangerboxcr10lidar_description/urdf/rangercr10lidar.urdf`。

转换步骤：
1. 提取 CR10 部分（base_link → Link1 → ... → Link6），忽略 Ranger 底盘
2. 添加 AG95 夹爪（简化为平行夹爪）
3. 添加桌面、物体、相机、光照
4. 用 `mujoco.mjcf.from_xml_string()` 或手动编写 MJCF

参考：`examples/aloha_sim/` 的 MuJoCo 场景结构。

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/openpi/training/config.py` | 修改 | 新增 `pi05_cr10_libero` 配置（~20 行） |
| `src/openpi/policies/cr10_ik.py` | 新建 | IK 求解器，双后端（~200 行） |
| `src/openpi/policies/cr10_policy.py` | 新建 | CR10 输入输出变换（~80 行） |
| `examples/cr10_sim/__init__.py` | 新建 | 包初始化 |
| `examples/cr10_sim/cr10_scene.xml` | 新建 | MuJoCo 场景文件 |
| `examples/cr10_sim/env.py` | 新建 | Gymnasium 环境封装（~150 行） |
| `examples/cr10_sim/main.py` | 新建 | 推理可视化入口（~100 行） |

## 依赖

- 现有 openpi 依赖（JAX, Flax, PyTorch 等）
- 新增：`pykdl2`（IK 求解）
- 新增：`mujoco`（仿真，已有则无需安装）

## 内存需求

- 训练：LoRA 微调约 20GB，batch_size=8 在 A6000 上可行
- 推理：约 8-10GB

## 验证计划

1. **训练验证**：loss 下降，1000 步后 loss 应明显低于初始值
2. **IK 验证**：给定已知末端位姿，IK 求解结果的 FK 应与目标一致
3. **仿真验证**：MuJoCo 中 CR10 能正常运动，不穿模
4. **端到端验证**：模型推理 → IK 转换 → 仿真执行 → 可视化中看到机器人响应
