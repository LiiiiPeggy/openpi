# openpi 项目详解

**openpi** — Physical Intelligence 发布的开源机器人基础模型仓库，包含视觉-语言-动作（VLA）模型的训练、推理和部署全流程。

> 仓库地址: https://github.com/Physical-Intelligence/openpi
> 版本: 0.1.0 | Python: 3.11 | 支持后端: JAX + PyTorch


---

## 一、项目整体概述

openpi 提供三个 VLA（Vision-Language-Action）模型家族，用于机器人操控任务：

| 模型 | 类型 | 特点 |
|------|------|------|
| **pi0** | Flow-based VLA | 基于流匹配（Flow Matching）的动作生成，通过迭代去噪产生连续动作 |
| **pi0-FAST** | Autoregressive VLA | 基于 FAST 动作分词器的自回归模型，将动作离散化为 token 进行 next-token 预测 |
| **pi0.5** | 改进版 pi0 | 使用知识隔离（Knowledge Insulation）训练，具有更好的开放世界泛化能力 |

核心特点：
- **预训练基座**: 在 10,000+ 小时的机器人数据上预训练，提供可直接使用的 base checkpoint
- **多平台支持**: 已验证的机器人平台包括 ALOHA（双臂）、DROID（Franka Panda）、LIBERO（仿真）、UR5
- **双后端**: 同时支持 JAX（Flax）和 PyTorch 实现
- **客户端-服务器架构**: 支持远程推理，模型可在 GPU 服务器上运行，通过 WebSocket 流式传输动作到机器人

模型架构基于 **PaliGemma**（SigLIP 视觉编码器 + Gemma 语言模型），加上一个专用的 Action Expert（Gemma 300M）处理动作生成。

**硬件需求**:

| 用途 | 显存需求 | 示例 GPU |
|------|----------|----------|
| 推理 | > 8 GB | RTX 4090 |
| 微调（LoRA） | > 22.5 GB | RTX 4090 |
| 全参数微调 | > 70 GB | A100 (80GB) / H100 |

**官方参考**:
- 博客: https://www.physicalintelligence.company/blog/pi0
- pi0-FAST 论文: https://www.physicalintelligence.company/research/fast
- pi0.5 博客: https://www.physicalintelligence.company/blog/pi05
- 知识隔离论文: https://www.physicalintelligence.company/research/knowledge_insulation

---

## 二、代码模块组成与文件夹作用

### 2.1 顶层目录

```
openpi/
├── src/openpi/              # 核心 Python 包源码
├── scripts/                 # 训练、服务、工具入口脚本
├── examples/                # 各机器人平台的示例（数据转换、评估、部署）
├── packages/                # 子包（openpi-client 轻量客户端）
├── checkpoints/             # 下载的模型检查点（pi05_droid、pi0_fast_droid）
├── docs/                    # 补充文档（Docker、归一化统计、远程推理）
├── third_party/             # Git 子模块（aloha、libero 仿真环境）
├── droid_examples/          # DROID 数据集示例 TFRecord 文件
├── pyproject.toml           # 项目元数据、依赖、构建配置
├── uv.lock                  # uv 依赖锁文件
├── README.md                # 官方文档
└── .python-version          # 固定 Python 3.11
```

### 2.2 核心源码包 `src/openpi/`

#### models/ — 模型架构定义

| 文件 | 功能 |
|------|------|
| `model.py` | 核心抽象：`ModelType` 枚举（PI0/PI0_FAST/PI05）、`Observation` 数据类、`BaseModel` 抽象基类（`compute_loss`、`sample_actions`）、图像预处理 |
| `pi0.py` | pi0/pi0.5 模型（JAX/Flax）：流匹配动作生成，PaliGemma 前缀 + Action Expert 后缀 |
| `pi0_fast.py` | pi0-FAST 模型（JAX/Flax）：自回归动作生成，FAST 分词器 + next-token 预测 |
| `pi0_config.py` | Pi0Config 数据类：action_dim=32、action_horizon=50、LoRA 配置 |
| `gemma.py` | Gemma 语言模型（JAX/Flax）：RMSNorm、GQA、SwiGLU MLP、LoRA 支持、adaRMS |
| `siglip.py` | SigLIP 视觉编码器（ViT So400m/14） |
| `tokenizer.py` | 分词器：`PaligemmaTokenizer`（SentencePiece）、`FASTTokenizer`（动作分词） |
| `lora.py` | LoRA 实现：可配置 rank、alpha、rslora |

#### models_pytorch/ — PyTorch 模型实现

| 文件 | 功能 |
|------|------|
| `pi0_pytorch.py` | PyTorch 版 pi0/pi0.5，支持 `torch.compile`、梯度检查点 |
| `gemma_pytorch.py` | `PaliGemmaWithExpertModel`：封装 HuggingFace Transformers |
| `transformers_replace/` | 修补的 HuggingFace transformers 文件（adaRMS 支持、精度控制、KV cache 修复） |

#### policies/ — 策略定义（机器人特定的输入/输出变换）

| 文件 | 功能 |
|------|------|
| `policy.py` | `Policy` 类：封装模型 + 输入/输出变换，`infer` 方法执行推理 |
| `policy_config.py` | `create_trained_policy` 工厂函数：加载检查点、创建变换管道、返回 Policy |
| `aloha_policy.py` | ALOHA 双臂机器人的输入/输出变换（4 相机、14 维状态/动作） |
| `droid_policy.py` | DROID Franka 的输入/输出变换（3 相机、8 维状态/动作） |
| `libero_policy.py` | LIBERO 仿真的输入/输出变换（2 相机、8 维状态、7 维动作） |

#### training/ — 训练基础设施

| 文件 | 功能 |
|------|------|
| `config.py` | **核心配置文件**（~990 行）：定义所有 `TrainConfig`、`DataConfig`、命名配置（25+ 个） |
| `data_loader.py` | 数据加载：`TorchDataLoader`、`RLDSDataLoader`、`FakeDataset` |
| `checkpoints.py` | 检查点管理：Orbax `CheckpointManager`，保存/恢复 train_state |
| `optimizer.py` | 优化器：`CosineDecaySchedule`（warmup + 余弦衰减）、`AdamW`（梯度裁剪） |
| `weight_loaders.py` | 权重加载：`CheckpointWeightLoader`（加载并合并 LoRA）、`PaliGemmaWeightLoader` |
| `sharding.py` | FSDP 分片：`make_mesh`、`fsdp_sharding` |

#### transforms.py — 数据变换管道

定义训练和推理共用的变换管道：
- `Normalize` / `Unnormalize` — z-score 或分位数归一化
- `ResizeImages` — 图像缩放到 224x224
- `DeltaActions` / `AbsoluteActions` — 绝对/增量动作空间转换
- `TokenizePrompt` / `TokenizeFASTInputs` — 文本/动作分词
- `RepackTransform` — 字典键重映射
- `PadStatesAndActions` — 维度填充

#### serving/ — 策略服务器

| 文件 | 功能 |
|------|------|
| `websocket_policy_server.py` | `WebsocketPolicyServer`：异步 WebSocket 服务器，接收观测 → 运行推理 → 返回动作 |

#### shared/ — 共享工具

| 文件 | 功能 |
|------|------|
| `normalize.py` | `NormStats`（均值/标准差/分位数）、`RunningStats`（在线统计计算） |
| `download.py` | 检查点下载（GCS、本地路径） |
| `image_tools.py` | 图像缩放（带填充）、uint8 转换 |

### 2.3 入口脚本 `scripts/`

| 脚本 | 功能 | 启动命令 |
|------|------|----------|
| `train.py` | JAX 训练脚本（FSDP 分片、jit 编译） | `uv run scripts/train.py <config_name>` |
| `train_pytorch.py` | PyTorch 训练脚本（单卡/DDP/多节点） | `uv run scripts/train_pytorch.py <config_name>` |
| `serve_policy.py` | 策略服务器（WebSocket 推理服务） | `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<name> --policy.dir=<path>` |
| `compute_norm_stats.py` | 计算归一化统计 | `uv run scripts/compute_norm_stats.py --config-name=<name>` |

### 2.4 示例目录 `examples/`

| 目录 | 机器人 | 关键文件 |
|------|--------|----------|
| `aloha_real/` | ALOHA 双臂（实物） | 数据转换、环境封装、机器人工具、Docker |
| `aloha_sim/` | ALOHA 双臂（仿真） | 仿真环境、评估主程序 |
| `droid/` | DROID Franka | 数据转换、空闲帧过滤、全量训练说明 |
| `libero/` | LIBERO 仿真 Panda | 数据转换、评估脚本、Docker |
| `ur5/` | UR5 机械臂 | 设置说明 |
| `simple_client/` | 无机器人测试 | 随机观测推理、延迟测试 |
| `convert_jax_model_to_pytorch.py` | — | JAX → PyTorch 检查点转换 |

### 2.5 客户端包 `packages/openpi-client/`

轻量级客户端（Python >=3.7），用于连接策略服务器：
- `websocket_client_policy.py` — `WebsocketClientPolicy`：WebSocket 连接、msgpack 编解码
- `image_tools.py` — 客户端图像预处理
- `action_chunk_broker.py` — 动作块管理

### 2.6 依赖关系

```
openpi (核心包)
├── models/ ← 模型架构（JAX/Flax + PyTorch）
├── policies/ ← 依赖 models/ + transforms/
├── training/ ← 依赖 models/ + transforms/ + shared/
├── serving/ ← 依赖 policies/
├── transforms/ ← 独立的数据变换
└── shared/ ← 独立的工具函数

scripts/ ← 依赖以上所有模块
examples/ ← 依赖 policies/ + training/
packages/openpi-client/ ← 独立的轻量客户端
```

---

## 三、使用自己的数据

### 3.1 数据格式要求

openpi 使用 **LeRobot v2.0 数据集格式**（HuggingFace 的机器人数据标准）。不同机器人平台有各自的数据规格：

#### ALOHA 双臂机器人

| 项目 | 规格 |
|------|------|
| 图像尺寸 | 480x640 像素（HWC），4 个相机：`cam_high`、`cam_low`、`cam_left_wrist`、`cam_right_wrist` |
| 状态维度 | 14 维（每臂 7 维：6 关节角 + 1 夹爪） |
| 动作维度 | 14 维（同状态布局） |
| 关节单位 | 弧度，夹爪 [0.0, 1.0]（0=全开，1=全闭） |
| 控制频率 | 50 Hz |
| 数据来源 | 来自 `examples/aloha_real/convert_aloha_data_to_lerobot.py` |

#### DROID Franka 机器人

| 项目 | 规格 |
|------|------|
| 图像尺寸 | 180x320 像素（HWC），3 个相机：`exterior_image_1_left`、`exterior_image_2_left`、`wrist_image_left` |
| 状态维度 | 8 维（7 关节位置 + 1 夹爪位置） |
| 动作维度 | 8 维（7 关节速度 + 1 夹爪，或 7 关节位置 + 1 夹爪） |
| 控制频率 | 15 Hz |
| 数据来源 | 来自 `examples/droid/convert_droid_data_to_lerobot.py` |

#### LIBERO 仿真（Panda）

| 项目 | 规格 |
|------|------|
| 图像尺寸 | 256x256 像素（HWC），2 个相机：`image`（第三人称）、`wrist_image`（腕部） |
| 状态维度 | 8 维（末端位置 [3] + 轴角姿态 [3] + 夹爪关节 [2]） |
| 动作维度 | 7 维（增量位置 [3] + 增量姿态 [3] + 夹爪 [1]） |
| 控制频率 | 10 Hz |
| 数据来源 | 来自 `examples/libero/convert_libero_data_to_lerobot.py` |

**重要**: 所有图像在送入模型前会被统一缩放到 **224x224** 像素（由 `ResizeImages(224, 224)` 变换自动处理）。

### 3.2 LeRobot 数据集格式说明

LeRobot v2.0 数据集使用 HuggingFace `datasets` 库存储，包含：
- **Parquet 文件**: 表格/元数据（episode 索引、task 索引等）
- **图像/视频子目录**: 存储相机观测
- **元数据 JSON**: 描述数据集 schema

数据集通过 `LeRobotDataset.create()` 创建，关键参数：
- `repo_id`: HuggingFace 风格标识符（如 `"your_username/my_dataset"`）
- `robot_type`: 机器人平台名
- `fps`: 控制频率
- `features`: 字典，定义所有字段的 dtype、shape、names

### 3.3 数据转换流程

以 ALOHA 为例（来自 `examples/aloha_real/convert_aloha_data_to_lerobot.py`）：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 1. 创建数据集
dataset = LeRobotDataset.create(
    repo_id="your_username/my_aloha_data",
    robot_type="aloha",
    fps=50,
    features={
        "observation.images.cam_high": {"dtype": "image", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "action": {"dtype": "float32", "shape": (14,), "names": None},
    },
)

# 2. 逐帧添加数据（每个 episode）
for frame in episode_frames:
    dataset.add_frame({
        "observation.images.cam_high": frame["cam_high_image"],  # numpy array
        "observation.state": frame["qpos"],                      # 14-dim float32
        "action": frame["action"],                                # 14-dim float32
    })

# 3. 保存 episode
dataset.save_episode()

# 4.（可选）推送到 HuggingFace Hub
dataset.push_to_hub()
```

**原始数据来源**（各平台）：
- ALOHA: HDF5 文件（`episode_*.hdf5`），包含 `/observations/qpos`、`/action`、`/observations/images/{camera}`
- DROID: HDF5 轨迹文件（`trajectory.h5`）+ MP4 视频
- LIBERO: RLDS/TFDS 格式（TensorFlow Datasets）

**运行转换脚本**:
```bash
# ALOHA
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --data_dir /path/to/aloha/data

# LIBERO
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/libero/data

# DROID
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/droid/data
```

### 3.4 数据验证方法

1. **训练脚本自动验证**（来自 `scripts/train.py` 第 226-234 行）:
   - 训练开始时会打印第一个 batch 的数组形状和结构
   - 会将样本图像记录到 WandB 供视觉检查

2. **数据加载器验证**（来自 `src/openpi/training/data_loader.py`）:
   - 验证 batch size 不超过数据集大小
   - 验证归一化统计文件存在

3. **手动检查**:
   ```python
   from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
   dataset = LeRobotDataset("your_username/my_dataset")
   print(f"Episodes: {dataset.num_episodes}")
   print(f"Frames: {dataset.num_frames}")
   sample = dataset[0]
   print(f"Keys: {sample.keys()}")
   print(f"State shape: {sample['observation.state'].shape}")
   print(f"Action shape: {sample['action'].shape}")
   ```

4. **DROID 空闲帧过滤**（来自 `examples/droid/compute_droid_nonidle_ranges.py`）:
   - 自动检测并移除连续关节速度变化 < 1e-3 的空闲帧
   - 保留 >= 16 帧的非空闲片段

### 3.5 自定义数据推理示例

来自 `src/test/4InputSelfDataInferExample.py`，展示如何用自己的图像进行推理：

```python
from openpi.policies import policy_config
from openpi.training import config as _config

config = _config.get_config("pi05_droid")
policy = policy_config.create_trained_policy(config, "./checkpoints/pi05_droid")

example = {
    "observation/exterior_image_1_left": exterior_array,  # 224x224x3, uint8
    "observation/wrist_image_left": wrist_array,          # 224x224x3, uint8
    "observation/joint_position": joint_position,         # 7-dim float32
    "observation/gripper_position": gripper_position,     # 1-dim float32
    "prompt": "pick up the red cup",                       # 语言指令
}
actions = policy.infer(example)["actions"]  # shape: [action_horizon, 8]
```

---

## 四、训练自己的模型

### 4.1 训练入口与配置文件

**JAX 训练**（推荐用于全参数微调和 LoRA）:
```bash
# 计算归一化统计（训练前必须）
uv run scripts/compute_norm_stats.py --config-name=pi05_libero

# 启动训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

**PyTorch 训练**（支持单卡/DDP/多节点）:
```bash
# 单卡训练
uv run scripts/train_pytorch.py pi0_aloha_sim --exp_name my_pytorch_exp

# 多卡 DDP 训练
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name my_ddp_exp

# 断点续训
uv run scripts/train_pytorch.py pi0_aloha_sim --exp_name my_exp --resume
```

**PyTorch 训练前准备**（需要打补丁）:
```bash
uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

### 4.2 配置体系

所有训练配置定义在 `src/openpi/training/config.py`（~990 行），通过命名配置管理：

**核心配置类 `TrainConfig`** 关键字段：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `model` | 模型配置（Pi0Config） | pi0/pi0-FAST/pi0.5 |
| `data` | 数据配置（DataConfig 子类） | 机器人特定 |
| `batch_size` | 训练批次大小 | 32-256（因配置而异） |
| `num_train_steps` | 总训练步数 | 20,000-100,000 |
| `learning_rate` | 学习率 | 2.5e-5 ~ 5e-5 |
| `warmup_steps` | 学习率预热步数 | 1,000 |
| `weight_decay` | 权重衰减 | 0.0 |
| `save_interval` | 检查点保存间隔 | 5,000-10,000 |
| `resume` | 是否断点续训 | False |
| `fsdp_devices` | FSDP 设备数 | 1 |

**已提供的命名配置**（部分）：

| 配置名 | 模型 | 机器人 | 用途 |
|--------|------|--------|------|
| `pi0_aloha` | pi0 | ALOHA | 推理 |
| `pi05_aloha` | pi0.5 | ALOHA | 推理 |
| `pi0_droid` | pi0 | DROID | 推理 |
| `pi0_fast_droid` | pi0-FAST | DROID | 推理 |
| `pi05_droid` | pi0.5 | DROID | 推理/微调 |
| `pi0_libero` | pi0 | LIBERO | 微调 |
| `pi0_fast_libero` | pi0-FAST | LIBERO | 微调 |
| `pi05_libero` | pi0.5 | LIBERO | 微调 |
| `pi0_aloha_sim` | pi0 | ALOHA 仿真 | 微调 |
| `debug` | — | — | 调试用 |

### 4.3 关键超参数调整建议

| 参数 | 建议范围 | 说明 |
|------|----------|------|
| `batch_size` | 32-256 | 受限于 GPU 显存，RTX 4090 建议 32-64，A100 可用 128-256 |
| `learning_rate` | 1e-5 ~ 5e-5 | LoRA 微调可用较高学习率，全参数微调用较低值 |
| `num_train_steps` | 10,000-100,000 | 小数据集 10k-20k 步通常足够，大数据集需要更多 |
| `save_interval` | 1,000-10,000 | 频繁保存便于回退选择最佳检查点 |
| `weight_loader` | CheckpointWeightLoader / PaliGemmaWeightLoader | 从预训练 checkpoint 加载权重 |
| `lora_rank` | 16-64 | LoRA 微调时的低秩维度，越大越接近全参数微调 |

**GPU 显存优化**:
- 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 允许 JAX 使用 90% GPU 显存
- 使用 `fsdp_devices` 启用 FSDP 分片降低单卡显存需求
- PyTorch 训练支持梯度检查点（`gradient_checkpointing=True`）
- 降低 `batch_size` 或使用 LoRA 微调

### 4.4 归一化统计

训练前必须计算归一化统计（均值、标准差、分位数）：

```bash
uv run scripts/compute_norm_stats.py --config-name=pi05_libero
```

统计文件保存为 `norm_stats.json`，结构：
```json
{
  "norm_stats": {
    "state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
    "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
  }
}
```

**归一化方式**（来自 `src/openpi/training/config.py` 第 187 行）：
- pi0: z-score 归一化 → `(x - mean) / (std + 1e-6)`
- pi0.5 / pi0-FAST: 分位数归一化 → `(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0`

**可选：复用预训练统计**（来自 `docs/norm_stats.md`）：
如果你的机器人与预训练数据中的机器人匹配，可以复用已有的归一化统计：
```python
data=LeRobotAlohaDataConfig(
    assets=AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",  # ALOHA 机器人
    ),
)
```

可用的 asset_id: `trossen`（ALOHA）、`droid`（DROID Franka）、`franka`（非 DROID Franka）、`ur5e`、`ur5e_dual`、`arx`、`arx_mobile`、`fibocom_mobile`。

### 4.5 训练监控与输出

**WandB 监控**:
- 训练自动将指标记录到 Weights & Biases
- 记录内容：loss、reward、学习率、样本图像

**检查点输出**:
```
checkpoints/<config_name>/<experiment_name>/
├── <step>/           # 每个保存间隔的检查点目录
│   ├── params/       # 模型参数（Orbax 格式）
│   └── assets/       # 归一化统计等
└── ...
```

PyTorch 检查点使用 safetensors 格式保存。

---

## 五、调用已训练好的模型进行推理

### 5.1 直接推理（Python API）

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# 加载配置和检查点
config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 构造输入
example = {
    "observation/exterior_image_1_left": image_1,  # 224x224x3, uint8
    "observation/wrist_image_left": image_wrist,   # 224x224x3, uint8
    "observation/joint_position": joint_pos,        # 7-dim float32
    "observation/gripper_position": gripper_pos,    # 1-dim float32
    "prompt": "pick up the fork",
}

# 推理
action_chunk = policy.infer(example)["actions"]
# shape: [action_horizon, action_dim]，例如 [50, 8]
```

### 5.2 策略服务器模式（推荐用于部署）

**启动服务器**:
```bash
# 使用预训练 checkpoint
uv run scripts/serve_policy.py --env=DROID

# 或指定自定义 checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=checkpoints/pi05_droid/my_experiment/20000
```

服务器默认监听端口 8000，提供 `/healthz` 健康检查端点。

**客户端查询**:
```bash
# 先安装客户端包
cd packages/openpi-client && pip install -e .
```

```python
from openpi_client import image_tools, websocket_client_policy

# 初始化客户端
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

# 构造观测（图像在客户端预处理为 224x224 uint8）
observation = {
    "observation/image": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, 224, 224)
    ),
    "observation/wrist_image": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    ),
    "observation/state": state,  # 原始状态，归一化在服务端处理
    "prompt": task_instruction,
}

# 获取动作
action_chunk = client.infer(observation)["actions"]
# shape: [action_horizon, action_dim]
```

### 5.3 输入预处理要求

| 输入 | 预处理 |
|------|--------|
| 图像 | 缩放到 224x224，转为 uint8 HWC 格式。使用 `image_tools.resize_with_pad()` 保持宽高比 |
| 状态 | **不需要手动归一化**，服务端自动处理 |
| 语言指令 | 自然语言字符串，描述任务目标 |

### 5.4 输出解读

`policy.infer(example)` 返回字典，`"actions"` 键对应形状为 `[action_horizon, action_dim]` 的数组：
- **action_horizon**: 动作预测步数（通常为 50）
- **action_dim**: 动作维度（机器人特定，如 DROID 为 8）

通常不需要执行全部 50 步，而是每隔 N 步调用一次推理，中间开环执行预测的动作块。

---

## 六、修改并部署到自己的真实机器人设备

### 6.1 需要修改的代码部分

#### 输入/输出变换（必须修改）

创建自定义的 policy 变换类，参考现有实现：

| 文件 | 参考内容 |
|------|----------|
| `src/openpi/policies/droid_policy.py` | `DroidInputs` / `DroidOutputs` — 适用于单臂机械臂 |
| `src/openpi/policies/aloha_policy.py` | `AlohaInputs` / `AlohaOutputs` — 适用于双臂机器人 |
| `src/openpi/policies/libero_policy.py` | `LiberoInputs` / `LiberoOutputs` — 适用于仿真环境 |

关键修改点：
1. **`YourInputs` 类**: 将你的机器人观测数据映射到模型期望的格式
   - 图像键名和相机数量
   - 状态向量的维度和含义
   - 语言指令的处理
2. **`YourOutputs` 类**: 将模型输出的动作映射到你的机器人控制接口
   - 动作维度裁剪（模型输出 32 维，通常只用前 N 维）
   - 增量/绝对动作转换

#### 数据配置（必须修改）

在 `src/openpi/training/config.py` 中添加你的 `DataConfig` 子类：
```python
@dataclasses.dataclass
class YourRobotDataConfig(DataConfig):
    # 定义 repack 变换（原始数据键 → 模型输入键）
    # 定义数据变换（归一化、图像缩放等）
    # 定义模型变换（分词、填充等）
```

#### 训练配置（必须修改）

在 `config.py` 的 `_CONFIGS` 列表中添加你的 `TrainConfig`。

#### 通信接口（部署时必须修改）

参考 `examples/aloha_real/main.py`（实物部署主程序）和 `examples/aloha_real/env.py`（环境封装），需要实现：
- 机器人状态读取接口
- 动作执行接口
- 图像采集接口

### 6.2 URDF 文件是否足够？

**仅 URDF 文件不够**。URDF 定义了机器人的运动学/动力学结构，但 openpi 的 VLA 模型**不直接使用 URDF**。模型通过学习从图像和状态到动作的映射来工作，不依赖显式的运动学求解。

URDF 的作用：
- **仿真环境**（如 Isaac Lab、MuJoCo）中用于物理仿真
- **运动学验证**: 检查模型输出的关节角度是否在限位内
- **逆运动学**: 如果需要将末端执行器动作转换为关节动作

### 6.3 部署到真实机器人还需要什么

| 类别 | 所需材料 | 说明 |
|------|----------|------|
| **相机标定** | 相机内参矩阵、畸变系数 | 确保图像预处理与训练数据一致 |
| **手眼标定** | 相机到机器人基座的变换矩阵 | 确保视角匹配 |
| **关节限位** | 各关节的角度/速度/力矩限制 | 安全检查，防止损坏机器人 |
| **控制接口** | 关节位置/速度/力矩控制 API | 如 ROS/ROS2 话题、CAN 总线、以太网协议 |
| **传感器接口** | 关节编码器读取、夹爪状态读取 | 获取机器人当前状态 |
| **控制频率** | 确认控制频率与训练数据一致 | ALOHA 50Hz、DROID 15Hz、UR5e 20Hz |
| **动作空间** | 确认动作含义（位置/速度/力矩） | DROID 用关节速度，ALOHA 用关节位置 |

### 6.4 完整部署流程

```
1. 数据采集
   └── 使用你的机器人采集训练数据（图像 + 状态 + 动作 + 语言指令）

2. 数据转换
   └── 转换为 LeRobot 格式（参考 examples/ 下的转换脚本）

3. 定义配置
   ├── 创建 YourInputs/YourOutputs 变换类
   ├── 创建 YourRobotDataConfig
   └── 创建 TrainConfig

4. 计算归一化统计
   └── uv run scripts/compute_norm_stats.py --config-name=your_config

5. 训练
   └── uv run scripts/train.py your_config --exp-name=your_exp

6. 验证（仿真或回放）
   └── 检查模型输出的动作是否合理

7. 部署策略服务器
   └── uv run scripts/serve_policy.py policy:checkpoint --policy.config=your_config --policy.dir=checkpoints/your_ckpt

8. 编写机器人客户端
   ├── 安装 openpi-client: cd packages/openpi-client && pip install -e .
   ├── 实现图像采集和预处理
   ├── 实现状态读取
   ├── 连接策略服务器
   └── 实现动作执行

9. 安全检查
   ├── 先在低速度/小幅度下测试
   ├── 检查关节限位
   ├── 设置急停按钮
   └── 逐步增加动作幅度
```

### 6.5 参考部署示例

- **ALOHA 实物**: `examples/aloha_real/` — 包含完整的数据转换、环境封装、Docker 部署
- **ALOHA 仿真**: `examples/aloha_sim/` — MuJoCo 仿真环境
- **DROID**: `examples/droid/` — 数据转换、全量训练说明
- **LIBERO**: `examples/libero/` — 仿真评估、Docker 部署
- **UR5**: `examples/ur5/` — 设置说明
- **ROS/ROS2 桥接**: 来自网络搜索，社区用户通常使用 `rosbridge` 或自定义 ROS 节点连接 WebSocket 策略服务器（推断，未在官方代码中找到具体实现）

---

## 七、常见问题与调试建议

### 7.1 官方 Troubleshooting（来自 README.md）

| 问题 | 解决方案 |
|------|----------|
| `uv sync` 依赖冲突 | 删除虚拟环境 `rm -rf .venv`，重新 `uv sync`。确保 uv 为最新版 `uv self update` |
| 训练 GPU 显存不足 | 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`；使用 `--fsdp-devices <n>` 启用 FSDP；禁用 EMA |
| 策略服务器连接错误 | 检查服务器是否运行、端口是否正确、防火墙设置 |
| 训练时缺少归一化统计 | 先运行 `uv run scripts/compute_norm_stats.py --config-name=<name>` |
| 数据集下载失败 | 检查网络；HuggingFace 数据集需先登录 `huggingface-cli login` |
| CUDA/GPU 错误 | 验证 NVIDIA 驱动；Docker 中需安装 nvidia-container-toolkit；**不需要系统级 CUDA 库**（通过 uv 安装），可尝试卸载系统 CUDA 库避免冲突 |
| 导入错误 | 确保 `uv sync` 安装了所有依赖 |
| 动作维度不匹配 | 检查数据处理变换中的动作空间定义是否与机器人一致 |
| 训练 loss 发散 | 检查 `norm_stats.json` 中的 `q01`、`q99`、`std` 值；某些维度统计值过小会导致归一化后数值过大，可手动调整 |

### 7.2 补充问题

| 问题 | 解决方案 |
|------|----------|
| PyTorch 训练 loss 高于 JAX | PyTorch 默认使用 bfloat16 全精度训练，设置 `pytorch_training_precision="float32"` 可改善（来自 README PyTorch 章节） |
| 图像预处理不一致 | 确保推理时的图像缩放方式与训练时一致，使用 `image_tools.resize_with_pad()` |
| WebSocket 延迟过高 | 图像在客户端预处理后再发送（缩放到 224x224 + uint8），减少传输数据量 |
| JAX 和 PyTorch checkpoint 不互通 | 使用 `examples/convert_jax_model_to_pytorch.py` 转换 |
| 子模块克隆慢 | 使用 `git submodule update --init --recursive --depth 1` 浅克隆 |
| transformers 库打补丁后影响其他项目 | 运行 `uv cache clean transformers` 恢复（来自 README PyTorch 章节） |
| 训练数据中动作空间不匹配 | 使用 `DeltaActions` / `AbsoluteActions` 变换在绝对和增量动作间转换（来自 `src/openpi/transforms.py`） |

### 7.3 各阶段调试建议

**数据准备阶段**:
- 先用小数据集（几十个 episode）验证流程
- 可视化 LeRobot 数据集中的图像和动作
- 检查动作分布是否合理（无异常值）

**训练阶段**:
- 从 `debug` 配置开始，验证代码无误
- 使用 WandB 监控 loss 曲线
- 如果 loss 发散，降低学习率或增加 warmup
- 定期保存检查点，选择最佳检查点

**推理阶段**:
- 先用 `examples/simple_client/` 测试推理流程
- 检查输出动作的范围和趋势是否合理
- 对比不同检查点的推理结果

**部署阶段**:
- 先在仿真环境中验证
- 实机测试时设置低速度和关节限位保护
- 准备急停按钮
- 逐步增加任务复杂度

---

## 附录：关键文件速查

| 用途 | 文件路径 |
|------|----------|
| 官方文档 | `README.md` |
| 远程推理文档 | `docs/remote_inference.md` |
| 归一化统计文档 | `docs/norm_stats.md` |
| Docker 文档 | `docs/docker.md` |
| JAX 训练脚本 | `scripts/train.py` |
| PyTorch 训练脚本 | `scripts/train_pytorch.py` |
| 策略服务器 | `scripts/serve_policy.py` |
| 归一化统计计算 | `scripts/compute_norm_stats.py` |
| JAX→PyTorch 转换 | `examples/convert_jax_model_to_pytorch.py` |
| 训练配置（全部） | `src/openpi/training/config.py` |
| 模型定义（pi0） | `src/openpi/models/pi0.py` |
| 模型定义（pi0-FAST） | `src/openpi/models/pi0_fast.py` |
| PyTorch 模型 | `src/openpi/models_pytorch/pi0_pytorch.py` |
| 策略类 | `src/openpi/policies/policy.py` |
| 策略工厂 | `src/openpi/policies/policy_config.py` |
| 数据变换 | `src/openpi/transforms.py` |
| 归一化工具 | `src/openpi/shared/normalize.py` |
| ALOHA 策略变换 | `src/openpi/policies/aloha_policy.py` |
| DROID 策略变换 | `src/openpi/policies/droid_policy.py` |
| LIBERO 策略变换 | `src/openpi/policies/libero_policy.py` |
| ALOHA 数据转换 | `examples/aloha_real/convert_aloha_data_to_lerobot.py` |
| DROID 数据转换 | `examples/droid/convert_droid_data_to_lerobot.py` |
| LIBERO 数据转换 | `examples/libero/convert_libero_data_to_lerobot.py` |
| 自定义数据推理 | `src/test/4InputSelfDataInferExample.py` |
| WebSocket 客户端 | `packages/openpi-client/src/openpi_client/websocket_client_policy.py` |
