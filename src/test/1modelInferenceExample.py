import dataclasses
import jax
from pathlib import Path

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# 获取模型配置（pi05_droid配置）
config = _config.get_config("pi05_droid")

# ========== 修改：使用本地路径 ==========
# 原始路径：checkpoint_dir = Path("./checkpoints/pi0_fast_droid")
# 修改为本地已下载的checkpoint路径
checkpoint_dir = Path("./checkpoints/pi05_droid")

# 验证checkpoint路径是否存在，不存在则报错
assert checkpoint_dir.exists(), f"Checkpoint not found: {checkpoint_dir}"
print(f"Loading checkpoint from: {checkpoint_dir.absolute()}")

# 创建训练好的策略（加载模型权重和配置）
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# 使用虚拟示例运行推理（生成动作预测）
example = droid_policy.make_droid_example()
result = policy.infer(example)

# 删除策略对象以释放内存（GPU显存/JAX内存）
del policy

# 输出预测动作的形状（通常格式为 [时间步数, 动作维度]）
print("Actions shape:", result["actions"].shape)