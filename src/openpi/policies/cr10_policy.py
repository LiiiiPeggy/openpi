"""CR10 robotic arm policy transforms for openpi.

Training uses LIBERO dataset (task-space actions). At inference, IK converts
task-space actions to CR10 joint-space actions.

Input observations (same as LIBERO):
  - observation/image: third-person camera (224x224x3)
  - observation/wrist_image: wrist camera (224x224x3)
  - observation/state: 8D (3 pos + 3 axis-angle + 2 gripper)
  - prompt: language instruction

Output actions:
  - Training: 7D task-space (dx, dy, dz, drx, dry, drz, gripper) — same as LIBERO
  - Inference: 7D joint-space (j1-j6, gripper) — converted via IK
"""

from __future__ import annotations

import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger(__name__)


def make_cr10_example() -> dict:
    """Creates a random input example for the CR10 policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the block",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CR10Inputs(transforms.DataTransformFn):
    """Converts CR10 observations to model input format.

    This is identical to LiberoInputs since we train on the LIBERO dataset.
    The CR10 and LIBERO share the same observation structure:
    - Third-person image (base camera)
    - Wrist camera image
    - 8D state (3 pos + 3 axis-angle + 2 gripper)
    - Language prompt
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CR10Outputs(transforms.DataTransformFn):
    """Converts model output to CR10 actions.

    At inference time, applies IK to convert task-space actions to joint-space.
    If no IK solver is provided, returns raw task-space actions (for training/evaluation).

    Note: main.py applies IK manually (outside the transform pipeline) because
    IK requires the current joint state which is updated step-by-step. This class
    is provided as a standalone utility for other integration patterns.
    """

    ik_solver: object | None = None
    current_joint_pos: np.ndarray | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :7])

        if self.ik_solver is None:
            return {"actions": actions}

        joint_actions = []
        current_pos = self.current_joint_pos.copy() if self.current_joint_pos is not None else np.zeros(6)

        for t in range(actions.shape[0]):
            delta_ee = actions[t, :6]
            gripper = actions[t, 6:7]

            try:
                target_joints = self.ik_solver.solve(current_pos, delta_ee)
            except Exception as e:
                logger.warning(f"IK failed at step {t}: {e}, using current joints")
                target_joints = current_pos.copy()

            joint_action = np.concatenate([target_joints, gripper])
            joint_actions.append(joint_action)
            current_pos = target_joints

        return {"actions": np.array(joint_actions)}
