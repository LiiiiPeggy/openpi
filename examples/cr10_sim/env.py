"""CR10 MuJoCo simulation environment.

Gymnasium-compatible environment for the CR10 arm in a tabletop setting.
Observations match the LIBERO format for compatibility with pi0.5 models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# Joint names in the MuJoCo model
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_joint"]
NUM_ARM_JOINTS = 6

# Default joint positions (CR10 home pose)
DEFAULT_JOINT_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class CR10SimEnv(gym.Env):
    """CR10 robotic arm MuJoCo simulation environment.

    Observation space matches LIBERO format:
        - observation/image: third-person camera (224x224x3) uint8
        - observation/wrist_image: wrist camera (224x224x3) uint8
        - observation/state: 8D float32 (3 pos + 3 axis-angle + 2 gripper)

    Action space: 7D float32 (j1-j6 joint angles + gripper)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        render_mode: str = "rgb_array",
        image_size: tuple[int, int] = (224, 224),
        max_steps: int = 500,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.image_size = image_size
        self.max_steps = max_steps
        self._step_count = 0

        if xml_path is None:
            xml_path = Path(__file__).parent / "cr10_scene.xml"
        self.xml_path = Path(xml_path)

        self.model = mujoco.MjModel.from_xml_string(self.xml_path.read_text())
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(self.model, height=image_size[0], width=image_size[1])

        self._joint_ids = []
        for name in JOINT_NAMES:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self._joint_ids.append(jid)
            except Exception:
                logger.warning(f"Joint '{name}' not found in model")

        self._actuator_ids = []
        for name in [f"act_{n}" for n in JOINT_NAMES]:
            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self._actuator_ids.append(aid)
            except Exception:
                pass

        self._third_person_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "third_person"
        )
        self._wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )

        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "observation/image": gym.spaces.Box(0, 255, (*image_size, 3), np.uint8),
            "observation/wrist_image": gym.spaces.Box(0, 255, (*image_size, 3), np.uint8),
            "observation/state": gym.spaces.Box(-np.inf, np.inf, (8,), np.float32),
        })

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._step_count = 0

        mujoco.mj_resetData(self.model, self.data)

        for i, jid in enumerate(self._joint_ids):
            qpos_adr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_adr] = DEFAULT_JOINT_POS[i] if i < len(DEFAULT_JOINT_POS) else 0.0

        for i in range(NUM_ARM_JOINTS):
            if i < len(self._actuator_ids):
                self.data.ctrl[self._actuator_ids[i]] = DEFAULT_JOINT_POS[i]

        block_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
        if block_jid >= 0:
            qpos_adr = self.model.jnt_qposadr[block_jid]
            self.data.qpos[qpos_adr:qpos_adr+3] = [0.45, 0.1, 0.45]
            self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._step_count += 1

        # Clip joint angles to [-pi, pi], gripper to [0, 1]
        joint_action = np.clip(action[:6], -3.14, 3.14)
        gripper_action = np.clip(action[6:7], 0.0, 1.0) if len(action) > 6 else np.zeros(1)
        clipped = np.concatenate([joint_action, gripper_action])

        for i in range(min(len(clipped), len(self._actuator_ids))):
            self.data.ctrl[self._actuator_ids[i]] = clipped[i]

        # Multiple substeps for position control to converge
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> dict:
        third_person_img = self.render_camera("third_person")
        wrist_img = self.render_camera("wrist_cam")
        state = self._get_state()

        return {
            "observation/image": third_person_img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
        }

    def _get_state(self) -> np.ndarray:
        """Get 8D state: end-effector pos (3) + axis-angle (3) + gripper (2)."""
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        if ee_body_id < 0:
            return np.zeros(8, dtype=np.float32)

        pos = self.data.xpos[ee_body_id].copy()
        rot_mat = self.data.xmat[ee_body_id].reshape(3, 3).copy()
        axis_angle = self._rotation_matrix_to_axis_angle(rot_mat)

        gripper_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")
        if gripper_jid >= 0:
            qpos_adr = self.model.jnt_qposadr[gripper_jid]
            gripper_pos = self.data.qpos[qpos_adr]
            gripper_state = np.array([gripper_pos, gripper_pos])
        else:
            gripper_state = np.zeros(2)

        return np.concatenate([pos, axis_angle, gripper_state]).astype(np.float32)

    @staticmethod
    def _rotation_matrix_to_axis_angle(rot_mat: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to axis-angle representation."""
        trace = np.clip(np.trace(rot_mat), -1.0, 3.0)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

        if angle < 1e-6:
            return np.zeros(3)
        elif np.abs(angle - np.pi) < 1e-6:
            rx = np.sqrt(max((rot_mat[0, 0] + 1) / 2, 0))
            ry = np.sqrt(max((rot_mat[1, 1] + 1) / 2, 0))
            rz = np.sqrt(max((rot_mat[2, 2] + 1) / 2, 0))
            axis = np.array([rx, ry, rz])
            if rot_mat[0, 1] < 0:
                axis[1] = -axis[1]
            return axis * angle
        else:
            axis = np.array([
                rot_mat[2, 1] - rot_mat[1, 2],
                rot_mat[0, 2] - rot_mat[2, 0],
                rot_mat[1, 0] - rot_mat[0, 1],
            ])
            axis = axis / (2 * np.sin(angle))
            return axis * angle

    def render_camera(self, camera_name: str) -> np.ndarray:
        """Render an image from the specified camera."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.renderer.update_scene(self.data, camera=cam_id)
        return self.renderer.render()

    def render(self) -> np.ndarray:
        return self.render_camera("third_person")

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()
