"""CR10 arm inference with MuJoCo visualization.

Loads a trained pi0.5 model and runs inference in the CR10 MuJoCo environment.
Actions are converted from task-space to joint-space via IK.

Usage:
    # Real-time viewer
    python examples/cr10_sim/main.py --checkpoint checkpoints/pi05_cr10_libero/cr10_libero_v1

    # Record video
    python examples/cr10_sim/main.py --checkpoint checkpoints/pi05_cr10_libero/cr10_libero_v1 --video output.mp4

    # With custom prompt
    python examples/cr10_sim/main.py --checkpoint checkpoints/pi05_cr10_libero/cr10_libero_v1 --prompt "pick up the red block"
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CR10 arm inference with MuJoCo visualization")
    parser.add_argument("--config", type=str, default="pi05_cr10_libero", help="Training config name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="pick up the red block", help="Language instruction")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--video", type=str, default=None, help="Path to save video (e.g., output.mp4)")
    parser.add_argument("--viewer", action="store_true", help="Show real-time MuJoCo viewer")
    parser.add_argument("--ik-backend", type=str, default="numpy", choices=["numpy", "pykdl2", "moveit_kdl"],
                        help="IK solver backend")
    parser.add_argument("--urdf-path", type=str, default=None, help="Path to CR10 URDF file")
    parser.add_argument("--replan-steps", type=int, default=5, help="Steps to execute per action chunk")
    args = parser.parse_args()

    # --- 1. Load model ---
    logger.info(f"Loading model from config '{args.config}', checkpoint '{args.checkpoint}'")
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    config = _config.get_config(args.config)
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    logger.info("Model loaded successfully")

    # --- 2. Create IK solver ---
    from openpi.policies.cr10_ik import CR10IKSolver, get_default_cr10_urdf_path

    urdf_path = args.urdf_path or str(get_default_cr10_urdf_path())
    logger.info(f"Initializing IK solver (backend={args.ik_backend}, urdf={urdf_path})")

    base_offset = (0.0, 0.0, 0.5)  # CR10 base height in the MuJoCo scene
    ik_solver = CR10IKSolver(urdf_path, backend=args.ik_backend, base_offset=base_offset)

    # --- 3. Create environment ---
    from examples.cr10_sim.env import CR10SimEnv

    env = CR10SimEnv(
        xml_path=Path(__file__).parent / "cr10_scene.xml",
        max_steps=args.max_steps,
    )
    logger.info("CR10 MuJoCo environment created")

    # --- 4. Optional: set up viewer ---
    viewer = None
    if args.viewer:
        import mujoco
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        logger.info("MuJoCo viewer launched")

    # --- 5. Optional: set up video recording ---
    video_frames = []
    if args.video:
        logger.info(f"Recording video to {args.video}")

    # --- 6. Run inference loop ---
    current_joint_pos = np.zeros(6)

    for episode in range(args.num_episodes):
        logger.info(f"=== Episode {episode + 1}/{args.num_episodes} ===")
        obs, _ = env.reset()
        current_joint_pos = np.zeros(6)
        done = False
        step = 0
        total_reward = 0.0

        while not done:
            # Prepare observation for policy
            policy_obs = {
                "observation/image": obs["observation/image"],
                "observation/wrist_image": obs["observation/wrist_image"],
                "observation/state": obs["observation/state"],
                "prompt": args.prompt,
            }

            # Get action from model
            result = policy.infer(policy_obs)
            task_actions = result["actions"]  # shape: [action_horizon, 7]

            # Execute first replan_steps actions via IK
            steps_to_execute = min(args.replan_steps, task_actions.shape[0])
            for t in range(steps_to_execute):
                delta_ee = task_actions[t, :6]
                gripper = task_actions[t, 6]

                try:
                    target_joints = ik_solver.solve(current_joint_pos, delta_ee)
                    current_joint_pos = target_joints
                except Exception as e:
                    logger.warning(f"IK failed at step {step}: {e}")

                # Action: 6 joint angles + 1 gripper
                action = np.concatenate([current_joint_pos, [gripper]])
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                # Update viewer
                if viewer is not None:
                    viewer.sync()
                    time.sleep(0.02)  # ~50fps

                # Record frame
                if args.video:
                    frame = env.render_camera("third_person")
                    video_frames.append(frame.copy())

                if terminated or truncated:
                    done = True
                    break

        logger.info(f"Episode {episode + 1} finished: {step} steps, reward={total_reward:.2f}")

    # --- 7. Save video ---
    if args.video and video_frames:
        logger.info(f"Saving {len(video_frames)} frames to {args.video}")
        try:
            import imageio
            imageio.mimsave(args.video, video_frames, fps=30)
            logger.info(f"Video saved to {args.video}")
        except ImportError:
            logger.warning("imageio not installed. Install with: pip install imageio[ffmpeg]")
            # Fallback: save as individual frames
            out_dir = Path(args.video).with_suffix("")
            out_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(video_frames):
                from PIL import Image
                Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.png")
            logger.info(f"Frames saved to {out_dir}/")

    # --- 8. Cleanup ---
    env.close()
    if viewer is not None:
        viewer.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
