"""CR10 robotic arm Inverse Kinematics solver.

Supports two backends:
- pykdl2 (default): Pure Python KDL bindings, no ROS dependency
- moveit_kdl: Uses MoveIt's KDL plugin via ROS (requires ROS environment)

CR10 kinematic chain (from URDF):
  base_link -> joint1(Z, d=0.1765) -> Link1 -> joint2(Z, rpy=pi/2,pi/2,0) -> Link2
  -> joint3(Z, a=-0.607) -> Link3 -> joint4(Z, a=-0.568, d=0.191, rpy=0,0,-pi/2) -> Link4
  -> joint5(Z, a=-0.125, rpy=pi/2,0,0) -> Link5 -> joint6(Z, d=0.1084, rpy=-pi/2,0,0) -> Link6
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# CR10 joint limits (from rangercr10lidar.urdf)
JOINT_LIMITS_LOWER = np.array([-3.92, -1.57, -2.86, -3.14, -3.14, -3.14])
JOINT_LIMITS_UPPER = np.array([0.94, 1.57, 2.86, 3.14, 3.14, 3.14])


class CR10IKSolver:
    """IK solver for the CR10 6-DOF robotic arm.

    Args:
        urdf_path: Path to the CR10 URDF file.
        backend: IK backend to use. "pykdl2" (default) or "moveit_kdl".
        base_offset: XYZ offset of the arm base from world origin [x, y, z].
    """

    def __init__(
        self,
        urdf_path: str | Path,
        backend: str = "pykdl2",
        base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.urdf_path = Path(urdf_path)
        self.backend = backend
        self.base_offset = np.array(base_offset)
        self._chain = None
        self._fk_solver = None
        self._ik_solver = None

        if backend == "pykdl2":
            self._init_pykdl()
        elif backend == "moveit_kdl":
            self._init_moveit_kdl()
        else:
            raise ValueError(f"Unknown IK backend: {backend}. Use 'pykdl2' or 'moveit_kdl'.")

    def _init_pykdl(self) -> None:
        """Initialize KDL chain from URDF using pykdl2."""
        try:
            import PyKDL
            import urdf_parser_py.urdf as urdf
        except ImportError as e:
            raise ImportError(
                "pykdl2 and urdf_parser_py are required for the pykdl2 backend. "
                "Install with: pip install pykdl2 urdf-parser-py"
            ) from e

        robot = urdf.URDF.from_xml_file(str(self.urdf_path))
        chain = PyKDL.Chain()
        base_frame = PyKDL.Frame(PyKDL.Vector(*self.base_offset))
        no_joint = getattr(PyKDL.Joint, "None")
        chain.addSegment(PyKDL.Segment(PyKDL.Joint(no_joint), base_frame))

        for i in range(1, 7):
            joint_name = f"cr10_joint{i}"
            urdf_joint = next(j for j in robot.joints if j.name == joint_name)
            origin = urdf_joint.origin
            xyz = [origin[0], origin[1], origin[2]] if origin is not None else [0, 0, 0]
            rpy = [origin[3], origin[4], origin[5]] if origin is not None and len(origin) > 3 else [0, 0, 0]

            frame = _rpy_to_kdl_frame(xyz, rpy)
            kdl_joint = PyKDL.Joint(PyKDL.Joint.RotZ)
            chain.addSegment(PyKDL.Segment(kdl_joint, frame))

        ee_offset = PyKDL.Frame(PyKDL.Vector(0, 0.1084, 0))
        chain.addSegment(PyKDL.Segment(PyKDL.Joint(no_joint), ee_offset))

        self._chain = chain
        self._fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
        self._ik_solver = PyKDL.ChainIkSolverPos_LMA(chain)
        logger.info("Initialized pykdl2 IK solver for CR10")

    def _init_moveit_kdl(self) -> None:
        """Initialize MoveIt KDL backend (requires ROS environment)."""
        try:
            import rospy
            from moveit_commander import MoveGroupCommander
        except ImportError as e:
            raise ImportError(
                "ROS and MoveIt are required for the moveit_kdl backend. "
                "Source your ROS workspace first."
            ) from e

        if not rospy.is_initialized():
            rospy.init_node("cr10_ik_solver", anonymous=True)
        self._move_group = MoveGroupCommander("arm")
        self._move_group.set_planning_time(0.05)
        logger.info("Initialized MoveIt KDL IK solver for CR10")

    def solve(
        self,
        current_joint_pos: np.ndarray,
        delta_ee_pose: np.ndarray,
    ) -> np.ndarray:
        """Solve IK for a delta end-effector pose.

        Args:
            current_joint_pos: Current joint angles [6].
            delta_ee_pose: Delta end-effector pose [dx, dy, dz, drx, dry, drz] in radians.

        Returns:
            Target joint angles [6], clamped to joint limits.
        """
        if self.backend == "pykdl2":
            return self._solve_pykdl(current_joint_pos, delta_ee_pose)
        else:
            return self._solve_moveit(current_joint_pos, delta_ee_pose)

    def solve_cartesian(
        self,
        current_joint_pos: np.ndarray,
        target_ee_pose: np.ndarray,
    ) -> np.ndarray:
        """Solve IK for an absolute end-effector pose.

        Args:
            current_joint_pos: Current joint angles [6].
            target_ee_pose: Target end-effector pose [x, y, z, rx, ry, rz].

        Returns:
            Target joint angles [6], clamped to joint limits.
        """
        current_ee = self.fk(current_joint_pos)
        delta = target_ee_pose - current_ee
        return self.solve(current_joint_pos, delta)

    def fk(self, joint_pos: np.ndarray) -> np.ndarray:
        """Compute forward kinematics.

        Args:
            joint_pos: Joint angles [6].

        Returns:
            End-effector pose [x, y, z, rx, ry, rz].
        """
        if self.backend == "pykdl2":
            return self._fk_pykdl(joint_pos)
        else:
            return self._fk_moveit(joint_pos)

    def _solve_pykdl(self, current_joint_pos: np.ndarray, delta_ee_pose: np.ndarray) -> np.ndarray:
        """Solve IK using pykdl2."""
        import PyKDL

        current_frame = PyKDL.Frame()
        q_current = PyKDL.JntArray(6)
        for i in range(6):
            q_current[i] = current_joint_pos[i]
        self._fk_solver.JntToCart(q_current, current_frame)

        delta_frame = PyKDL.Frame(
            PyKDL.Rotation.RPY(delta_ee_pose[3], delta_ee_pose[4], delta_ee_pose[5]),
            PyKDL.Vector(delta_ee_pose[0], delta_ee_pose[1], delta_ee_pose[2]),
        )
        target_frame = current_frame * delta_frame

        q_result = PyKDL.JntArray(6)
        q_init = PyKDL.JntArray(6)
        for i in range(6):
            q_init[i] = current_joint_pos[i]

        result = self._ik_solver.CartToJnt(q_init, target_frame, q_result)
        if result < 0:
            logger.warning("pykdl2 IK failed, falling back to Jacobian pseudo-inverse")
            return self._solve_jacobian_pinv(current_joint_pos, delta_ee_pose)

        joints = np.array([q_result[i] for i in range(6)])
        return np.clip(joints, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    def _solve_jacobian_pinv(
        self, current_joint_pos: np.ndarray, delta_ee_pose: np.ndarray
    ) -> np.ndarray:
        """Fallback IK using Jacobian pseudo-inverse with small steps and convergence check."""
        import PyKDL

        # Compute target frame from current FK + delta
        q_current_kdl = PyKDL.JntArray(6)
        for i in range(6):
            q_current_kdl[i] = current_joint_pos[i]
        current_frame = PyKDL.Frame()
        self._fk_solver.JntToCart(q_current_kdl, current_frame)

        delta_frame = PyKDL.Frame(
            PyKDL.Rotation.RPY(delta_ee_pose[3], delta_ee_pose[4], delta_ee_pose[5]),
            PyKDL.Vector(delta_ee_pose[0], delta_ee_pose[1], delta_ee_pose[2]),
        )
        target_frame = current_frame * delta_frame

        q = current_joint_pos.copy()
        n_steps = 10
        step = delta_ee_pose.copy() / n_steps

        for _ in range(n_steps):
            q_kdl = PyKDL.JntArray(6)
            for i in range(6):
                q_kdl[i] = q[i]

            # Compute Jacobian
            jac = PyKDL.Jacobian(6)
            jac_solver = PyKDL.ChainJntToJacSolver(self._chain)
            jac_solver.JntToJac(q_kdl, jac)

            jac_np = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    jac_np[i, j] = jac[i, j]

            dq = np.linalg.pinv(jac_np) @ step
            q += dq
            q = np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

            # Check convergence: FK error below 1mm threshold
            frame = PyKDL.Frame()
            for i in range(6):
                q_kdl[i] = q[i]
            self._fk_solver.JntToCart(q_kdl, frame)
            pos_err = np.linalg.norm([
                frame.p[0] - target_frame.p[0],
                frame.p[1] - target_frame.p[1],
                frame.p[2] - target_frame.p[2],
            ])
            if pos_err < 1e-3:
                break

        return q

    def _solve_moveit(self, current_joint_pos: np.ndarray, delta_ee_pose: np.ndarray) -> np.ndarray:
        """Solve IK using MoveIt KDL."""
        import geometry_msgs.msg as geom_msg

        current_ee = self._fk_moveit(current_joint_pos)
        target_pos = current_ee[:3] + delta_ee_pose[:3]
        target_rpy = current_ee[3:] + delta_ee_pose[3:]

        pose = geom_msg.PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = target_pos[0]
        pose.pose.position.y = target_pos[1]
        pose.pose.position.z = target_pos[2]

        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(*target_rpy)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self._move_group.set_pose_target(pose)
        plan = self._move_group.plan()
        if plan[0]:
            trajectory = plan[1]
            last_point = trajectory.joint_trajectory.points[-1]
            joints = np.array(last_point.positions)
            return np.clip(joints, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
        else:
            logger.warning("MoveIt IK failed, returning current joints")
            return current_joint_pos.copy()

    def _fk_pykdl(self, joint_pos: np.ndarray) -> np.ndarray:
        """Compute FK using pykdl2."""
        import PyKDL

        q = PyKDL.JntArray(6)
        for i in range(6):
            q[i] = joint_pos[i]
        frame = PyKDL.Frame()
        self._fk_solver.JntToCart(q, frame)

        pos = np.array([frame.p[0], frame.p[1], frame.p[2]])
        rpy = np.array(frame.M.GetRPY())
        return np.concatenate([pos, rpy])

    def _fk_moveit(self, joint_pos: np.ndarray) -> np.ndarray:
        """Compute FK using MoveIt."""
        joint_names = [f"joint{i}" for i in range(1, 7)]
        self._move_group.set_joint_value_target(dict(zip(joint_names, joint_pos.tolist())))
        pose = self._move_group.get_current_pose().pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])

        import tf.transformations as tft
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        rpy = np.array(tft.euler_from_quaternion(q))
        return np.concatenate([pos, rpy])


def _rpy_to_kdl_frame(xyz: list[float], rpy: list[float]):
    """Convert XYZ + RPY to a KDL Frame."""
    import PyKDL

    rot = PyKDL.Rotation.RPY(rpy[0], rpy[1], rpy[2])
    vec = PyKDL.Vector(xyz[0], xyz[1], xyz[2])
    return PyKDL.Frame(rot, vec)


def get_default_cr10_urdf_path() -> Path:
    """Get the default CR10 URDF path from the openpi project."""
    candidates = [
        Path(__file__).parent.parent.parent.parent / "rangerboxcr10lidar_description" / "urdf" / "rangercr10lidar.urdf",
        Path.home() / "locomani" / "openpi" / "rangerboxcr10lidar_description" / "urdf" / "rangercr10lidar.urdf",
        Path.home() / "locomani" / "agx" / "TCP-IP-ROS-6AXis" / "dobot_description" / "urdf" / "cr10_robot.urdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find CR10 URDF file. Please provide the path explicitly.")
