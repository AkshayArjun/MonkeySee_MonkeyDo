"""
ocra_addverb.py

Full retargeting node for the Addverb Heal cobot (real hardware).

Arm:
  /human/skeletal_data (PoseArray) → OCRA SLSQP → joint angles
  Sends directly to /joint_impedance_controller/follow_joint_trajectory
  as a single-point PTP goal. The controller handles motion planning internally.
  Reads joint states from /joint_states.
  Publishes commanded joint angles to /ocra/commanded_joints (JointState)
  for visualisation — updated every time a new IK solution is dispatched.

Gripper:
  /human/gripper_cmd (Bool) → GripperCommand action
  Sends directly to /gripper_controller/gripper_cmd.

Parallelisation rationale:
  - Gripper: pure event-driven on state change, near-zero CPU, runs on executor thread.
  - Arm IK: SLSQP + JAX runs in a daemon thread per solve so the ROS executor
    never blocks. Lock is only held for short array copies, never during computation,
    so no contention between arm solver thread and gripper callback.
  - MultiThreadedExecutor (2 threads): arm and gripper subscriptions can fire
    concurrently without one blocking the other.

Action client notes:
  - robot_hardware_bridge.py is no longer needed.
  - Single-point trajectory = PTP mode on the joint impedance controller.
    The controller's own motion planner handles smooth movement to the target.
  - _arm_busy flag mirrors the controller's own ongoing_prev_traj_ state —
    the solver won't start a new IK solve while the controller is executing.
  - The controller rejects cancellation, so goals always run to completion.
  - Dynamic time uses whole-integer seconds (no nanoseconds) to avoid any
    float precision issues with the controller's 100Hz delta_t validation.
  - goal_time_tolerance matches the manufacturer CLI example: {sec:1, nanosec:4}.
"""

import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Bool
from control_msgs.action import FollowJointTrajectory, GripperCommand
from scipy.optimize import minimize, Bounds
import numpy as np
import jax.numpy as jnp

from . import ocra_kinematics as ok

# ── Arm config ────────────────────────────────────────────────────────────────
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
N_JOINTS    = len(JOINT_NAMES)

LOOP_RATE = 20
ALPHA     = 0.6
BETA      = 0.4
GAMMA     = 1.2

_lower    = np.array(ok.robot.joints.lower_limits, dtype=np.float64)[:N_JOINTS]
_upper    = np.array(ok.robot.joints.upper_limits, dtype=np.float64)[:N_JOINTS]
ARM_BOUNDS = Bounds(lb=_lower, ub=_upper, keep_feasible=True)

_ARM_IDX = jnp.arange(N_JOINTS)
_WEIGHTS  = jnp.array([ALPHA, BETA, GAMMA])

# ── Arm timing config ─────────────────────────────────────────────────────────
# Conservative max velocity: 1.57 rad / 15 sec ≈ 0.1047 rad/s
V_MAX      = 1.57 / 15.0
MIN_TIME_S = 2   # minimum seconds per PTP goal (whole integer)

# ── Gripper config ────────────────────────────────────────────────────────────
GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 1.0
GRIPPER_EFFORT = 50.0


class OcraAddverbNode(Node):
    def __init__(self):
        super().__init__('ocra_addverb')

        # ── Arm: direct action client to joint impedance controller ───────────
        self._arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_impedance_controller/follow_joint_trajectory'
        )
        self._arm_goal: ClientGoalHandle | None = None
        self._arm_busy  = False   # True while controller is executing a goal
        self._arm_ready = False

        # ── Gripper: direct action client to gripper controller ───────────────
        self._gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )
        self._gripper_goal: ClientGoalHandle | None = None
        self._gripper_busy  = False
        self._gripper_ready = False

        # ── Arm subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            PoseArray, '/human/skeletal_data', self.human_callback, 1)
        self.create_subscription(
            JointState, '/joint_states', self.robot_state_callback, 10)

        # ── Gripper subscription ──────────────────────────────────────────────
        self.create_subscription(
            Bool, '/human/gripper_cmd', self._on_gripper_cmd, 10)

        # ── Commanded joints publisher (for visualiser) ───────────────────────
        # Publishes the OCRA solution every time a new goal is dispatched.
        # Visualiser subscribes to /ocra/commanded_joints to show orange FK chain.
        self._cmd_pub = self.create_publisher(
            JointState, '/ocra/commanded_joints', 10)

        # ── Shared state (lock for arm only — gripper is event-driven) ────────
        self._lock              = threading.Lock()
        self.current_joints     = np.zeros(N_JOINTS)
        self.last_solution      = (_lower + _upper) / 2.0
        self.latest_target_flat = None
        self._opt_busy          = False
        self._first_solve       = True

        # Gripper state — no lock needed, only touched in one callback
        self._gripper_state: bool | None = None

        # ── Server readiness check ────────────────────────────────────────────
        self._check_timer = self.create_timer(1.0, self._check_servers)

        self.timer = self.create_timer(1.0 / LOOP_RATE, self.control_loop)
        self._warmup_jit()

        self.get_logger().info(
            f'ocra_addverb ready @ {LOOP_RATE} Hz.\n'
            f'  Arm    : /human/skeletal_data'
            f' → /joint_impedance_controller/follow_joint_trajectory\n'
            f'  Cmd vis: /ocra/commanded_joints\n'
            f'  Gripper: /human/gripper_cmd'
            f' → /gripper_controller/gripper_cmd'
        )

    # ── Server readiness ──────────────────────────────────────────────────────
    def _check_servers(self):
        if not self._arm_ready and self._arm_client.server_is_ready():
            self._arm_ready = True
            self.get_logger().info('Arm action server ready.')
        if not self._gripper_ready and self._gripper_client.server_is_ready():
            self._gripper_ready = True
            self.get_logger().info('Gripper action server ready.')
        if self._arm_ready and self._gripper_ready:
            self._check_timer.cancel()

    # ── JIT warmup ────────────────────────────────────────────────────────────
    def _warmup_jit(self):
        dummy_q = jnp.zeros(ok.N_JOINTS)
        dummy_t = jnp.zeros(13).at[12].set(1.0)
        _ = ok.loss_and_grad(dummy_q, dummy_t, _WEIGHTS)
        self.get_logger().info('JAX JIT warmup complete.')

    # ── Arm callbacks ─────────────────────────────────────────────────────────
    def robot_state_callback(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        joints = np.array(
            [name_to_pos.get(n, 0.0) for n in JOINT_NAMES], dtype=np.float64)
        with self._lock:
            self.current_joints = joints

    def human_callback(self, msg: PoseArray):
        if len(msg.poses) < 3:
            return
        sh = msg.poses[0]; el = msg.poses[1]; ha = msg.poses[2]
        shoulder = np.array([sh.position.x, sh.position.y, sh.position.z])
        elbow    = np.array([el.position.x, el.position.y, el.position.z])
        hand     = np.array([ha.position.x, ha.position.y, ha.position.z])
        quat     = np.array([ha.orientation.x, ha.orientation.y,
                             ha.orientation.z,  ha.orientation.w])
        quat    /= (np.linalg.norm(quat) + 1e-8)
        with self._lock:
            self.latest_target_flat = np.concatenate([shoulder, elbow, hand, quat])

    # ── Arm control loop ──────────────────────────────────────────────────────
    def control_loop(self):
        with self._lock:
            target = self.latest_target_flat
            x0     = self.current_joints.copy()

        if target is None:
            self.get_logger().info(
                'Waiting for /human/skeletal_data...', throttle_duration_sec=5.0)
            return

        if self._opt_busy:
            return

        # Don't solve while the controller is still executing a goal —
        # it will reject any new goal anyway (ongoing_prev_traj_ guard in C++)
        if self._arm_busy:
            return

        target_snap = target.copy()
        x0_snap     = x0 if self._first_solve else self.last_solution.copy()

        self._opt_busy = True
        threading.Thread(
            target=self._solve, args=(x0_snap, target_snap), daemon=True
        ).start()

    def _solve(self, x0: np.ndarray, target_flat: np.ndarray):
        try:
            target_jax = jnp.array(target_flat)

            def loss_fn(q_np):
                full_q = jnp.zeros(ok.N_JOINTS).at[_ARM_IDX].set(jnp.array(q_np))
                val, grad = ok.loss_and_grad(full_q, target_jax, _WEIGHTS)
                val_np  = float(val)
                grad_np = np.array(grad, dtype=np.float64)[:N_JOINTS]
                if not np.isfinite(val_np):
                    val_np = 1e6
                grad_np = np.where(np.isfinite(grad_np), grad_np, 0.0)
                return val_np, grad_np

            res = minimize(
                fun=loss_fn,
                x0=np.array(x0, dtype=np.float64),
                method='SLSQP',
                jac=True,
                bounds=ARM_BOUNDS,
                options={'maxiter': 5, 'ftol': 1e-2}
            )

            if res.success or 'iteration' in res.message.lower():
                with self._lock:
                    self.last_solution = res.x
                    self._first_solve  = False
                self._send_arm_goal(res.x)
                self.get_logger().info(
                    f'Loss: {res.fun:.4f}', throttle_duration_sec=1.0)
            else:
                self.get_logger().warn(
                    f'Optimizer: {res.message}', throttle_duration_sec=2.0)
        finally:
            self._opt_busy = False

    # ── Arm action ────────────────────────────────────────────────────────────
    def _send_arm_goal(self, positions: np.ndarray):
        if not self._arm_ready:
            self.get_logger().warn(
                'Arm action server not ready, dropping.',
                throttle_duration_sec=2.0)
            return

        if self._arm_busy:
            return  # controller still executing, don't send

        with self._lock:
            current = np.array(self.current_joints)

        # ── Dynamic time: whole-integer seconds, nanosec always 0 ────────────
        # Single-point = PTP mode. Controller plans the motion internally.
        # Whole integers avoid all float precision issues with the controller's
        # 100Hz delta_t validation.
        max_distance = float(np.max(np.abs(positions - current)))
        total_sec    = max(MIN_TIME_S, int(np.ceil(max_distance / V_MAX)))

        # ── Single target point ───────────────────────────────────────────────
        pt = JointTrajectoryPoint()
        pt.positions       = positions.tolist()
        pt.velocities      = [0.0] * N_JOINTS
        pt.time_from_start = Duration(sec=total_sec, nanosec=0)

        traj             = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        traj.points      = [pt]

        goal                    = FollowJointTrajectory.Goal()
        goal.trajectory         = traj
        # Match manufacturer CLI example exactly
        goal.goal_time_tolerance = Duration(sec=1, nanosec=4)

        # ── Publish commanded joints for visualiser ───────────────────────────
        cmd_msg          = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name     = JOINT_NAMES
        cmd_msg.position = positions.tolist()
        self._cmd_pub.publish(cmd_msg)

        self._arm_busy = True
        future = self._arm_client.send_goal_async(goal)
        future.add_done_callback(self._on_arm_response)

        self.get_logger().info(
            f'Arm goal sent: {total_sec}s to '
            f'{[round(p, 3) for p in positions.tolist()]} '
            f'(max dist: {max_distance:.3f} rad)',
            throttle_duration_sec=1.0)

    def _on_arm_response(self, future):
        handle = future.result()
        if handle.accepted:
            self._arm_goal = handle
            # Listen for result so we know exactly when the controller is done
            handle.get_result_async().add_done_callback(self._on_arm_result)
        else:
            self.get_logger().warn('Arm goal rejected.')
            self._arm_busy = False  # free up immediately on rejection

    def _on_arm_result(self, future):
        self._arm_busy = False
        self._arm_goal = None
        result = future.result().result
        self.get_logger().info(
            f'Arm goal finished. Code: {result.error_code}',
            throttle_duration_sec=1.0)

    # ── Gripper action ────────────────────────────────────────────────────────
    def _on_gripper_cmd(self, msg: Bool):
        if msg.data == self._gripper_state:
            return  # no state change, nothing to do
        self._gripper_state = msg.data

        if not self._gripper_ready:
            self.get_logger().warn(
                'Gripper action server not ready, dropping.',
                throttle_duration_sec=2.0)
            return

        if self._gripper_busy:
            return

        goal          = GripperCommand.Goal()
        goal.command.position   = GRIPPER_OPEN if msg.data else GRIPPER_CLOSED
        goal.command.max_effort = GRIPPER_EFFORT

        self._gripper_busy = True
        future = self._gripper_client.send_goal_async(goal)
        future.add_done_callback(self._on_gripper_response)

        self.get_logger().info(
            f'Gripper → {"OPEN" if msg.data else "CLOSE"} ({goal.command.position:.3f} m)')

    def _on_gripper_response(self, future):
        handle = future.result()
        if handle.accepted:
            self._gripper_goal = handle
            handle.get_result_async().add_done_callback(self._on_gripper_result)
        else:
            self.get_logger().warn('Gripper goal rejected.')
            self._gripper_busy = False

    def _on_gripper_result(self, future):
        self._gripper_busy = False
        self._gripper_goal = None


def main(args=None):
    rclpy.init(args=args)
    node = OcraAddverbNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()