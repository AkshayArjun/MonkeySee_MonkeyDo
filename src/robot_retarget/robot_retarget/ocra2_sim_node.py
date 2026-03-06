import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from scipy.optimize import minimize, Bounds
import numpy as np
import jax.numpy as jnp

from . import ocra_kinematics as ok

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


class OCRA2SimNode(Node):
    def __init__(self):
        super().__init__('ocra2_sim_node')

        self.target_sub = self.create_subscription(
            PoseArray, '/human/skeletal_data', self.human_callback, 1)
        self.robot_sub = self.create_subscription(
            JointState, '/joint_states', self.robot_state_callback, 10)

        # Publish to _raw — bridge picks this up and forwards via action server
        self.cmd_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory_raw', 10)

        self._lock              = threading.Lock()
        self.current_joints     = np.zeros(N_JOINTS)
        self.last_solution      = (_lower + _upper) / 2.0
        self.latest_target_flat = None
        self._opt_busy          = False
        self._first_solve       = True

        self.timer = self.create_timer(1.0 / LOOP_RATE, self.control_loop)
        self._warmup_jit()
        self.get_logger().info(f'OCRA2 Node Ready @ {LOOP_RATE} Hz. Joints: {JOINT_NAMES}')

    def _warmup_jit(self):
        dummy_q = jnp.zeros(ok.N_JOINTS)
        dummy_t = jnp.zeros(13).at[12].set(1.0)
        _ = ok.loss_and_grad(dummy_q, dummy_t, _WEIGHTS)
        self.get_logger().info('JAX JIT warmup complete.')

    def robot_state_callback(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        joints = np.array([name_to_pos.get(n, 0.0) for n in JOINT_NAMES], dtype=np.float64)
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

        target_snap = target.copy()
        x0_snap     = x0 if self._first_solve else self.last_solution.copy()

        self._opt_busy = True
        threading.Thread(target=self._solve, args=(x0_snap, target_snap), daemon=True).start()

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
                self._publish(res.x)
                self.get_logger().info(f'Loss: {res.fun:.4f}', throttle_duration_sec=1.0)
            else:
                self.get_logger().warn(f'Optimizer: {res.message}', throttle_duration_sec=2.0)
        finally:
            self._opt_busy = False

    def _publish(self, positions: np.ndarray):
        with self._lock:
            current = self.current_joints.tolist()

        msg              = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names  = JOINT_NAMES

        # pt0: current position anchor
        pt0             = JointTrajectoryPoint()
        pt0.positions   = current
        pt0.velocities  = [0.0] * N_JOINTS
        pt0.time_from_start = Duration(sec=0, nanosec=0)

        # pt1: OCRA target
        pt1             = JointTrajectoryPoint()
        pt1.positions   = positions.tolist()
        pt1.velocities  = [0.0] * N_JOINTS
        pt1.time_from_start = Duration(sec=0, nanosec=300_000_000)

        msg.points = [pt0, pt1]
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OCRA2SimNode()
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
