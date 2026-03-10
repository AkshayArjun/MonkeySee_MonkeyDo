import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from geometry_msgs.msg import PoseArray
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize, Bounds

from . import rx200_kinematics as rx_kine

# Constants from the refined sim logic
LOOP_RATE = 10
ALPHA     = 0.67
BETA      = 0.33
GAMMA     = 2 * ALPHA  # Enhanced weight for hand position convergence

JOINT_NAMES = [
    'waist',
    'shoulder',
    'elbow',
    'wrist_angle',
    'wrist_rotate'
]

# Bounds using the more robust Scipy Bounds object
BOUNDS = Bounds(
    lb=[-3.1416, -1.8849, -1.8849, -1.7453, -3.1416],
    ub=[ 3.1416,  1.9722,  1.6231,  2.1467,  3.1416],
    keep_feasible=True
)

class OCRANode(Node):
    def __init__(self):
        super().__init__('ocra_controller')

        self.cb = ReentrantCallbackGroup()

        # --- Subscribers (Hardware Topics) ---
        self.target_sub = self.create_subscription(
            PoseArray,
            '/human/skeletal_data',
            self.human_callback,
            1,
            callback_group=self.cb
        )

        self.robot_sub = self.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.robot_state_callback,
            10,
            callback_group=self.cb
        )

        # --- Publisher (Hardware Command) ---
        self.cmd_pub = self.create_publisher(
            JointGroupCommand,
            '/rx200/commands/joint_group',
            10
        )

        # --- Publisher (Visualiser: commanded joints) ---
        # Publishes the OCRA solution as a JointState every time a new command
        # is dispatched, so the visualiser can show the orange commanded FK chain.
        self.vis_cmd_pub = self.create_publisher(
            JointState,
            '/ocra/commanded_joints',
            10
        )

        self.current_joints = np.zeros(5)
        self.last_solution = np.zeros(5)
        self.latest_target_flat = None
        self.first_solve = True

        # --- JAX JIT Warmup ---
        self._warmup_jit()

        self.timer = self.create_timer(
            1.0 / LOOP_RATE,
            self.control_loop,
            callback_group=self.cb
        )

        self.get_logger().info("OCRA Hardware Node Initialized with Sim-Logic updates.")

    def _warmup_jit(self):
        """Prevents NaNs and latency spikes on the first optimization call."""
        dummy_q = jnp.zeros(5)
        dummy_target = jnp.zeros(13) # 3(sh) + 3(el) + 3(ha) + 4(quat)
        dummy_w = jnp.array([ALPHA, BETA, GAMMA])
        _ = rx_kine.loss_and_grad_fn(dummy_q, dummy_target, dummy_w)
        self.get_logger().info("JAX JIT warmup complete.")

    def robot_state_callback(self, msg):
        """Improved name-based lookup to ensure joint order is correct."""
        name_to_pos = dict(zip(msg.name, msg.position))
        self.current_joints = np.array([
            name_to_pos.get(n, 0.0) for n in JOINT_NAMES
        ], dtype=np.float64)
        
    def human_callback(self, msg):
        if len(msg.poses) < 3:
            return
        
        shoulder = np.array([msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z])
        elbow    = np.array([msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z])
        hand     = np.array([msg.poses[2].position.x, msg.poses[2].position.y, msg.poses[2].position.z])
        
        quat = np.array([
            msg.poses[2].orientation.x,
            msg.poses[2].orientation.y,
            msg.poses[2].orientation.z,
            msg.poses[2].orientation.w])
        quat = quat / (np.linalg.norm(quat) + 1e-8)

        self.latest_target_flat = np.concatenate([shoulder, elbow, hand, quat])

    def control_loop(self):
        if self.latest_target_flat is None:
            return

        # Use current state for first solve, then warm-start with last solution
        x0 = self.current_joints if self.first_solve else self.last_solution
        max_iter = 50 if self.first_solve else 10

        # Refined loss function with sanitization logic from Sim
        def loss_fn(x):
            val, grad = rx_kine.loss_and_grad_fn(
                jnp.array(x),
                jnp.array(self.latest_target_flat),
                jnp.array([ALPHA, BETA, GAMMA])
            )
            val_np = float(val)
            grad_np = np.array(grad, dtype=np.float64)
            
            if not np.isfinite(val_np):
                val_np = 1e6
            grad_np = np.where(np.isfinite(grad_np), grad_np, 0.0)
            return val_np, grad_np

        res = minimize(
            fun=loss_fn,
            x0=np.array(x0, dtype=np.float64),
            method='SLSQP',
            jac=True,
            bounds=BOUNDS,
            options={'maxiter': max_iter, 'ftol': 1e-4}
        )

        if res.success or 'iteration' in res.message.lower():
            self.last_solution = res.x
            self.first_solve = False
            
            # Publish to Hardware
            cmd_msg = JointGroupCommand()
            cmd_msg.name = "arm"
            cmd_msg.cmd = res.x.tolist()
            self.cmd_pub.publish(cmd_msg)

            # Publish to visualiser
            vis_msg = JointState()
            vis_msg.header.stamp = self.get_clock().now().to_msg()
            vis_msg.name     = JOINT_NAMES
            vis_msg.position = res.x.tolist()
            self.vis_cmd_pub.publish(vis_msg)
        else:
            self.get_logger().warn(f"Optimization failed: {res.message}", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    ocra_node = OCRANode()
    executor = MultiThreadedExecutor()
    executor.add_node(ocra_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ocra_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()