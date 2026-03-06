import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from scipy.optimize import minimize, Bounds
import numpy as np
import jax.numpy as jnp

from . import ocra_kinematics as ok

# ── Controller Configuration ──────────────────────────────────────────────────
# Explicitly match the active 'arm_controller' joint list
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
N_JOINTS    = len(JOINT_NAMES)

LOOP_RATE = 10       
ALPHA     = 0.6      
BETA      = 0.4     
GAMMA     = 0.0      

# Use limits from the URDF for these specific joints
_lower = np.array(ok.robot.joints.lower_limits, dtype=np.float64)[:N_JOINTS]
_upper = np.array(ok.robot.joints.upper_limits, dtype=np.float64)[:N_JOINTS]
ARM_BOUNDS = Bounds(lb=_lower, ub=_upper, keep_feasible=True)

class OCRA2SimNode(Node):
    def __init__(self):
        super().__init__('ocra2_sim_node')

        self.target_sub = self.create_subscription(
            PoseArray,
            '/human/skeletal_data',
            self.human_callback,
            1
        )

        self.robot_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.robot_state_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.current_joints = np.zeros(N_JOINTS)
        self.last_solution = np.zeros(N_JOINTS)
        self.latest_target_flat = None
        self.first_solve = True

        self.last_publish_time  = self.get_clock().now()
        self.min_publish_interval = 1.0 / LOOP_RATE
        
        self.timer = self.create_timer(1.0 / LOOP_RATE, self.control_loop)
        self._warmup_jit()

        self.get_logger().info(f"OCRA2 Node Ready. Controlling: {JOINT_NAMES}")

    def _warmup_jit(self):
        # Match the 6-DOF input expected by the updated kinematics
        dummy_q      = jnp.zeros(ok.N_JOINTS) 
        dummy_target = jnp.zeros(13).at[12].set(1.0)
        dummy_w      = jnp.array([ALPHA, BETA, GAMMA])
        _ = ok.loss_and_grad(dummy_q, dummy_target, dummy_w)
        self.get_logger().info("JAX JIT warmup complete.")
    
    def robot_state_callback(self, msg):
        # Map incoming joint states to our active joint order
        name_to_pos = dict(zip(msg.name, msg.position))
        self.current_joints = np.array([
            name_to_pos.get(n, 0.0) for n in JOINT_NAMES
        ], dtype=np.float64)

    def human_callback(self, msg: PoseArray):
        if len(msg.poses) < 3: return
        
        # Extract shoulder, elbow, hand
        shoulder = np.array([msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z])
        elbow    = np.array([msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z])
        hand     = np.array([msg.poses[2].position.x, msg.poses[2].position.y, msg.poses[2].position.z])
        quat     = np.array([msg.poses[2].orientation.x, msg.poses[2].orientation.y, 
                             msg.poses[2].orientation.z, msg.poses[2].orientation.w])
        
        self.latest_target_flat = np.concatenate([shoulder, elbow, hand, quat / (np.linalg.norm(quat) + 1e-8)])

    def control_loop(self):
        if self.latest_target_flat is None:
            # Heartbeat to show node is alive but waiting for data
            self.get_logger().info("Waiting for /human/skeletal_data...", throttle_duration_sec=5.0)
            return

        x0 = self.current_joints if self.first_solve else self.last_solution
        max_iters = 50 if self.first_solve else 10


        target_jax = jnp.array(self.latest_target_flat)
        weights    = jnp.array([ALPHA, BETA, GAMMA])
        
        def loss_fn(q_np):
            # Ensure we pass the full 7-DOF vector if required by ok.N_JOINTS
            full_q = jnp.zeros(ok.N_JOINTS).at[jnp.arange(N_JOINTS)].set(jnp.array(q_np))
            val, grad = ok.loss_and_grad(full_q, target_jax, weights)
            return float(val), np.array(grad)[:N_JOINTS].astype(np.float64)

        res = minimize(
            fun=loss_fn,
            x0=x0,
            method='SLSQP',
            jac=True,
            bounds=ARM_BOUNDS,
            options={'maxiter': max_iters, 'ftol': 1e-3}
        )

        if res.success:
            self.last_solution = res.x
            self.first_solve = False
            self._publish_trajectory(res.x)
            # Log progress so terminal doesn't look stuck
            self.get_logger().info(f"Loss: {res.fun:.4f}", throttle_duration_sec=1.0)
        else:
            self.get_logger().warn(f"Optimization failed: {res.message}")

    def _publish_trajectory(self, positions):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names  = JOINT_NAMES # Explicitly 6 arm joints

        pt = JointTrajectoryPoint()
        pt.positions     = positions.tolist()
        pt.velocities    = [0.0] * N_JOINTS
        pt.accelerations = [0.0] * N_JOINTS

        pt.time_from_start = Duration(sec=0, nanosec=150_000_000)

        msg.points = [pt]
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = OCRA2SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()