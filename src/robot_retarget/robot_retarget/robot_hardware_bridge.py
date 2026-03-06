"""
robot_hardware_bridge.py

Bridges simulation-side topics to the real Addverb Heal robot's action servers.

  /arm_controller/joint_trajectory_raw  (JointTrajectory topic, from ocra2_sim_node)
      → /joint_impedance_controller/follow_joint_trajectory  (FollowJointTrajectory action)

  /gripper/cmd  (GripperCommand topic)
      → /gripper_controller/gripper_cmd  (GripperCommand action)

Run this instead of trajectory_bridge.py when controlling the real robot.
The sim launch and ocra2_sim_node stay completely unchanged.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory, GripperCommand
from control_msgs.msg import GripperCommand as GripperCommandMsg


class RobotHardwareBridge(Node):
    def __init__(self):
        super().__init__('robot_hardware_bridge')

        # ── Arm: topic → FollowJointTrajectory action ─────────────────────────
        self._arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_impedance_controller/follow_joint_trajectory'
        )
        self._arm_goal: ClientGoalHandle | None = None
        self._arm_send_in_progress = False

        self.create_subscription(
            JointTrajectory,
            '/arm_controller/joint_trajectory_raw',
            self._on_arm_trajectory,
            10
        )

        # ── Gripper: topic → GripperCommand action ────────────────────────────
        self._gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )
        self._gripper_goal: ClientGoalHandle | None = None
        self._gripper_send_in_progress = False

        self.create_subscription(
            GripperCommandMsg,
            '/gripper/cmd',
            self._on_gripper_cmd,
            10
        )

        # ── Server readiness check ────────────────────────────────────────────
        self._check_timer = self.create_timer(1.0, self._check_servers)
        self._arm_ready     = False
        self._gripper_ready = False

        self.get_logger().info(
            'Robot hardware bridge started.\n'
            '  Arm    : /arm_controller/joint_trajectory_raw'
            ' → /joint_impedance_controller/follow_joint_trajectory\n'
            '  Gripper: /gripper/cmd'
            ' → /gripper_controller/gripper_cmd'
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

    # ── Arm ───────────────────────────────────────────────────────────────────
    def _on_arm_trajectory(self, msg: JointTrajectory):
        if not self._arm_ready:
            self.get_logger().warn(
                'Arm action server not ready, dropping.',
                throttle_duration_sec=2.0)
            return
        if self._arm_send_in_progress:
            return

        if self._arm_goal is not None:
            try:
                self._arm_goal.cancel_goal_async()
            except Exception:
                pass
            self._arm_goal = None

        goal            = FollowJointTrajectory.Goal()
        goal.trajectory = msg

        self._arm_send_in_progress = True
        future = self._arm_client.send_goal_async(goal)
        future.add_done_callback(self._on_arm_response)

    def _on_arm_response(self, future):
        self._arm_send_in_progress = False
        handle = future.result()
        if handle.accepted:
            self._arm_goal = handle
        else:
            self.get_logger().warn('Arm goal rejected.', throttle_duration_sec=1.0)

    # ── Gripper ───────────────────────────────────────────────────────────────
    def _on_gripper_cmd(self, msg: GripperCommandMsg):
        if not self._gripper_ready:
            self.get_logger().warn(
                'Gripper action server not ready, dropping.',
                throttle_duration_sec=2.0)
            return
        if self._gripper_send_in_progress:
            return

        if self._gripper_goal is not None:
            try:
                self._gripper_goal.cancel_goal_async()
            except Exception:
                pass
            self._gripper_goal = None

        goal         = GripperCommand.Goal()
        goal.command = msg

        self._gripper_send_in_progress = True
        future = self._gripper_client.send_goal_async(goal)
        future.add_done_callback(self._on_gripper_response)

    def _on_gripper_response(self, future):
        self._gripper_send_in_progress = False
        handle = future.result()
        if handle.accepted:
            self._gripper_goal = handle
        else:
            self.get_logger().warn('Gripper goal rejected.', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = RobotHardwareBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()