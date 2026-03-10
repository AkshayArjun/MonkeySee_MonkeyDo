"""
trajectory_bridge.py

/arm_controller/joint_trajectory_raw  (from ocra2_sim_node)
→ /arm_controller/follow_joint_trajectory  (action server on controller)

The action server is what makes motion work reliably — it handles goal
replacement cleanly and doesn't have the stale-stamp race of the raw topic.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory


class TrajectoryBridge(Node):
    def __init__(self):
        super().__init__('trajectory_bridge')

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        self._active_goal: ClientGoalHandle | None = None
        self._send_in_progress = False

        self.create_subscription(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            self._on_trajectory,
            10
        )

        self._check_timer = self.create_timer(1.0, self._check_server)
        self.get_logger().info('Trajectory bridge started, waiting for action server...')

    def _check_server(self):
        if self._action_client.server_is_ready():
            self.get_logger().info('Action server ready. Bridge active. yes')
            self._check_timer.cancel()

    def _on_trajectory(self, msg: JointTrajectory):
        if not self._action_client.server_is_ready():
            return
        if self._send_in_progress:
            return

        # Cancel previous goal before sending new one
        if self._active_goal is not None:
            try:
                self._active_goal.cancel_goal_async()
            except Exception:
                pass
            self._active_goal = None

        goal            = FollowJointTrajectory.Goal()
        goal.trajectory = msg   # already has pt0 + pt1 from the node

        self._send_in_progress = True
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        self._send_in_progress = False
        handle = future.result()
        if handle.accepted:
            self._active_goal = handle
        else:
            self.get_logger().warn('Goal rejected.', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryBridge()
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