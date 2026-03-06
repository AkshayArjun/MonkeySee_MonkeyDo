import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration

TRAJ_HORIZON_NS = 300_000_000   # 300 ms — target point offset from now


class TrajectoryBridge(Node):
    def __init__(self):
        super().__init__('trajectory_bridge')

        # ── Action client ─────────────────────────────────────────────────────
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        self._active_goal: ClientGoalHandle | None = None
        self._send_in_progress = False

        # ── Topic subscriber ──────────────────────────────────────────────────
        self.create_subscription(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            self._on_trajectory,
            10
        )

        self.get_logger().info(
            'Trajectory bridge started.\n'
            '  Listening : /arm_controller/joint_trajectory (topic)\n'
            '  Forwarding: /arm_controller/follow_joint_trajectory (action)\n'
            '  Waiting for action server...'
        )

        # Non-blocking server wait — log once when it comes up
        self._server_check_timer = self.create_timer(1.0, self._check_server)

    # ── Server readiness check ────────────────────────────────────────────────
    def _check_server(self):
        if self._action_client.server_is_ready():
            self.get_logger().info('Action server is ready. Bridge active.')
            self._server_check_timer.cancel()

    # ── Incoming trajectory callback ──────────────────────────────────────────
    def _on_trajectory(self, msg: JointTrajectory):
        if not self._action_client.server_is_ready():
            self.get_logger().warn(
                'Action server not ready, dropping trajectory.',
                throttle_duration_sec=2.0
            )
            return

        if self._send_in_progress:
            # Goal send round-trip hasn't completed yet — skip this message.
            # At 10 Hz this recovers within one tick.
            return

        # ── Rebuild as two-point trajectory ───────────────────────────────────
        # The incoming message is typically a single point at t=150ms which
        # races against processing latency. We rewrite it as:
        #   pt0: current commanded position at t=0   (immediate anchor)
        #   pt1: the OCRA target at t=TRAJ_HORIZON   (smooth execution window)
        #
        # If the message already has 2+ points we forward it unchanged — the
        # sender already handled timing correctly.
        if len(msg.points) >= 2:
            traj = msg
        else:
            traj             = JointTrajectory()
            traj.joint_names = msg.joint_names

            src = msg.points[0]

            # pt0 — anchor at the position the controller is currently at.
            # We use the incoming position as pt0 because ocra2_sim_node seeds
            # it from current_joints (actual robot state), so this is correct.
            pt0               = JointTrajectoryPoint()
            pt0.positions     = src.positions
            pt0.velocities    = [0.0] * len(src.positions)
            pt0.time_from_start = Duration(sec=0, nanosec=0)

            # pt1 — the OCRA target
            pt1               = JointTrajectoryPoint()
            pt1.positions     = src.positions   # same target; pt0 is the anchor
            pt1.velocities    = [0.0] * len(src.positions)
            pt1.time_from_start = Duration(sec=0, nanosec=TRAJ_HORIZON_NS)

            traj.points = [pt0, pt1]

        # ── Cancel previous goal and send new one ─────────────────────────────
        if self._active_goal is not None:
            try:
                self._active_goal.cancel_goal_async()
            except Exception:
                pass
            self._active_goal = None

        goal             = FollowJointTrajectory.Goal()
        goal.trajectory  = traj

        self._send_in_progress = True
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    # ── Goal response ─────────────────────────────────────────────────────────
    def _on_goal_response(self, future):
        self._send_in_progress = False
        handle = future.result()
        if handle.accepted:
            self._active_goal = handle
        else:
            self.get_logger().warn(
                'Goal rejected by action server.',
                throttle_duration_sec=1.0
            )


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