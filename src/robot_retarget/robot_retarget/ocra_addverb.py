"""
ocra_addverb.py

High-level retargeting node for the Addverb Heal cobot.
Contains all logic — bridge is a dumb converter.

Current: gripper only.
  /human/gripper_cmd (Bool) → converts to GripperCommand → /gripper/cmd

Later: arm retargeting will be added here.
  /human/skeletal_data (PoseArray) → OCRA IK → /arm_controller/joint_trajectory_raw
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from control_msgs.msg import GripperCommand

GRIPPER_OPEN   = 0.0     # metres — tune to your Robotiq 85
GRIPPER_CLOSED = 1.0   # metres
GRIPPER_EFFORT = 50.0    # Newtons


class OcraAddverbNode(Node):
    def __init__(self):
        super().__init__('ocra_addverb')

        self.create_subscription(
            Bool,
            '/human/gripper_cmd',
            self._on_gripper_cmd,
            10
        )

        self._gripper_pub = self.create_publisher(
            GripperCommand,
            '/gripper/cmd',
            10
        )

        self._last_state: bool | None = None

        self.get_logger().info(
            'ocra_addverb ready.\n'
            '  /human/gripper_cmd (Bool) → /gripper/cmd (GripperCommand)'
        )

    def _on_gripper_cmd(self, msg: Bool):
        # Only act on state changes — don't spam the bridge
        if msg.data == self._last_state:
            return
        self._last_state = msg.data

        cmd            = GripperCommand()
        cmd.position   = GRIPPER_OPEN if msg.data else GRIPPER_CLOSED
        cmd.max_effort = GRIPPER_EFFORT

        self._gripper_pub.publish(cmd)
        self.get_logger().info(
            f'Gripper → {"OPEN" if msg.data else "CLOSE"} ({cmd.position:.3f} m)')


def main(args=None):
    rclpy.init(args=args)
    node = OcraAddverbNode()
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