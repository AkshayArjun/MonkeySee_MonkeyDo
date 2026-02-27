#!/usr/bin/env python3
"""
fake_human_pub.py — Publishes fake skeletal data to test ocra_sim_node
Simulates a simple arm motion: shoulder fixed, elbow and hand moving in a arc

Run: python3 fake_human_pub.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import numpy as np


class FakeHumanPublisher(Node):
    def __init__(self):
        super().__init__('fake_human_publisher')

        self.pub = self.create_publisher(PoseArray, '/human/skeletal_data', 10)
        self.timer = self.create_timer(0.1, self.publish_pose)  # 10Hz

        self.t = 0.0
        self.get_logger().info("Fake Human Publisher started...")

    def publish_pose(self):
        self.t += 0.05  # slower oscillation

        shoulder = np.array([0.0, 0.0, 0.0])  # at robot base height

        # stay within ~0.3m reach, mostly forward
        elbow = np.array([
            0.15 + 0.05 * np.sin(self.t),   # x: 0.10 to 0.20
            0.05 * np.cos(self.t),           # y: small side motion
            0.2                              # z: fixed height
        ])

        hand = np.array([
            0.25 + 0.08 * np.sin(self.t),   # x: 0.17 to 0.33
            0.08 * np.cos(self.t),           # y: small side motion
            0.15                             # z: fixed height
        ])

        quat = np.array([0.0, 0.0, 0.0, 1.0])
        # ── Build PoseArray ──────────────────────────────────────
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'mocap_world'

        # poses[0] → shoulder
        p0 = Pose()
        p0.position.x = float(shoulder[0])
        p0.position.y = float(shoulder[1])
        p0.position.z = float(shoulder[2])
        p0.orientation.w = 1.0

        # poses[1] → elbow
        p1 = Pose()
        p1.position.x = float(elbow[0])
        p1.position.y = float(elbow[1])
        p1.position.z = float(elbow[2])
        p1.orientation.w = 1.0

        # poses[2] → hand + quaternion
        p2 = Pose()
        p2.position.x    = float(hand[0])
        p2.position.y    = float(hand[1])
        p2.position.z    = float(hand[2])
        p2.orientation.x = float(quat[0])
        p2.orientation.y = float(quat[1])
        p2.orientation.z = float(quat[2])
        p2.orientation.w = float(quat[3])

        msg.poses = [p0, p1, p2]
        self.pub.publish(msg)

        self.get_logger().info(
            f"t={self.t:.1f} | shoulder={shoulder} | elbow={elbow} | hand={hand}",
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = FakeHumanPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()