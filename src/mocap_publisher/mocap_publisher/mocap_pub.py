"""
mocap_pub.py — ROS2 Publisher Node

Listens on UDP port 5005 for JSON data from camera_tracker.py and publishes:
  /mocap/joints           → geometry_msgs/PoseArray   (shoulder, elbow, wrist)
  /mocap/hand_orientation → geometry_msgs/QuaternionStamped
  /mocap/state            → std_msgs/String  ("TRACKING" or "CALIBRATION")

How to run:
  ros2 run mocap_publisher mocap_pub_node

Ensure camera_tracker.py is also running to send UDP data.
"""

import json
import socket

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose, QuaternionStamped, Quaternion, Point
from std_msgs.msg import String


UDP_IP   = "0.0.0.0"   # Listen on all interfaces
UDP_PORT = 5005


class MocapPublisher(Node):
    def __init__(self):
        super().__init__('mocap_publisher')

        # ── Publishers ──────────────────────────────────────────────────────
        # PoseArray with 3 entries: [shoulder, elbow, wrist]
        self.pub_joints = self.create_publisher(
            PoseArray, '/mocap/joints', 10
        )
        # Hand quaternion
        self.pub_hand = self.create_publisher(
            QuaternionStamped, '/mocap/hand_orientation', 10
        )
        # Tracker state ("TRACKING" / "CALIBRATION")
        self.pub_state = self.create_publisher(
            String, '/mocap/state', 10
        )

        # ── UDP socket (non-blocking) ────────────────────────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)
        self.get_logger().info(
            f'Mocap Publisher listening on UDP {UDP_IP}:{UDP_PORT} ...'
        )

        # ── Timer: poll UDP at 60 Hz, publish when data arrives ─────────────
        self.create_timer(1.0 / 60.0, self.timer_callback)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _make_pose(self, xyz):
        """Build a geometry_msgs/Pose from a [x, y, z] list (no rotation)."""
        pose = Pose()
        pose.position.x = float(xyz[0])
        pose.position.y = float(xyz[1])
        pose.position.z = float(xyz[2])
        # Orientation identity (rotation not meaningful for a point landmark)
        pose.orientation.w = 1.0
        return pose

    def _make_quaternion(self, quat):
        """Build a geometry_msgs/Quaternion from [qx, qy, qz, qw]."""
        q = Quaternion()
        q.x = float(quat[0])
        q.y = float(quat[1])
        q.z = float(quat[2])
        q.w = float(quat[3])
        return q

    # ── main callback ────────────────────────────────────────────────────────

    def timer_callback(self):
        """Drain the UDP buffer and publish the latest packet."""
        payload = None

        # Read all pending datagrams — keep only the freshest one
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
                payload = json.loads(data.decode())
            except BlockingIOError:
                break           # no more data right now
            except json.JSONDecodeError as e:
                self.get_logger().warn(f'JSON decode error: {e}')
                break

        if payload is None:
            return  # nothing received this tick

        now = self.get_clock().now().to_msg()
        state_str = payload.get('state', 'UNKNOWN')

        # ── Publish state ────────────────────────────────────────────────────
        state_msg = String()
        state_msg.data = state_str
        self.pub_state.publish(state_msg)

        if state_str != 'TRACKING':
            return  # skip joint publishing during calibration

        # ── Publish joints as PoseArray ──────────────────────────────────────
        # Order: [0] shoulder, [1] elbow, [2] wrist
        pose_array = PoseArray()
        pose_array.header.stamp    = now
        pose_array.header.frame_id = 'mocap_world'

        try:
            pose_array.poses = [
                self._make_pose(payload['h_shoulder']),
                self._make_pose(payload['h_elbow']),
                self._make_pose(payload['h_wrist']),
            ]
        except KeyError as e:
            self.get_logger().warn(f'Missing key in payload: {e}')
            return

        self.pub_joints.publish(pose_array)

        # ── Publish hand orientation ─────────────────────────────────────────
        hand_msg = QuaternionStamped()
        hand_msg.header.stamp    = now
        hand_msg.header.frame_id = 'mocap_world'
        hand_msg.quaternion = self._make_quaternion(
            payload.get('h_hand_quat', [0.0, 0.0, 0.0, 1.0])
        )
        self.pub_hand.publish(hand_msg)

        self.get_logger().debug(
            f'Published joints | shoulder={payload["h_shoulder"]} '
            f'elbow={payload["h_elbow"]} wrist={payload["h_wrist"]}'
        )

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MocapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
