"""Find the table marker's pose in base_link using the arm itself as the ruler.

For each spot: the camera measures the cube (relative to the table marker), then the arm, moved by hand,
measures the same spot (gripper tips around the cube; joint angles + URDF give the position in base_link).
A 2-D rigid fit over 4+ spots gives table_marker_pose [x, y, 0, yaw] for so101_perception/config/aruco.yaml.

Needs (ROS sourced, system Python, not a venv):
  ros2 launch so101_hardware real_view.launch.py           # read-only: limp arm, TF from the real joints
  ros2 launch so101_perception perception.launch.py
Usage: python3 scripts/calibrate_table_marker.py
"""
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

CONFIG = Path(__file__).resolve().parents[1] / "ros2_ws/src/so101_perception/config/aruco.yaml"


def pose2d(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


def fit_rigid_2d(p, q):
    """R, t minimising |R p + t - q| (Kabsch in 2-D)."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    r = pose2d(0, 0, yaw)[:2, :2]
    return yaw, qc - r @ pc


class Recorder(Node):
    def __init__(self):
        super().__init__("calibrate_table_marker")
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.poses = []
        self.create_subscription(PoseStamped, "/object_pose", self.poses.append, 10)

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def cube_xy(self):
        self.poses.clear()
        self.spin_for(1.0)
        if len(self.poses) < 5:
            return None
        return np.mean([[p.pose.position.x, p.pose.position.y] for p in self.poses], axis=0)

    def tcp_xy(self):
        self.spin_for(0.3)
        tips = []
        for link in ("Fixed_Jaw_tip", "Moving_Jaw_tip"):
            t = self.tf.lookup_transform("base_link", link, rclpy.time.Time()).transform.translation
            tips.append([t.x, t.y])
        return np.mean(tips, axis=0)


def main():
    cfg = yaml.safe_load(CONFIG.read_text())["/**"]["ros__parameters"]
    x, y, _, yaw = cfg["table_marker_pose"]
    # /object_pose is computed with the current (possibly wrong) table pose: undo it to get the cube in the
    # table marker's own frame, which depends only on the camera.
    table_from_base = np.linalg.inv(pose2d(x, y, yaw))

    rclpy.init()
    node = Recorder()
    node.spin_for(2.0)
    table_pts, base_pts = [], []
    print("Spread the spots over the area the arm reaches (at least 4; 6 is better).")
    while True:
        cmd = input(f"\nSpot {len(table_pts) + 1}: put the cube down (top marker up, arm out of the camera's "
                    "way), then ENTER (q + ENTER to finish): ").strip()
        if cmd == "q":
            break
        cube = node.cube_xy()
        if cube is None:
            print("  camera does not see the cube (or the table marker): move it and retry")
            continue
        cmd = input("  now move the gripper by hand: pointing straight down, jaw tips on both sides of the cube, "
                    "touching it, without pushing it. ENTER (q + ENTER drops this spot and finishes): ").strip()
        if cmd == "q":
            break
        tcp = node.tcp_xy()
        table_pts.append((table_from_base @ [*cube, 1])[:2])
        base_pts.append(tcp)
        print(f"  cube (camera, old config) {cube.round(3)}  gripper {tcp.round(3)}")

    if len(table_pts) < 4:
        print("need at least 4 spots")
        return
    p, q = np.array(table_pts), np.array(base_pts)
    # Spots bunched together leave the yaw undetermined (first run: 5 spots within 5 cm, RMS 58 mm).
    spread = max(np.linalg.norm(a - b) for a in q for b in q)
    if spread < 0.15:
        print(f"spots span only {spread * 100:.0f} cm: spread them over 15+ cm and run again")
        return
    new_yaw, t = fit_rigid_2d(p, q)
    res = (pose2d(0, 0, new_yaw)[:2, :2] @ p.T).T + t - q
    err = np.linalg.norm(res, axis=1) * 1000
    print("\nper-spot error (mm):", err.round(1).tolist(), f" RMS {math.sqrt(np.mean(err ** 2)):.1f}")
    print(f"old table_marker_pose: [{x:.4f}, {y:.4f}, 0.0, {yaw:.4f}]")
    print(f"new table_marker_pose: [{t[0]:.4f}, {t[1]:.4f}, 0.0, {new_yaw:.4f}]")
    print(f"(put the new line in {CONFIG.relative_to(CONFIG.parents[4])})")


if __name__ == "__main__":
    main()
