"""Overhead camera + ArUco cube pose: publishes /object_pose (base_link) and TF base_link -> camera."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory("so101_perception")) / "config"
    return LaunchDescription([
        Node(package="usb_cam", executable="usb_cam_node_exe", name="overhead_camera",
             namespace="overhead_camera", output="screen",
             parameters=[str(config / "overhead_camera.yaml")]),
        Node(package="so101_perception", executable="aruco_node", name="aruco_cube",
             namespace="overhead_camera", output="screen",
             parameters=[str(config / "aruco.yaml")]),
    ])
