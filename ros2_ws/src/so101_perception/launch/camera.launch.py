"""Overhead camera: publishes /overhead_camera/image_raw and /overhead_camera/camera_info."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("so101_perception"))
    return LaunchDescription([
        Node(
            package="usb_cam", executable="usb_cam_node_exe", name="overhead_camera",
            namespace="overhead_camera", output="screen",
            parameters=[str(share / "config" / "overhead_camera.yaml")],
        ),
    ])
