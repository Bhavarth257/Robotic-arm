"""Real SO-101 in RViz, read-only: the servos stay limp and RViz follows the arm moved by hand."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DEFAULT_CALIB = str(Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower.json")


def generate_launch_description():
    share = Path(get_package_share_directory("so101_hardware"))
    desc_share = Path(get_package_share_directory("so101_description"))
    robot_description = ParameterValue(Command([
        "xacro ", str(share / "urdf" / "so101_real.urdf.xacro"),
        " port:=", LaunchConfiguration("port"),
        " calibration_file:=", LaunchConfiguration("calibration_file"),
    ]), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("calibration_file", default_value=DEFAULT_CALIB),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[{"robot_description": robot_description}, str(share / "config" / "controllers.yaml")],
             output="screen"),
        Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(desc_share / "rviz" / "view.rviz")]),
    ])
