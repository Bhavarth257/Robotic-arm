"""Policy bridge. Dry run by default: `ros2 launch so101_policy_bridge bridge.launch.py dry_run:=false` to command."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("so101_policy_bridge"))
    return LaunchDescription([
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("model_path", default_value=str(share / "models" / "so101_pickplace.onnx")),
        Node(
            package="so101_policy_bridge", executable="bridge_node", output="screen",
            parameters=[
                str(share / "config" / "bridge.yaml"),
                {"dry_run": ParameterValue(LaunchConfiguration("dry_run"), value_type=bool),
                 "model_path": LaunchConfiguration("model_path")},
            ],
        ),
    ])
