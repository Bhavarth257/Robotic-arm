"""Real SO-101 with MoveIt. Read-only unless read_only:=false (then TORQUE ON: the arm moves).

ros2 launch so101_hardware real_moveit.launch.py                    # limp arm, MoveIt plans, nothing moves
ros2 launch so101_hardware real_moveit.launch.py read_only:=false   # powered
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import (generate_move_group_launch, generate_moveit_rviz_launch,
                                           generate_rsp_launch, generate_static_virtual_joint_tfs_launch)

DEFAULT_CALIB = str(Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower.json")


def launch_setup(context):
    share = Path(get_package_share_directory("so101_hardware"))
    args = {k: LaunchConfiguration(k).perform(context)
            for k in ("port", "calibration_file", "read_only", "torque_limit", "p_gain")}
    moveit_config = (MoveItConfigsBuilder("so101_new_calib", package_name="so101_moveit_config")
                     .robot_description(file_path=str(share / "urdf" / "so101_real.urdf.xacro"), mappings=args)
                     .to_moveit_configs())
    spawn = lambda name, *extra: Node(package="controller_manager", executable="spawner",
                                      arguments=[name, *extra])
    return [
        *generate_static_virtual_joint_tfs_launch(moveit_config).entities,
        *generate_rsp_launch(moveit_config).entities,
        Node(package="controller_manager", executable="ros2_control_node", output="screen",
             parameters=[moveit_config.robot_description, str(share / "config" / "controllers.yaml")]),
        spawn("joint_state_broadcaster"),
        spawn("arm_controller"),
        spawn("gripper_controller"),
        spawn("forward_position_controller", "--inactive"),
        *generate_move_group_launch(moveit_config).entities,
        *(generate_moveit_rviz_launch(moveit_config).entities
          if LaunchConfiguration("rviz").perform(context) == "true" else []),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("calibration_file", default_value=DEFAULT_CALIB),
        DeclareLaunchArgument("read_only", default_value="true"),
        DeclareLaunchArgument("torque_limit", default_value="500"),
        DeclareLaunchArgument("p_gain", default_value="32"),
        DeclareLaunchArgument("rviz", default_value="true"),
        OpaqueFunction(function=launch_setup),
    ])
