# This launch file runs the joy node, drive node, and the camera node
# This node allows you to drive the robot with the XBOX One Controller and also view images from the ZED2 camera
# It can be run with this command 'ros2 launch data_collection_launch' if this file is located in the launch folder in your colcon workspace

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='joy',
            namespace='',
            executable='joy_node',
            name='sim'
        ),
        Node(
            package='final',
            namespace='',
            executable='drive',
            name='sim',
        ),
        Node(
            package='usb_cam',
            namespace ='',
            executable='usb_cam_node_exe',
            arguments=['--ros-args', '--params-file', '/home/husarion/.ros/camera_info/default_cam.yaml']
            #parameters=[{'params-file': '/home/husarion/.ros/camera_info/default_cam.yaml'}]
        )
    ])