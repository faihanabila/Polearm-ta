from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='barcode_detection',
            executable='two_webcam_node',
            name='webcam_barcode_node',
            output='screen',
            parameters=[
                {'camera1_device': '/dev/video2'},  # C270
                {'camera2_device': '/dev/video4'},  # BRIO RGB
                {'use_v4l2': True},
                {'show_camera': True},
                {'results_dir': '/home/faiha/Polearm/src/barcode_detection/results'}
            ]
        )
    ])
