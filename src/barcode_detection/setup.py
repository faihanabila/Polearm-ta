from setuptools import setup
import os
from glob import glob

package_name = 'barcode_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='faiha',
    maintainer_email='faiha@todo.todo',
    description='Barcode detection with webcam and Raspberry Pi Camera using pyzbar, OpenCV, and ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webcam_node = barcode_detection.webcam_node:main',
            'two_webcam_node = barcode_detection.two_webcam_node:main',
            'raspicam_node = barcode_detection.raspicam_node:main',  # Node untuk PiCam
        ],
    },
)

