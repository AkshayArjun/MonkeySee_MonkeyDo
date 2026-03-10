from setuptools import find_packages, setup

package_name = 'mocap_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shreya',
    maintainer_email='shreyashah@example.com',
    description='ROS2 publisher for mocap joint positions and hand orientation from OAK-D + MediaPipe.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'human_pub = mocap_publisher.camera_tracker.py:main',
        ],
    },
)
