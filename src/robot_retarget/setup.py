from setuptools import find_packages, setup

package_name = 'robot_retarget'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'jax',       # Required for the optimization math
        'jaxlib',    # Required for JAX runtime
        'scipy',     # Required for the SLSQP solver
        'numpy',     # Required for data handling
        'pyroki',    # Required for robot kinematics
        'yourdfpy',  # Required for URDF parsing
        'jaxlie',    # Required for SE(3) math
    ],
    zip_safe=True,
    maintainer='focas',
    maintainer_email='aakshay1114@gmail.com',
    description='OCRA-based human-to-robot retargeting for the RX200 arm using JAX.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ocra_node = robot_retarget.ocra_node:main',
            'ocra_sim_node = robot_retarget.ocra_sim_node:main',
            'ocra2_sim_node = robot_retarget.ocra2_sim_node:main',
            'trajectory_bridge = robot_retarget.trajectory_bridge:main',
            'fake_human_pub = robot_retarget.fake_skele_pub:main',
            'robot_hardware_bridge = robot_retarget.robot_hardware_bridge:main',
            'ocra_addverb = robot_retarget.ocra_addverb:main',
        ],
    },
)