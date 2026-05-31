from setuptools import setup
import os
from glob import glob

package_name = 'pose_anchor_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tca_bevnav',
    maintainer_email='todo@example.com',
    description='State-agnostic pose anchor',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'pose_anchor_manager_node = pose_anchor_manager.pose_anchor_manager_node:main',
        ],
    },
)
