from setuptools import setup
import os
from glob import glob

package_name = 'bev_visualization'

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
    description='BEV visualization',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'bev_visualization_node = bev_visualization.bev_visualization_node:main',
        ],
    },
)
