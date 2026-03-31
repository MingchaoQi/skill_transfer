from setuptools import setup
import os

package_name = 'gel_boundary'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # include nnmini.pt file
        (os.path.join('share', package_name), ['nnmini.pt']),
    ],
    install_requires=['setuptools'],
    package_dir={'':'.'},
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tac3d_r = gel_boundary.tac3d_r:main',
            'listener = gel_boundary.subscriber_member_function:main',
            'initial = gel_boundary.initial:main',
            'listener_node = gel_boundary.listener:main',
            'orientation_publisher = gel_boundary.orientation_publisher:main',
            'pose_planning_node = gel_boundary.pose_planning_node:main'
            ],
    },
)

