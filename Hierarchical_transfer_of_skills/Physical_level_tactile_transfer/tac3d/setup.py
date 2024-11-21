from setuptools import setup
import os

package_name = 'tac3d'

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
    py_modules=['PyTac3D','image_boundary','gs3drecon'],
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
            'tac3d_r = tac3d.tac3d_r:main',
            'listener = tac3d.subscriber_member_function:main',
            'initial = tac3d.initial:main',
            'listener_node = tac3d.listener:main',
            'orientation_publisher = tac3d.orientation_publisher:main',
            'pose_planning_node = tac3d.pose_planning_node:main'
            ],
    },
)

