from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jazzy'

def get_data_files_from_folder(folder, dest):
    data_files = []
    for dirpath, dirnames, filenames in os.walk(folder):
        if filenames:
            install_path = os.path.join(dest, os.path.relpath(dirpath, folder))
            file_paths = [os.path.join(dirpath, f) for f in filenames]
            data_files.append((install_path, file_paths))
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'model'), glob('model/*')),
    ] + get_data_files_from_folder('libs', os.path.join('share', package_name, 'libs')),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cong',
    maintainer_email='pcong0280@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'test = jazzy.test:main',
            'cam = jazzy.camera:main',
            'process = jazzy.image_processor_node:main',
            'control = jazzy.motor_simulator_node:main',
        ],
    },
)
