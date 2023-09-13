#!/usr/bin/env python


"""
Example to launch a sensor_delta listener node.

.. seealso::
    https://index.ros.org/doc/ros2/Launch-system/
"""

import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression

configurable_parameters = [{'name': 'node_name',                 'default': 'msckf_node',   'description': 'yaml config file'},
                           {'name': 'config_file',               'default': "''",   'description': 'yaml config file'},
                           {'name': 'log_level',                 'default': 'INFO', 'description': 'debug log level [DEBUG|INFO|WARN|ERROR|FATAL]'},
                          ]

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def generate_launch_description():
    para_dir = os.path.join(get_package_share_directory('msckf_node'), 'config', 'msckf_param_node.yaml')
    return LaunchDescription(declare_configurable_parameters(configurable_parameters) + [
        launch_ros.actions.Node(
            package='msckf_node',
            name=LaunchConfiguration("node_name"),
            executable='msckf_node',
            prefix=['stdbuf -o L'],
            parameters=[para_dir],
            output='screen',
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            ),
        ])
    

