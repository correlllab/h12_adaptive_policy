import time
import argparse

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

import os
import sys
# add workspace root to path
workspace_root = os.path.abspath(os.path.join(__file__, '../../..'))
sys.path.append(workspace_root)
# add submodule to path so h12_ros2_controller package is importable
sys.path.append(os.path.join(workspace_root, 'submodules/h12_ros2_controller'))
from h12_ros2_controller.core.robot_model import RobotModel

def main():
    ChannelFactoryInitialize()
    robot_model = RobotModel('./submodules/h12_ros2_controller/assets/h1_2/h1_2_handless.urdf')
    robot_model.init_visualizer()
    robot_model.config_visualizer(show_sensors=True, show_com=True, show_zmp=True)
    robot_model.init_subscriber()

    # main loop shadowing robot states
    while True:
        robot_model.update_kinematics()
        robot_model.update_visualizer()

        # estimate force
        link_name = 'left_wrist_yaw_link'
        wrench = robot_model.get_frame_wrench(link_name)
        print(f'Force at {link_name}: {wrench[0:3]}')

        time.sleep(0.01)

if __name__ == '__main__':
    main()
