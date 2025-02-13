import pyrealsense2 as rs
import numpy as np
import cv2
import time

import init_path
from algos.single_hand_detector import SingleHandDetector
from algos.oy_retargeter import OYRetarget

from pymodbus import FramerType
from pymodbus.client import ModbusSerialClient
from protocol.roh_registers_v1 import *

from utils.realrobot_utils import real_angle_to_cmd, real_radian_to_cmd, cmd_to_real_angle, cmd_to_real_radian

# robot hand control
COM_PORT = '/dev/ttyUSB0'
NODE_ID = 2

upper_limits = [2.26, 100.21, 97.81, 101.37, 98.88, 89.98] # 握拳
lower_limits = [36.76, 178.33, 176.0, 176.49, 174.87, 0.01] # 张开

def get_actuate_joints(x):
    indexes = [9, 0, 4, 6, 2, 8]
    radian = x[indexes]
    radian[1:-1] += np.pi
    
    limit1_rad = np.array(upper_limits) * np.pi / 180
    limit2_rad = np.array(lower_limits) * np.pi / 180
    limit_min = np.minimum(limit1_rad, limit2_rad)
    limit_max = np.maximum(limit1_rad, limit2_rad)

    radian = np.clip(radian, limit_min, limit_max)
    return radian

if __name__ == '__main__':
    
    # start camera
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 启用彩色流

    pipeline.start(config)
    
    # real robot controller
    client = ModbusSerialClient(
        port=COM_PORT,
        baudrate=115200,
        framer='rtu'    # 使用 framer 替代 method
    )
    client.connect()
    
    # hand detector
    detector = SingleHandDetector(hand_type='Right', selfie=False)

    # dex retarget
    retargeter = OYRetarget()

    try:
        while True:

            # 等待一帧数据
            frames = pipeline.wait_for_frames()

            # 获取彩色帧
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # 将帧数据转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())

            bgr = color_image.copy()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)

            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            cv2.imshow("realtime_retargeting_demo", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            if joint_pos is not None:
                theta = retargeter.retarget(joint_pos)
                print("theta=", theta)

                targets = get_actuate_joints(theta)
                cmd = real_radian_to_cmd(targets)
                print(cmd[:])
                
                # write
                # resp = client.write_registers(ROH_FINGER_ANGLE_TARGET1, cmd[1:-1].tolist(), slave=NODE_ID) # index, middle, ring, pinky
                # resp = client.write_registers(ROH_FINGER_ANGLE_TARGET0, [cmd[0]], slave=NODE_ID) # thumb bend
                # resp = client.write_registers(ROH_FINGER_ANGLE_TARGET5, [cmd[-1]], slave=NODE_ID) # thumb rotate
                resp = client.write_registers(ROH_FINGER_ANGLE_TARGET0, cmd.tolist(), slave=NODE_ID) # all six finger joints
                time.sleep(1/30)
            

    finally:

        # 停止管道

        pipeline.stop()
        cv2.destroyAllWindows()

        # 张开
        resp = client.write_registers(ROH_FINGER_POS_TARGET0, [0], slave=NODE_ID)
        time.sleep(2)
        resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 0, 0], slave=NODE_ID)
        time.sleep(2)