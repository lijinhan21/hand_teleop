import pyrealsense2 as rs
import numpy as np
import cv2

import init_path
from algos.single_hand_detector import SingleHandDetector

# 创建管道对象
pipeline = rs.pipeline()

# 配置流
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 启用彩色流

# 开始流
pipeline.start(config)
detector = SingleHandDetector(hand_type='Right', selfie=False)

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

        # import pdb; pdb.set_trace()
        # if joint_pos is not None:
        #     print(joint_pos)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 显示图像

        # cv2.imshow('RealSense Color Stream', color_image)

        

        # 按下'q'键退出

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

finally:

    # 停止管道

    pipeline.stop()

    cv2.destroyAllWindows()