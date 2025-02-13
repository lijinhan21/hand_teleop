import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

import init_path
from algos.single_hand_detector import SingleHandDetector
from algos.oy_retargeter import OYRetarget

from utils.realrobot_utils import real_angle_to_cmd, real_radian_to_cmd, cmd_to_real_angle, cmd_to_real_radian

import mujoco
import mujoco.viewer
from utils.video_utils import VideoWriter
from utils.task_utils import Task

class Visualizer:
    def __init__(self, save_image=False):
        
        # only put a hand in the scene
        task = Task(robot_xml='./assets/dex-hand/oy.xml', arena_xml='./assets/table_arena.xml', object_xmls=[], object_names=[])

        # initialize the dex-hand base pose
        task.set_base_xpos([0.29, 0, 1.15])
        task.set_base_ori([1.57, 0, 0])
        print("base xpos=", task.get_base_xpos())

        print("new xml is saved to: new.xml")
        os.makedirs("./tmp_files", exist_ok=True)
        task.save_model("./tmp_files/new.xml")

        self.m = mujoco.MjModel.from_xml_path("./tmp_files/new.xml")
        self.d = mujoco.MjData(self.m)

        print('default gravity', self.m.opt.gravity)

        num_joints = self.m.nu
        print(f"Actuators (nu): {self.m.nu}")  # 执行器数量
        print(f"Position DOF (nq): {self.m.nq}")  # 位置自由度
        print(f"Velocity DOF (nv): {self.m.nv}")  # 速度自由度
        print(f"Number of bodies: {self.m.nbody}")  # 身体数量

        # 打印所有joint的名称和qpos
        for i in range(self.m.nq):
            joint_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"Joint {i}: {joint_name}, qpos value: {self.d.qpos[i]}")
        # 打印所有body的名称和位置
        for i in range(self.m.nbody):
            body_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"Body {i}: {body_name}, position: {self.d.xpos[i]}")

        # 查看场景中所有相机的名称
        for i in range(self.m.ncam):
            camera_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"Camera {i}: {camera_name}")

        os.makedirs("./tmp_videos", exist_ok=True)
        self.video = VideoWriter(video_path="./tmp_videos/", save_video=True, fps=30, single_video=True)
        
        self.save_image = save_image
    
    def step(self, thetas, num=0):
        with mujoco.Renderer(self.m, 480, 640) as renderer:
            
            self.d.qpos[3:14] = thetas
            
            # mujoco.mj_step(self.m, self.d)
            mujoco.mj_forward(self.m, self.d)
            
            renderer.update_scene(self.d, camera='agentview') 
            image = renderer.render()
            self.video.append_image(image)
            
            if self.save_image:
                cv2.imwrite(f"tmp_images/right{num}_retarget.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
        return image
        
    def save(self, video_name='test-dex-retarget.mp4'):
        print("simulation finished, begin saving video")     
        self.video.save(video_name)
        print(f"video saved at tmp_videos/{video_name}")

if __name__ == '__main__':

    # 创建管道对象
    pipeline = rs.pipeline()

    # 配置流
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 启用彩色流

    # 开始流
    pipeline.start(config)

    # hand detector
    detector = SingleHandDetector(hand_type='Right', selfie=False)

    # dex retarget
    retargeter = OYRetarget()

    # simulation
    vis = Visualizer()

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

                # send to sim
                img = vis.step(theta)
                
                cv2.imshow("sim", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            

    finally:

        # 停止管道

        pipeline.stop()
        cv2.destroyAllWindows()