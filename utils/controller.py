import time
import mujoco
import mujoco.viewer
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R

import init_path
from utils.video_utils import VideoWriter
from utils.task_utils import Task

def make_model_singlehand(object_xmls, object_names, lr='right', init_direction='up'):

    robot_xml = f'./assets/oyhands/oymotion_{lr}.xml'
    task = Task(robot_xml=robot_xml, arena_xml='./assets/table_arena.xml', object_xmls=object_xmls, object_names=object_names)

    # initialize the dex-hand base pose
    task.set_base_xpos([-0.2, 0, 0.85])
    if lr == 'right':
        if init_direction == 'up':
            task.set_base_ori([3.14, 0, 0])
        else:
            task.set_base_ori([1.57, 0, 0])
    else:
        if init_direction == 'up':
            task.set_base_ori([3.14, 0, 3.14])
        else:
            task.set_base_ori([-1.57, 0, 3.14])

    print("base xpos=", task.get_base_xpos())

    # initialize the object pose
    task.set_object_xpos([-0.05, 0.1, 0.8], object_names[-1])

    # print("new xml is saved to: new.xml")
    os.makedirs("./tmp_files", exist_ok=True)
    task.save_model("./tmp_files/new.xml")

    m = mujoco.MjModel.from_xml_path("./tmp_files/new.xml")
    d = mujoco.MjData(m)
    
    return m, d

def interpolate_t(current, target, t):
    return (np.array(current) + (np.array(target) - np.array(current)) * t).tolist()

def get_cmd(init_ctrl, targets, step, int_steps=10):
    if step == 0:
        return init_ctrl
    if step >= len(targets) * int_steps:
        return targets[-1]
    
    target_idx = step // int_steps
    t = (step % int_steps + 1) / int_steps
    last_target = targets[target_idx - 1] if target_idx > 0 else init_ctrl
    return interpolate_t(last_target, targets[target_idx], t)

def quat_wxyz_to_xyzw(quat_wxyz):
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    if isinstance(quat_wxyz, list):
        return quat_xyzw
    return np.array(quat_xyzw)

class SingleHandController:
    def __init__(self, m, d, lr='right', init_direction='up'):
        self.m = m
        self.d = d
        self.num_actuators = m.nu
        
        self.lr = lr
        self.hand_body_name = f"oymotion_{'r' if lr == 'right' else 'l'}_hand"
        self.hand_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.hand_body_name)
        print(f"hand body id: {self.hand_body_id}")
        
        mujoco.mj_forward(self.m, self.d)
        self.init_pos = self.d.xpos[self.hand_body_id].copy()
        self.init_quat = self.d.xquat[self.hand_body_id].copy()
        
        self.init_T = np.eye(4)
        self.init_T[:3, :3] = R.from_quat(
            quat_wxyz_to_xyzw(np.array(self.init_quat.copy()))
        ).as_matrix()
        self.init_T[:3, 3] = self.init_pos
        
        self.init_ctrl = self.get_joint_pos()
        
        print("after initialization, pos=", self.init_pos, "quat=", self.init_quat)
        
        # if self.lr == 'right':
        #     if init_direction == 'up':
        #         self.R_world_hand = np.array([
        #             [1, 0, 0],
        #             [0, -1, 0],
        #             [0, 0, -1]
        #         ])
        #     else:
        #         self.R_world_hand = np.array([
        #             [1, 0, 0],
        #             [0, 0, -1],
        #             [0, 1, 0]
        #         ])
        # else:
        #     if init_direction == 'up':
        #         self.R_world_hand = np.array([
        #             [-1, 0, 0],
        #             [0, 1, 0],
        #             [0, 0, -1]
        #         ])
        #     else:
        #         self.R_world_hand = np.array([
        #             [-1, 0, 0],
        #             [0, 0, 1],
        #             [0, 1, 0]
        #         ])
        
    def set_ctrl(self, ctrl):
        self.d.ctrl = ctrl
        
    def get_joint_pos(self):
        return [self.d.qpos[self.m.actuator_trnid[i, 0]] for i in range(self.num_actuators)]
    
    def get_pose_world(self):
        pos = self.d.xpos[self.hand_body_id].copy()
        quat = self.d.xquat[self.hand_body_id].copy()
        
    def cal_pose_cmd(self, pos, rot):
        """
        Given the desired position and orientation of the hand in the world frame, calculate the control command

        Args:
            pos (np.array(3,)): desired position of the hand in the world frame
            rot (np.array(3, 3)): desired rotation matrix of the hand in the world frame.
            
        Returns:
            np.array(6,): control command for the base position and orientation joints. The first 3 elements are for position, the last 3 elements are for orientation (roll, pitch, yaw).
        """
        pos = np.array(pos)
        rot = np.array(rot)
        
        target_T = np.eye(4)
        target_T[:3, :3] = rot
        target_T[:3, 3] = pos
        
        T_hand_target = np.linalg.inv(self.init_T) @ target_T
        cmd_pos = T_hand_target[:3, 3]
        cmd_euler = R.from_matrix(T_hand_target[:3, :3]).as_euler('xyz', degrees=False)
        
        return np.concatenate([cmd_pos, cmd_euler])
    
    def cal_full_cmd(self, pos, rot, finger_cmd):
        pos = np.array(pos)
        rot = np.array(rot)
        finger_cmd = np.array(finger_cmd)
        return np.concatenate([finger_cmd, self.cal_pose_cmd(pos, rot)]).tolist()

class CmdDrawer:
    def __init__(self, names):
        self.cmds = []
        self.reaches = []
        self.names = names
        
    def add(self, cmd, reach):
        self.cmds.append(cmd)
        self.reaches.append(reach)
        
    def draw(self, save_path):
        
        self.cmds = np.array(self.cmds)
        self.reaches = np.array(self.reaches)
        
        # 计算需要的行列数
        n = len(self.names)
        cols = min(3, n)  # 每行最多3张图
        rows = (n + cols - 1) // cols  # 向上取整计算行数
        
        # 创建子图
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n == 1:
            axes = np.array([axes])  # 确保axes是数组
        axes = axes.flatten()  # 将多维数组展平
        
        # 设置图表整体样式
        fig.suptitle('Command Following Graph', fontsize=16)
        
        # 绘制每个子图
        for i in range(n):
            ax = axes[i]
            time = np.arange(self.cmds.shape[0])
            
            # 绘制两条线
            ax.plot(time, self.cmds[:, i], 'b-', label='CMD', linewidth=2)
            ax.plot(time, self.reaches[:, i], 'r-', label='Reach', linewidth=2)
            
            # 设置图表样式
            ax.set_title(self.names[i])
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()
        
        # 移除多余的子图
        for i in range(n, len(axes)):
            fig.delaxes(axes[i])
            
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.cmds = self.cmds.tolist()
        self.reaches = self.reaches.tolist()
    
if __name__ == '__main__':

    lr = 'right'
    init_direction = 'side'
    m, d = make_model_singlehand(
        object_xmls=['./assets/objects/can.xml', './assets/ycb_objects/xml/mug.xml'], 
        object_names=['can', 'mug'], 
        lr=lr,
        init_direction=init_direction,
    )
    
    controller = SingleHandController(m, d, lr=lr, init_direction=init_direction)
    
    # with mujoco.Renderer(m) as renderer:
    #     mujoco.mj_forward(m, d)
    #     renderer.update_scene(d, camera='robotview')
        
    #     cv2.imwrite('tmp_images/test_orient.png', cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    
    # exit(0)
    
    init_ctrl = controller.init_ctrl
    targets = [
        controller.cal_full_cmd(
            pos=[-0.2, 0.1, 0.85],
            rot=np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ]),
            finger_cmd=[0., 3.02, 3.0, 3.0, 3.0, 0.]
        )
    ]

    os.makedirs("./tmp_images", exist_ok=True)
    drawer = CmdDrawer(
        names=[f"Joint {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])}" for i in range(m.nu)]
    )
    
    os.makedirs("./tmp_videos", exist_ok=True)
    video = VideoWriter(video_path="./tmp_videos/", save_video=True, fps=30, single_video=True)
    with mujoco.Renderer(m, 480, 640) as renderer:
        start_time = time.time()
        print("starting rendering!")
        step = 0
        while time.time() - start_time < 10:
            
            d.ctrl = get_cmd(init_ctrl, targets, step, int_steps=50)
            step += 1
            
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            
            renderer.update_scene(d, camera='robotview') 
            image = renderer.render()
            video.append_image(image)
            
            print("step=", step)
            for j in range(m.nu):
                joint_id = m.actuator_trnid[j, 0]
                joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                print(f"Joint {joint_name}: {d.qpos[joint_id]}, control={d.ctrl[j]}")
            print()
            
            cmd = d.ctrl.copy()
            reach = [d.qpos[m.actuator_trnid[i, 0]] for i in range(m.nu)]
            # print("cmd=", cmd, "reach=", reach)
            drawer.add(cmd, reach)
            
            time.sleep(0.03)
    
    print("simulation finished, begin saving video")     
    video.save(f"test-control-{lr}.mp4")
    print("video saved")
    
    drawer.draw(f"./tmp_images/test-control-{lr}.png")
    print("image saved")