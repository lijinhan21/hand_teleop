import numpy as np
import os
from easydict import EasyDict
import yaml
import pickle
import cv2
import argparse
from tqdm import tqdm

from dex_retargeting.constants import RobotName, RetargetingType, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

import time


class DexRetargetWrapper:
    
    def __init__(self, config):

        root_dir = os.path.dirname(os.path.abspath(__file__))
        print("root_dir=", root_dir)
        urdf_dir = os.path.join(root_dir, '../assets')
        print("urdf_dir=", urdf_dir)
        
        self.config = config
        
        RetargetingConfig.set_default_urdf_dir(urdf_dir)
        dex_retarget_config = config['retargeting'].copy()
        retargeting_config = RetargetingConfig.from_dict(dex_retarget_config)
        
        self.dex_retargeter = retargeting_config.build()

        print("self.config=", self.config['retargeting'])

    def calculate_robot_vec(self, thetas):
        
        self.dex_retargeter.optimizer.robot.compute_forward_kinematics(thetas)
        target_link_poses = [self.dex_retargeter.optimizer.robot.get_link_pose(index) for index in self.dex_retargeter.optimizer.computed_link_indices]
        body_pos = np.array([pose[:3, 3] for pose in target_link_poses])
        
        origin_link_pos = body_pos[self.dex_retargeter.optimizer.origin_link_indices, :]
        task_link_pos = body_pos[self.dex_retargeter.optimizer.task_link_indices, :]
        robot_vec = task_link_pos - origin_link_pos # all the retargeted robot vectors (all that used for optimization)
        
        link_names_21 = ['base_link', 
                         'base_link', 'th_link_1', 'th_link_3', 'thumb_tip', 'ff_proximal_link', 'ff_distal_link', 'ff_distal_link', 'index_tip', 'mf_proximal_link', 'mf_distal_link', 'mf_distal_link', 'middle_tip', 
                         'rf_proximal_link', 'rf_distal_link', 'rf_distal_link', 'ring_tip', 
                         'lf_proximal_link', 'lf_distal_link', 'lf_distal_link', 'pinky_tip']
        link_poses_21 = [self.dex_retargeter.optimizer.robot.get_link_pose(self.dex_retargeter.optimizer.robot.get_link_index(name)) for name in link_names_21]
        retarget_joint_pos = np.array([pose[:3, 3] for pose in link_poses_21]) # all the retargeted joint positions (in the order of 21 hand keypoints)
        
        return robot_vec, origin_link_pos, task_link_pos, retarget_joint_pos
    
    # def retarget_score(self, mano, thetas):
    #     '''
    #     calculate the score of the retargeting result.
    #     '''
    #     indices = self.dex_retargeter.optimizer.target_link_human_indices
    #     targets = mano['normalized_mano_joints'][1]
    #     joint_pos = self.target_coordinate_transformations(targets.copy())
        
    #     ref_value = joint_pos[indices[1, :], :] - joint_pos[indices[0, :], :]
    #     retarget_robot_vec = self.calculate_robot_vec(thetas)[0]
    #     score_vector = np.linalg.norm(ref_value - retarget_robot_vec)
        
    #     # print("mano=", mano)
    #     thumb_rot = mano['pred_mano_params'][1]['hand_pose'][12] # (3, 3)
    #     thumb_rot = np.arctan(thumb_rot[2, 1] / thumb_rot[1, 1])
    #     retarget_thumb_rot = thetas[-3]
    #     # print("thumb_rot=", thumb_rot, "retarget_thumb_rot=", retarget_thumb_rot, "in degree=", thumb_rot * 180 / np.pi, retarget_thumb_rot * 180 / np.pi)
    #     score_rot = np.abs(thumb_rot - retarget_thumb_rot)
        
    #     score = score_vector + score_rot
    #     # print("score=", score_vector, score_rot, score)
    #     return score

    def coordinate_transformation(self, detected_kpts):
        '''
        Transform keypoints into the dex-hand coordinate.
        '''
        
        targets = detected_kpts.copy()       
        ori_target0 = targets[:, 0].copy()
        
        targets[:, 0] = -targets[:, 2]
        targets[:, 2] = -targets[:, 1]
        targets[:, 1] = ori_target0
        
        return targets
    
    def retarget_score(self, detected_kpts, thetas):
        '''
        calculate the score of the retargeting result.
        '''
        indices = self.dex_retargeter.optimizer.target_link_human_indices
        targets = mano['normalized_mano_joints'][1]
        joint_pos = self.target_coordinate_transformations(targets.copy())
        
        ref_value = joint_pos[indices[1, :], :] - joint_pos[indices[0, :], :]
        retarget_robot_vec = self.calculate_robot_vec(thetas)[0]
        score_vector = np.linalg.norm(ref_value - retarget_robot_vec)
        
        # print("mano=", mano)
        thumb_rot = mano['pred_mano_params'][1]['hand_pose'][12] # (3, 3)
        thumb_rot = np.arctan(thumb_rot[2, 1] / thumb_rot[1, 1])
        retarget_thumb_rot = thetas[-3]
        # print("thumb_rot=", thumb_rot, "retarget_thumb_rot=", retarget_thumb_rot, "in degree=", thumb_rot * 180 / np.pi, retarget_thumb_rot * 180 / np.pi)
        score_rot = np.abs(thumb_rot - retarget_thumb_rot)
        
        score = score_vector + score_rot
        # print("score=", score_vector, score_rot, score)
        return score
    
    # def __call__(self, mano):
    #     '''
    #     order of the targets:
    #         base(1), thumb(4), index(4), middle(4), ring(4), pinky(4); for each finger, the order is mcp, pip, dip, tip
            
    #     order of the thetas:
    #         index(2), middle(2), ring(2), pinky(2), thumb(3 = 1rot + 2pos)
    #     '''
        
    #     indices = self.dex_retargeter.optimizer.target_link_human_indices
        
    #     targets = mano['normalized_mano_joints'][1]
    #     joint_pos = targets.copy()
    #     joint_pos = self.target_coordinate_transformations(joint_pos)
    #     # print("targets=", joint_pos)
        
    #     ref_value = joint_pos[indices[1, :], :] - joint_pos[indices[0, :], :]
    #     # print("ref value=", ref_value)
    #     thetas = self.dex_retargeter.retarget(ref_value)
        
    #     robot_vec, ori_link_pos, target_lin_pos, retarget_joint_pos = self.calculate_robot_vec(thetas)
    #     # print("thetas=", thetas)
        
    #     return thetas, {'ref_value': ref_value, 'target_joint_pos': joint_pos, 'robot_vec': robot_vec, 'robot_ori_link_pos': ori_link_pos, 'robot_target_link_pos': target_lin_pos, 'retarget_joint_pos': retarget_joint_pos, 'score': self.retarget_score(mano, thetas)}
    
    def __call__(self, joint_pos):
        
        print("detected kpts=", joint_pos)
        joint_pos = self.coordinate_transformation(joint_pos)
        print("targets=", joint_pos)
        
        indices = self.dex_retargeter.optimizer.target_link_human_indices
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        print("ref value=", ref_value)
        qpos = self.dex_retargeter.retarget(ref_value)
        
        robot_vec, origin_link_pos, task_link_pos, retarget_joint_pos = self.calculate_robot_vec(qpos)
        print("robot_vec=", robot_vec)

        return qpos

class OYRetarget:

    def __init__(self, discretize=False):
        
        root_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(root_dir, 'config/dex-hand-teleop.yml')
        with open(config_path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        
        # config_path_in = 'algos/config/dex-hand-fix-thumb-in.yml'
        # with open(config_path_in, 'r') as f:
        #     config_in = EasyDict(yaml.safe_load(f))
        
        # config_path_out = 'algos/config/dex-hand-fix-thumb-out.yml'
        # with open(config_path_out, 'r') as f:
        #     config_out = EasyDict(yaml.safe_load(f))
        
        self.dex_retarget = DexRetargetWrapper(config)
        # self.dex_retarget_in = DexRetargetWrapper(config_in)
        # self.dex_retarget_out = DexRetargetWrapper(config_out)

        self.discretize = discretize

    def retarget(self, detected_3d_kpts):
        return self.dex_retarget(detected_3d_kpts)
    

if __name__ == '__main__':
    
    pass