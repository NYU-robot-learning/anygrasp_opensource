import os
import sys
import numpy as np
import argparse
from PIL import Image, ImageDraw
import time
import scipy.io as scio
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

from matplotlib import pyplot as plt
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

import zmq
import math, copy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='./graspnet')
parser.add_argument('--checkpoint_path', default='./logs/minkuresunet_realsense.tar')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='./logs/')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=200000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=-1,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--scene', type=str, default='0118')
parser.add_argument('--index', type=str, default='0256')
parser.add_argument('--open_communication', action='store_true', help='Use image transferred from the robot')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)

def process():
    root = cfgs.dataset_root
    camera_type = cfgs.camera

    H, W = colors.shape
    camera = CameraInfo(H, W, fx, fy, cx, cy)


    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    mask = (depth_mask)

    cloud_masked = cloud[mask]
    colors_masked = colors[mask]

    # sample points random
    print(f"total points: {len(cloud_masked)}")
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    colors_masked = colors_masked[idxs]

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'colors': (colors_masked/255.0).astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                }
    
    gg = inference(ret_dict)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference(points, colors, cfgs):
    data_input = {}
    data_input['point_clouds'] = points.astype(np.float32)
    data_input['colors'] = colors.astype(np.float32)
    data_input['coors'] = points.astype(np.float32) / cfgs.voxel_size
    data_input['feats'] = np.ones_like(points).astype(np.float32)
    batch_data = minkowski_collate_fn([data_input])
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    net.eval()
    tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()

    # Filtering grasp poses for real-world execution. 
    # The first mask preserves the grasp poses that are within a 30-degree angle with the vertical pose and have a width of less than 9cm.
    # mask = (preds[:,9] > 0.9) & (preds[:,1] < 0.09)
    # The second mask preserves the grasp poses within the workspace of the robot.
    # workspace_mask = (preds[:,12] > -0.20) & (preds[:,12] < 0.21) & (preds[:,13] > -0.06) & (preds[:,13] < 0.18) & (preds[:,14] > 0.63) 
    # preds = preds[mask & workspace_mask]

    # if len(preds) == 0:
    #         print('No grasp detected after masking')
    #         return

    gg = GraspGroup(preds)
    # collision detection
    if cfgs.collision_thresh > 0:
        cloud = data_input['point_clouds']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]

    # save grasps
    # save_dir = os.path.join(cfgs.dump_dir, scene_id, cfgs.camera)
    # save_path = os.path.join(save_dir, cfgs.index + '.npy')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # gg.save_npy(save_path)

    toc = time.time()
    print('inference time: %fs' % (toc - tic))

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return gg, cloud


if __name__ == '__main__':
    scene_id = 'scene_' + cfgs.scene
    index = cfgs.index
    data_dict = process()

    if cfgs.infer:
        inference(data_dict)
    if cfgs.vis:
        pc = data_dict['point_clouds']
        cc = data_dict['colors']
        gg = np.load(os.path.join(cfgs.dump_dir, scene_id, cfgs.camera, cfgs.index + '.npy'))
        gg = GraspGroup(gg)
        gg = gg.nms()
        gg = gg.sort_by_score()
        print(f"num grapss : {gg.__len__()}, {gg[0].score}")
        # if gg.__len__() > 100:
        #     gg = gg[:100]
        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(cc.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])

        # # Example code for execution
        # g = gg[0]
        # translation = g.translation
        # rotation = g.rotation_matrix

        # pose = translation_rotation_2_matrix(translation,rotation) #transform into 4x4 matrix, should be easy
        # # Transform the grasp pose from camera frame to robot coordinate, implement according to your robot configuration
        # tcp_pose = Camera_To_Robot(pose)

        
        # tcp_ready_pose = copy.deepcopy(tcp_pose)
        # tcp_ready_pose[:3, 3] = tcp_ready_pose[:3, 3] - 0.1 * tcp_ready_pose[:3, 2] # The ready pose is backward along the actual grasp pose by 10cm to avoid collision
       
        # tcp_away_pose = copy.deepcopy(tcp_pose)
        
        # # to avoid the gripper rotate around the z_{tcp} axis in the clock-wise direction.
        # tcp_away_pose[3,:3] = np.array([0,0,-1], dtype=np.float64)
        
        # # to avoid the object collide with the scene.
        # tcp_away_pose[2,3] += 0.1

        # # We rely on python-urx to send the tcp pose the ur5 arm, the package is available at https://github.com/SintefManufacturing/python-urx
        # urx.movels([tcp_ready_pose, tcp_pose], acc = acc, vel = vel, radius = 0.05)

        # # CLOSE_GRIPPER(), implement according to your robot configuration
        # urx.movels([tcp_away_pose, self.throw_pose()], acc = 1.2 * acc, vel = 1.2 * vel, radius = 0.05, wait=False)

