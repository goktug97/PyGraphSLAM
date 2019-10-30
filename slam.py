#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys
import icp
import g2o
import pose_graph
import scipy
import argparse
import imageio
import time

plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit() if event.key == 'escape' else None])
plt.gcf().gca().set_aspect('equal')

parser = argparse.ArgumentParser(description='Python Graph Slam')

parser.add_argument('--seed', default=None, type=int,
                    help='Random number generator seed')

parser.add_argument('--dataset', default='intel', const='intel', nargs='?',
                    choices=['intel', 'fr', 'aces'], help='Datasets')

parser.add_argument('--save_gif', dest='save_gif', action='store_true')
parser.set_defaults(save_gif=False)

args = parser.parse_args()
    
if args.save_gif:
    import atexit
    images = []
    atexit.register(lambda: imageio.mimsave(f'./slam_{int(time.time())}.gif',
                                            images, fps=10))

if args.seed is not None:
    np.random.seed(args.seed) # For testing

# Starting point 
optimizer = pose_graph.PoseGraphOptimization()
pose = np.eye(3)
optimizer.add_vertex(0, g2o.SE2(g2o.Isometry2d(pose)), True)

lasers = np.load(f'./datasets/{args.dataset}_lasers.npy', allow_pickle=True)
odoms = np.load(f'./datasets/{args.dataset}_odoms.npy', allow_pickle=True)

init_pose = np.eye(3)
vertex_idx = 1
registered_lasers = []

for odom_idx, odom in enumerate(odoms):
    # Initialize
    if odom_idx == 0:
        prev_odom = odom.copy()
        prev_idx = 0
        B = lasers[odom_idx]
        registered_lasers.append(B)
        continue

    dx = odom - prev_odom
    if np.linalg.norm(dx[0:2]) > 0.5 or abs(dx[2]) > 0.2:
        # Scan Matching
        A = lasers[prev_idx]
        B = lasers[odom_idx]
        size = np.min([A.shape[0], B.shape[0]])
        prev_random_idx = np.random.choice(np.arange(A.shape[0]), size, replace=True)
        cur_random_idx = np.random.choice(np.arange(B.shape[0]), size, replace=True)

        try:
            tran, distances, iter = icp.icp(
                B[cur_random_idx], A[prev_random_idx], init_pose,
                max_iterations=80, tolerance=0.0001)
        except ValueError:
            continue

        init_pose = tran
        pose = np.matmul(pose, tran)
        optimizer.add_vertex(vertex_idx, g2o.SE2(g2o.Isometry2d(pose)))
        rk = g2o.RobustKernelDCS()
        information = np.eye(3)
        optimizer.add_edge([vertex_idx-1, vertex_idx],
                           g2o.SE2(g2o.Isometry2d(tran)),
                           information, robust_kernel=rk)

        prev_odom = odom
        prev_idx = odom_idx
        registered_lasers.append(B)

        # Loop Closure
        if vertex_idx > 1 and not vertex_idx % 10:
            poses = [optimizer.get_pose(idx).to_vector()[0:2]
                     for idx in range(vertex_idx-1)]
            kd = scipy.spatial.cKDTree(poses)
            x, y, theta = optimizer.get_pose(idx).to_vector()
            direction = np.array([np.cos(theta), np.sin(theta)])
            idxs = kd.query_ball_point(np.array([x, y]) + direction*2, r=2)
            for idx in idxs:
                A = registered_lasers[idx]
                size = np.min([A.shape[0], B.shape[0]])
                prev_random_idx = np.random.choice(
                    np.arange(A.shape[0]), size, replace=True)
                cur_random_idx = np.random.choice(
                    np.arange(B.shape[0]), size, replace=True)
                try:
                    tran, distances, iter = icp.icp(
                        A[prev_random_idx], B[cur_random_idx], np.eye(3),
                        max_iterations=80, tolerance=0.0001)
                except ValueError:
                    continue
                information = np.eye(3)
                if np.mean(distances) < 0.2:
                    rk = g2o.RobustKernelDCS()
                    optimizer.add_edge([vertex_idx, idx],
                                       g2o.SE2(g2o.Isometry2d(tran)),
                                       information, robust_kernel=rk)
            optimizer.optimize()
            pose = optimizer.get_pose(vertex_idx).to_isometry().matrix()

        # Draw trajectory and map
        map_size = 60
        traj = []
        point_cloud = []
        draw_last = float('inf')

        for idx in range(max(0, vertex_idx-draw_last), vertex_idx):
            x = optimizer.get_pose(idx)
            r = x.to_isometry().R
            t = x.to_isometry().t
            point_cloud.append((r @ registered_lasers[idx].T + t[:, np.newaxis]).T)
            traj.append(x.to_vector()[0:2])
        point_cloud = np.vstack(point_cloud)

        xyreso = 0.01 # Map resolution (m)
        point_cloud = (point_cloud / xyreso).astype('int')
        point_cloud = np.unique(point_cloud, axis=0)
        point_cloud = point_cloud * xyreso

        plt.cla()

        # To make map static, draw some fixed points
        plt.plot(map_size/2, map_size/2, '.b')
        plt.plot(-map_size/2, map_size/2, '.b')
        plt.plot(map_size/2, -map_size/2, '.b')
        plt.plot(-map_size/2, -map_size/2, '.b')

        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], '-g')
        plt.plot(point_cloud[:, 0], point_cloud[:, 1], '.b', markersize=0.1)
        plt.pause(0.0001)

        if args.save_gif:
            plt.gcf().canvas.draw()
            image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
            images.append(image)

        vertex_idx += 1



