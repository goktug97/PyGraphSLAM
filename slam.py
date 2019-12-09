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

from matplotlib.patches import Ellipse

def hessian_matrix(hessian_fun):
    hessian = np.ndarray((3, 3))
    for i in range(3):
        for j in range(3):
            hessian[i, j] = hessian_fun(i, j)
    return hessian

def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit() if event.key == 'escape' else None])
plt.gcf().gca().set_aspect('equal')
plt.gcf().canvas.set_window_title('float')
plt.gcf().tight_layout(pad=0)

parser = argparse.ArgumentParser(description='Python Graph Slam')
parser.add_argument('--input', type=str, help='Input CLF File.', required=True)
parser.add_argument('--draw_last', default=float('inf'), type=int,
                    help='Number of point clouds to draw.')
parser.add_argument('--save_gif', dest='save_gif', action='store_true')
parser.set_defaults(save_gif=False)
args = parser.parse_args()

if args.save_gif:
    import atexit
    images = []
    atexit.register(lambda: imageio.mimsave(f'./slam_{int(time.time())}.gif',
                                            images, fps=10))

# Read Data
with open(args.input, 'r') as f:
   lasers = []
   odoms = []
   for line in f:
       tokens = line.split(' ')
       if tokens[0] == 'FLASER':
           num_readings = int(tokens[1])
           scans = np.array(tokens[2:2+num_readings], dtype=np.float)
           scan_time = float(tokens[2+num_readings+6])
           index = np.arange(-90, 90+180/num_readings, 180/num_readings)
           index = np.delete(index, num_readings//2)
           converted_scans = []
           angles = np.radians(index)
           converted_scans = np.array([np.cos(angles), np.sin(angles)]).T * scans[:, np.newaxis]
           lasers.append(np.array(converted_scans))
           x = float(tokens[2+num_readings])
           y = float(tokens[3+num_readings])
           theta = float(tokens[4+num_readings])
           odoms.append([x, y, theta])

odoms = np.array(odoms)
lasers = np.array(lasers)

# Starting point 
optimizer = pose_graph.PoseGraphOptimization()
pose = np.eye(3)
optimizer.add_vertex(0, g2o.SE2(g2o.Isometry2d(pose)), True)

init_pose = np.eye(3)
vertex_idx = 1
registered_lasers = []

max_x = -float('inf')
max_y = -float('inf')
min_x = float('inf')
min_y = float('inf')

for odom_idx, odom in enumerate(odoms):
    # Initialize
    if odom_idx == 0:
        prev_odom = odom.copy()
        prev_idx = 0
        B = lasers[odom_idx]
        registered_lasers.append(B)
        continue

    dx = odom - prev_odom
    if np.linalg.norm(dx[0:2]) > 0.4 or abs(dx[2]) > 0.2:
        # Scan Matching
        A = lasers[prev_idx]
        B = lasers[odom_idx]
        x, y, yaw = dx[0], dx[1], dx[2]
        init_pose = np.array([[np.cos(yaw), -np.sin(yaw), x],
                              [np.sin(yaw), np.cos(yaw), y],
                              [0, 0, 1]])

        with np.errstate(all='raise'):
            try:
                tran, distances, iter, cov = icp.icp(
                    B, A, init_pose,
                    max_iterations=80, tolerance=0.0001)
            except Exception as e:
                continue

        init_pose = tran
        pose = np.matmul(pose, tran)
        optimizer.add_vertex(vertex_idx, g2o.SE2(g2o.Isometry2d(pose)))
        rk = g2o.RobustKernelDCS()
        information = np.linalg.inv(cov)
        optimizer.add_edge([vertex_idx-1, vertex_idx],
                           g2o.SE2(g2o.Isometry2d(tran)),
                           information, robust_kernel=rk)

        prev_odom = odom
        prev_idx = odom_idx
        registered_lasers.append(B)

        # Loop Closure
        if vertex_idx > 10 and not vertex_idx % 10:
            poses = [optimizer.get_pose(idx).to_vector()[0:2]
                    for idx in range(vertex_idx-1)]
            kd = scipy.spatial.cKDTree(poses)
            x, y, theta = optimizer.get_pose(idx).to_vector()
            direction = np.array([np.cos(theta), np.sin(theta)])
            idxs = kd.query_ball_point(np.array([x, y]), r=4.25)
            for idx in idxs:
                A = registered_lasers[idx]
                with np.errstate(all='raise'):
                    try:
                        tran, distances, iter, cov = icp.icp(
                            A, B, np.eye(3),
                            max_iterations=80, tolerance=0.0001)
                    except Exception as e:
                        continue
                information = np.linalg.inv(cov)
                if np.mean(distances) < 0.15:
                    rk = g2o.RobustKernelDCS()
                    optimizer.add_edge([vertex_idx, idx],
                                       g2o.SE2(g2o.Isometry2d(tran)),
                                       information, robust_kernel=rk)

            optimizer.optimize()
            pose = optimizer.get_pose(vertex_idx).to_isometry().matrix()

        # Draw trajectory and map
        traj = []
        point_cloud = []
        draw_last = args.draw_last

        for idx in range(max(0, vertex_idx-draw_last), vertex_idx):
            x = optimizer.get_pose(idx)
            r = x.to_isometry().R
            t = x.to_isometry().t
            filtered = registered_lasers[idx]
            filtered = filtered[np.linalg.norm(filtered, axis=1) < 80]
            point_cloud.append((r @ filtered.T + t[:, np.newaxis]).T)
            traj.append(x.to_vector()[0:2])
        point_cloud = np.vstack(point_cloud)

        xyreso = 0.01 # Map resolution (m)
        point_cloud = (point_cloud / xyreso).astype('int')
        point_cloud = np.unique(point_cloud, axis=0)
        point_cloud = point_cloud * xyreso

        current_max = np.max(point_cloud, axis=0)
        current_min = np.min(point_cloud, axis=0)
        max_x = max(max_x, current_max[0])
        max_y = max(max_y, current_max[1])
        min_x = min(min_x, current_min[0])
        min_y = min(min_y, current_min[1])

        plt.cla()
        plt.axis([min_x, max_x, min_y, max_y])

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
