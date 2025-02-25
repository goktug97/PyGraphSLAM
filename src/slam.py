#!/usr/bin/env python

import argparse
from pathlib import Path

import g2o
import numpy as np
import pyray
import raylib
import scipy

import icp
import pose_graph

# def hessian_matrix(hessian_fun):
#     hessian = np.ndarray((3, 3))
#     for i in range(3):
#         for j in range(3):
#             hessian[i, j] = hessian_fun(i, j)
#     return hessian


# def eigsorted(cov):
#     vals, vecs = np.linalg.eigh(cov)
#     order = vals.argsort()[::-1]
#     return vals[order], vecs[:, order]


def transformation_matrix(vector):
    x, y, theta = vector
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0.0, 0.0, 1.0],
        ]
    )


def load_clf_from_file(clf_file: Path):
    with clf_file.open() as f:
        laser_readings = []
        odoms = []
        for line in f:
            tokens = line.split(" ")
            if tokens[0] == "FLASER":
                num_readings = int(tokens[1])
                scans = np.array(tokens[2 : 2 + num_readings], dtype=float)

                angles = np.linspace(-np.pi / 2, np.pi / 2, num_readings)

                laser_readings.append(
                    np.array([np.cos(angles), np.sin(angles)]).T * scans[:, np.newaxis]
                )
                x = float(tokens[2 + num_readings])
                y = float(tokens[3 + num_readings])
                theta = float(tokens[4 + num_readings])
                odoms.append(np.array([x, y, theta]))

    return laser_readings, odoms


parser = argparse.ArgumentParser(description="Python Graph Slam")
parser.add_argument("--input", type=Path, help="Input CLF File.", required=True)
args = parser.parse_args()

raylib.InitWindow(800, 800, b"Python Graph SLAM")
raylib.SetExitKey(pyray.KEY_Q)
raylib.SetTargetFPS(60)

camera = pyray.Camera3D(
    (0.0, 0.0, 10.0),
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    45.0,
    pyray.CAMERA_PERSPECTIVE,
)

laser_readings, odoms = load_clf_from_file(args.input)

# Starting point
pose = transformation_matrix(odoms[0])
optimizer = pose_graph.PoseGraphOptimization()
optimizer.add_vertex(0, g2o.SE2(g2o.Isometry2d(pose)), True)

vertex_idx = 1
registered_lasers = []

prev_odom = odoms.pop(0)
prev_laser = laser_readings.pop(0)
registered_lasers = [prev_laser]

for odom, laser in zip(odoms, laser_readings):
    if raylib.WindowShouldClose():
        break

    dx = odom - prev_odom
    if np.linalg.norm(dx[0:2]) > 0.4 or abs(dx[2]) > 0.2:
        # Scan Matching
        init_pose = transformation_matrix(dx)
        transformation, distances, iter = icp.icp(
            laser, prev_laser, init_pose, max_iterations=80, tolerance=0.0001
        )

        pose = np.matmul(pose, transformation)
        optimizer.add_vertex(vertex_idx, g2o.SE2(g2o.Isometry2d(pose)))
        rk = g2o.RobustKernelHuber()
        optimizer.add_edge(
            [vertex_idx - 1, vertex_idx],
            g2o.SE2(g2o.Isometry2d(transformation)),
            robust_kernel=rk,
            information=100.0 * np.eye(3),
        )

        # Loop Closure
        if vertex_idx > 10 and not vertex_idx % 10:
            poses = [
                optimizer.get_pose(idx).to_vector()[0:2]
                for idx in range(vertex_idx - 1)
            ]
            kd = scipy.spatial.cKDTree(poses)
            pose = optimizer.get_pose(vertex_idx)
            x, y, _ = pose.to_vector()
            idxs = kd.query_ball_point(np.array([x, y]), r=4.25)
            for idx in idxs:
                prev_laser = registered_lasers[idx]

                transformation, distances, iter = icp.icp(
                    prev_laser,
                    laser,
                    np.eye(3),
                    max_iterations=80,
                    tolerance=0.0001,
                )

                if len(distances) and np.mean(distances) < 0.15:
                    (x, y, theta) = (
                        pose.inverse() * optimizer.get_pose(idx)
                    ).to_vector()

                    if np.sqrt(x**2 + y**2) < 5.0 and theta < np.pi / 2:
                        rk = g2o.RobustKernelDCS()
                        optimizer.add_edge(
                            [vertex_idx, idx],
                            g2o.SE2(g2o.Isometry2d(transformation)),
                            robust_kernel=rk,
                            information=np.eye(3) * 0.1,
                        )

        optimizer.optimize()
        pose = optimizer.get_pose(vertex_idx).to_isometry().matrix()

        vertex_idx += 1
        prev_odom = odom
        prev_laser = laser
        registered_lasers.append(laser)

        # Draw trajectory and map
        raylib.BeginDrawing()
        raylib.ClearBackground(pyray.WHITE)

        pyray.begin_mode_3d(camera)
        pyray.update_camera(camera, pyray.CAMERA_THIRD_PERSON)

        prev = None
        for idx in range(0, vertex_idx):
            x = optimizer.get_pose(idx)
            r = x.to_isometry().R
            t = x.to_isometry().t
            filtered = registered_lasers[idx]
            filtered = filtered[np.linalg.norm(filtered, axis=1) < 80]
            for point in (r @ filtered.T + t[:, np.newaxis]).T:
                raylib.DrawPoint3D((point[0], 0.0, point[1]), pyray.BLUE)
            position = x.to_vector()
            position = (position[0], 0.0, position[1])
            if prev:
                raylib.DrawLine3D(prev, position, pyray.RED)
            prev = position

        raylib.DrawGrid(50, 1.0)
        raylib.EndMode3D()
        raylib.EndDrawing()


pyray.close_window()
