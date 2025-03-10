#!/usr/bin/env python

import argparse
import heapq
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import g2o
import numpy as np
import pyray
import raylib
import skimage
from numpy.typing import NDArray

import icp
import pose_graph


@dataclass
class Laser:
    angle_min: float
    angle_max: float
    angle_increment: float
    max_distance: float
    distances: list[float] | NDArray[float]


@dataclass
class Odometry:
    transform: NDArray[float]


@dataclass
class Reading:
    data: Laser | Odometry
    timestamp: float = field(default_factory=time.time)

    def __eq__(self, other):
        self.timestamp == other.timestamp

    def __lt__(self, other):
        self.timestamp < other.timestamp

    def __le__(self, other):
        (self < other) or (self == other)

    def __gt__(self, other):
        other < self

    def __ge__(self, other):
        other <= self


class SLAM:
    def __init__(self):
        self.optimizer = pose_graph.PoseGraphOptimization()
        self.vertex_counter = itertools.count(0)
        self.prev_odom = g2o.SE2(g2o.Isometry2d(np.eye(3)))
        self.optimizer.add_vertex(next(self.vertex_counter), self.prev_odom, True)
        self.last_vertex_idx = 0
        self.registered_scans = dict()

    def add_reading(self, reading: Reading):
        data = reading.data

        cleanup = False

        if isinstance(data, Laser):
            angles = np.arange(data.angle_min, data.angle_max, data.angle_increment)
            positions = (
                np.array([np.cos(angles), np.sin(angles)]).T
                * data.distances[:, np.newaxis]
            )
            mask = np.linalg.norm(positions, axis=1) < data.max_distance
            positions = positions[mask]

            if not len(self.registered_scans):
                self.registered_scans[self.last_vertex_idx] = positions
                return

            prev_idx = next(reversed(self.registered_scans.keys()))
            prev_scan = self.registered_scans[prev_idx]
            prev_pose = self.optimizer.get_pose(prev_idx)
            current_pose = self.optimizer.get_pose(self.last_vertex_idx)
            transform = prev_pose.inverse() * current_pose
            if (
                np.linalg.norm(transform.translation()) > 0.4
                or abs(transform.rotation().angle()) > 0.2
            ):
                transformation, distances, iter = icp.icp(
                    positions,
                    prev_scan,
                    transform.to_isometry().matrix(),
                    max_iterations=80,
                    tolerance=0.00001,
                )
                self.optimizer.add_edge(
                    [prev_idx, self.last_vertex_idx],
                    g2o.SE2(g2o.Isometry2d(transformation)),
                    # TODO: How to calculate information matrix?
                    information=100.0 / np.mean(distances) * np.eye(3),
                )
                self.registered_scans[self.last_vertex_idx] = positions
                cleanup = True

                # Loop closure only makes sense if the last reading is a valid laser scan
                self.loop_closure()

        elif isinstance(data, Odometry):
            current_odom = g2o.SE2(g2o.Isometry2d(data.transform))
            transform = self.prev_odom.inverse() * current_odom

            if (np.linalg.norm(transform.translation()) < np.finfo(float).eps) and (
                transform.rotation().angle() < np.finfo(float).eps
            ):
                return

            vertex_idx = next(self.vertex_counter)
            assert data.transform.shape == (3, 3)
            self.optimizer.add_vertex(vertex_idx, current_odom)
            self.optimizer.add_edge(
                [vertex_idx - 1, vertex_idx],
                transform,
                information=np.eye(3),
            )
            self.last_vertex_idx = vertex_idx
            self.prev_odom = current_odom
        else:
            raise ValueError(f"Unknown data type: {type(data)}")

        if self.last_vertex_idx > 1:
            # self.optimizer.set_verbose(True)
            self.optimizer.optimize()

            if cleanup:
                keys = reversed(self.registered_scans.keys())
                end = next(keys)
                start = next(keys)
                for vertex_idx in range(start + 1, end):
                    self.optimizer.remove_vertex(self.optimizer.vertex(vertex_idx))

    def loop_closure(self):
        if len(self.registered_scans) < 2:
            return

        scans = enumerate(reversed(self.registered_scans.items()))
        scan_idx, (current_idx, current_scan) = next(scans)
        assert current_idx == self.last_vertex_idx
        current_pose = self.optimizer.get_pose(current_idx)

        for prev_scan_idx, (prev_idx, prev_scan) in scans:
            if prev_scan_idx - scan_idx < 10:
                # Too recent
                continue

            vertex = self.optimizer.vertex(prev_idx)
            inv_hessian, valid = self.optimizer.compute_marginals(vertex)

            if not valid:
                continue

            prev_pose = vertex.estimate()
            position_diff = (prev_pose.inverse() * current_pose).translation()

            for index, block in enumerate(inv_hessian.block_cols()):
                if block:
                    cov = block[index]
                    break

            cov = np.linalg.inv(cov)
            position_cov = cov[:2, :2]

            mahalanobis_dist = np.sqrt(
                position_diff.T @ np.linalg.inv(position_cov) @ position_diff
            )

            if mahalanobis_dist < 3:
                transformation, distances, iter = icp.icp(
                    prev_scan,
                    current_scan,
                    (prev_pose.inverse() * current_pose).to_isometry().matrix(),
                    max_iterations=80,
                    tolerance=0.0001,
                )

                if len(distances) and np.mean(distances) < 0.15:
                    rk = g2o.RobustKernelDCS(10)
                    self.optimizer.add_edge(
                        [current_idx, prev_idx],
                        g2o.SE2(g2o.Isometry2d(transformation)),
                        robust_kernel=rk,
                        information=0.1 / np.mean(distances) * np.eye(3),
                    )


def homogeneous_transform(vector):
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
        readings = []
        for line in f:
            tokens = line.split(" ")
            if tokens[0] == "FLASER":
                num_readings = int(tokens[1])
                scans = np.array(tokens[2 : 2 + num_readings], dtype=float)
                timestamp = float(tokens[8 + num_readings])

                heapq.heappush(
                    readings,
                    Reading(
                        data=Laser(
                            angle_min=-np.pi / 2,
                            angle_max=np.pi / 2,
                            angle_increment=np.pi / num_readings,
                            max_distance=80,
                            distances=scans,
                        ),
                        timestamp=timestamp,
                    ),
                )
            elif tokens[0] == "ODOM":
                x = float(tokens[1])
                y = float(tokens[2])
                theta = float(tokens[3])
                timestamp = float(tokens[7])
                heapq.heappush(
                    readings,
                    Reading(
                        data=Odometry(transform=homogeneous_transform([x, y, theta])),
                        timestamp=timestamp,
                    ),
                )

    return readings


parser = argparse.ArgumentParser(description="Python Graph SLAM")
parser.add_argument("--input", type=Path, help="Input CLF File.", required=True)
args = parser.parse_args()

raylib.InitWindow(800, 800, b"Python Graph SLAM")
raylib.SetExitKey(pyray.KEY_Q)

camera = pyray.Camera3D(
    (0.0, 0.0, 10.0),
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    45.0,
    pyray.CAMERA_PERSPECTIVE,
)

resolution = 0.1
width = 200
height = 200
occupancy_model = pyray.load_model_from_mesh(pyray.gen_mesh_plane(width, height, 1, 1))
image = pyray.Image()
image.width = int(width / resolution)
image.height = int(height / resolution)
image.mipmaps = 1
image.format = pyray.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
occupancy_update_interval = 50
texture = raylib.LoadTextureFromImage(image)
pyray.set_material_texture(
    occupancy_model.materials[0],
    pyray.MATERIAL_MAP_DIFFUSE,
    texture,
)
grid = np.zeros((int(height / resolution), int(width / resolution)))

slam = SLAM()

for idx, reading in enumerate(load_clf_from_file(args.input)):
    slam.add_reading(reading)

    raylib.BeginDrawing()
    raylib.ClearBackground(pyray.WHITE)

    pyray.begin_mode_3d(camera)
    pyray.update_camera(camera, pyray.CAMERA_THIRD_PERSON)

    prev_pose = None
    if not (idx % occupancy_update_interval):
        grid = np.zeros((int(height / resolution), int(width / resolution)))

    for vertex_idx, scan in slam.registered_scans.items():
        pose = slam.optimizer.get_pose(vertex_idx)
        rotation_matrix = pose.rotation().rotation_matrix()
        translation = pose.translation()
        scan = (rotation_matrix @ scan.T + translation[:, np.newaxis]).T
        for point in scan:
            # Draw point cloud
            # raylib.DrawPoint3D(
            #     (point[0], 0.1, point[1]),
            #     pyray.BLUE,
            # )

            # Build occupancy map
            if not (idx % occupancy_update_interval):
                line = list(
                    zip(
                        *skimage.draw.line(
                            int(translation[1] / resolution + height / resolution / 2),
                            int(translation[0] / resolution + width / resolution / 2),
                            int(point[1] / resolution + height / resolution / 2),
                            int(point[0] / resolution + width / resolution / 2),
                        )
                    )
                )

                grid[line[-1]] += np.log10(0.7 / 0.3)
                for pos in line[:-1]:
                    grid[pos] += np.log10(0.3 / 0.7)

        if prev_pose:
            # Draw trajectory
            prev_translation = prev_pose.translation()
            raylib.DrawLine3D(
                (prev_translation[0], 0.1, prev_translation[1]),
                (translation[0], 0.1, translation[1]),
                pyray.RED,
            )

        prev_pose = pose

    if not (idx % occupancy_update_interval):
        gridp = (255 - 255 / (1 + np.exp(-grid))).astype(np.uint8)
        raylib.UpdateTexture(texture, pyray.ffi.from_buffer(gridp.data))

    # Draw occupancy map
    pyray.draw_model(occupancy_model, (0.0, 0.0, 0.0), 1.0, pyray.WHITE)

    # Draw robot position
    size = 0.3
    pose = slam.optimizer.get_pose(slam.last_vertex_idx)
    transform_matrix = pose.to_isometry().matrix()
    tip = transform_matrix @ np.array([size, 0.0, 1.0])
    left = transform_matrix @ np.array([-size * 0.5, -size * 0.5, 1.0])
    right = transform_matrix @ np.array([-size * 0.5, size * 0.5, 1.0])
    raylib.DrawTriangle3D(
        (tip[0], 0.1, tip[1]),
        (left[0], 0.1, left[1]),
        (right[0], 0.1, right[1]),
        pyray.RED,
    )

    # raylib.DrawGrid(50, 1.0)
    raylib.EndMode3D()
    raylib.EndDrawing()


pyray.close_window()
