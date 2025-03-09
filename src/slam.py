#!/usr/bin/env python

import argparse
from pathlib import Path
import time
import itertools
from dataclasses import dataclass, field
import heapq

import g2o
import numpy as np
from numpy.typing import NDArray
import pyray
import raylib

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
            self.optimizer.optimize()

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

            cov = inv_hessian.block(prev_idx - 1, prev_idx - 1)
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
                            angle_increment=np.pi / 180,
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


def translation_to_raylib(translation):
    return (
        translation[0],
        0.0,
        translation[1],
    )


def pose_to_raylib(pose):
    return (translation_to_raylib(pose.translation()), pose.rotation().angle())


parser = argparse.ArgumentParser(description="Python Graph Slam")
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

slam = SLAM()
readings = load_clf_from_file(args.input)
for idx, reading in enumerate(readings):
    slam.add_reading(reading)

    raylib.BeginDrawing()
    raylib.ClearBackground(pyray.WHITE)

    pyray.begin_mode_3d(camera)
    pyray.update_camera(camera, pyray.CAMERA_THIRD_PERSON)

    prev_pose = None
    for vertex_idx, scan in slam.registered_scans.items():
        pose = slam.optimizer.get_pose(vertex_idx)
        r = pose.rotation().rotation_matrix()
        t = pose.translation()
        scan = (r @ scan.T + t[:, np.newaxis]).T
        for point in scan:
            raylib.DrawPoint3D(
                translation_to_raylib(point),
                pyray.BLUE,
            )
        translation, rotation = pose_to_raylib(pose)

        if prev_pose:
            raylib.DrawLine3D(
                translation_to_raylib(prev_pose.translation()), translation, pyray.RED
            )

        prev_pose = pose

    size = 0.3
    pose = slam.optimizer.get_pose(slam.last_vertex_idx)
    transform_matrix = pose.to_isometry().matrix()
    tip = transform_matrix @ np.array([size, 0.0, 1.0])
    left = transform_matrix @ np.array([-size * 0.5, -size * 0.5, 1.0])
    right = transform_matrix @ np.array([-size * 0.5, size * 0.5, 1.0])
    raylib.DrawTriangle3D(
        translation_to_raylib(left[:2]),
        translation_to_raylib(right[:2]),
        translation_to_raylib(tip[:2]),
        pyray.RED,
    )

    raylib.DrawGrid(50, 1.0)
    raylib.EndMode3D()
    raylib.EndDrawing()


pyray.close_window()
