import math

import cv2
import numpy as np
import pytest

from app.analyzers.pose import (
    HeadPose,
    angles_from_rotation,
    camera_matrix,
    canonical_model,
    estimate_head_pose,
)

WIDTH, HEIGHT = 640, 480
DISTANCE = 55.0


def rotation_y(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_x(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_z(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def project(rotation):
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.array([[0.0], [0.0], [DISTANCE]])
    points, _ = cv2.projectPoints(canonical_model(), rvec, tvec, camera_matrix(WIDTH, HEIGHT), None)
    return points.reshape(-1, 2)


def test_canonical_model_shape_and_frame():
    model = canonical_model()
    assert model.shape == (468, 3)
    nose, chin = model[1], model[152]
    assert nose[2] < chin[2]
    assert chin[1] > nose[1]


def test_identity_gives_zero_angles():
    pose = estimate_head_pose(project(np.eye(3)), WIDTH, HEIGHT)
    assert abs(pose.yaw) < 1.0
    assert abs(pose.pitch) < 1.0
    assert abs(pose.roll) < 1.0


@pytest.mark.parametrize("degrees", [20.0, -20.0])
def test_yaw_sign_matches_nose_direction(degrees):
    points = project(rotation_y(degrees))
    pose = estimate_head_pose(points, WIDTH, HEIGHT)
    assert pose.yaw == pytest.approx(degrees, abs=1.0)
    assert abs(pose.pitch) < 1.0
    nose_x = points[1][0]
    eyes_mid_x = (points[33][0] + points[263][0]) / 2
    assert (nose_x < eyes_mid_x) == (degrees > 0)


def test_pitch_sign_matches_chin_direction():
    points = project(rotation_x(-15.0))
    pose = estimate_head_pose(points, WIDTH, HEIGHT)
    assert pose.pitch == pytest.approx(15.0, abs=1.0)
    frontal = project(np.eye(3))
    assert points[1][1] < frontal[1][1]


def test_roll_sign_is_clockwise_in_image():
    points = project(rotation_z(12.0))
    pose = estimate_head_pose(points, WIDTH, HEIGHT)
    assert pose.roll == pytest.approx(12.0, abs=1.0)
    assert points[263][1] > points[33][1]


def test_combined_rotation_is_recovered():
    rotation = rotation_y(25.0) @ rotation_x(-10.0) @ rotation_z(5.0)
    expected = angles_from_rotation(rotation)
    pose = estimate_head_pose(project(rotation), WIDTH, HEIGHT)
    assert pose.yaw == pytest.approx(expected.yaw, abs=1.0)
    assert pose.pitch == pytest.approx(expected.pitch, abs=1.0)
    assert pose.roll == pytest.approx(expected.roll, abs=1.0)


def test_too_few_landmarks_is_none():
    assert estimate_head_pose(np.zeros((10, 2)), WIDTH, HEIGHT) is None


def test_head_pose_is_a_plain_dataclass():
    assert HeadPose(1.0, 2.0, 3.0).roll == 3.0
