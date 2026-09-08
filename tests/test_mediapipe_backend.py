import cv2
import numpy as np
import pytest

from app.analyzers import REGISTRY
from app.analyzers.context import AnalysisContext
from app.analyzers.pose import estimate_head_pose
from app.imageio import read_bgr

pytestmark = pytest.mark.mediapipe

PORTRAITS = ("portrait_frontal.jpg", "portrait_second.jpg", "portrait_turned.jpg")


def measure(ctx, name):
    return REGISTRY[name].fn(ctx)


@pytest.fixture(scope="module")
def portraits(fixtures_dir, backend):
    return {name: AnalysisContext(read_bgr(fixtures_dir / name), backend) for name in PORTRAITS}


@pytest.mark.parametrize("name", PORTRAITS)
def test_portrait_has_one_face_with_full_mesh(portraits, name):
    ctx = portraits[name]
    assert len(ctx.faces) == 1
    face = ctx.primary_face
    assert face.landmarks_norm.shape == (478, 3)
    inside = (face.landmarks_norm[:, :2] >= 0) & (face.landmarks_norm[:, :2] <= 1)
    assert inside.mean() > 0.9
    assert 0.05 < measure(ctx, "geometry.face_area_ratio").values["face_area_ratio"] < 0.7


def test_landscape_has_no_face(fixtures_dir, backend):
    ctx = AnalysisContext(read_bgr(fixtures_dir / "landscape.jpg"), backend)
    assert ctx.faces == ()
    assert measure(ctx, "face.count").values == {"count": 0}
    assert not measure(ctx, "pose.yaw_pitch").applicable


def test_collage_has_two_faces(portraits, backend):
    left = portraits["portrait_frontal.jpg"].bgr
    right = portraits["portrait_second.jpg"].bgr
    height = min(left.shape[0], right.shape[0])
    collage = np.hstack([left[:height], right[:height]])
    assert len(AnalysisContext(collage, backend).faces) == 2


def test_person_mask_is_high_on_face_and_low_in_corners(portraits):
    for ctx in portraits.values():
        mask = ctx.person_confidence
        assert mask.shape == (ctx.height, ctx.width)
        assert mask.dtype == np.float32
        box = ctx.primary_face.bbox
        face_region = mask[int(box.y0) : int(box.y1), int(box.x0) : int(box.x1)]
        assert face_region.mean() > 0.8
        corners = np.mean([mask[:16, :16].mean(), mask[:16, -16:].mean()])
        assert corners < 0.2


def test_frontal_portrait_pose_and_turned_portrait_yaw(portraits):
    frontal = measure(portraits["portrait_frontal.jpg"], "pose.yaw_pitch").values
    assert abs(frontal["yaw"]) < 8
    assert abs(frontal["pitch"]) < 12
    assert abs(frontal["roll"]) < 8
    turned = measure(portraits["portrait_turned.jpg"], "pose.yaw_pitch").values
    assert turned["yaw"] > 15


@pytest.mark.parametrize("name", PORTRAITS)
def test_flip_negates_yaw(portraits, backend, name):
    ctx = portraits[name]
    flipped = AnalysisContext(np.ascontiguousarray(cv2.flip(ctx.bgr, 1)), backend)
    yaw = measure(ctx, "pose.yaw_pitch").values["yaw"]
    flipped_yaw = measure(flipped, "pose.yaw_pitch").values["yaw"]
    assert flipped_yaw == pytest.approx(-yaw, abs=4.0)


def test_rotation_shifts_roll(portraits, backend):
    ctx = portraits["portrait_second.jpg"]
    matrix = cv2.getRotationMatrix2D((ctx.width / 2, ctx.height / 2), 15.0, 1.0)
    rotated = cv2.warpAffine(ctx.bgr, matrix, (ctx.width, ctx.height))
    base = estimate_head_pose(ctx.primary_face.landmarks_px, ctx.width, ctx.height)
    turned = AnalysisContext(rotated, backend)
    pose = estimate_head_pose(turned.primary_face.landmarks_px, ctx.width, ctx.height)
    assert pose.roll - base.roll == pytest.approx(-15.0, abs=3.0)


def test_eyes_are_open_on_portraits(portraits):
    for ctx in portraits.values():
        assert measure(ctx, "eyes.min_ear").values["min_ear"] > 0.18


def test_blur_lowers_face_sharpness(portraits, backend):
    ctx = portraits["portrait_frontal.jpg"]
    blurred = AnalysisContext(cv2.GaussianBlur(ctx.bgr, (0, 0), 3.0), backend)
    sharp = measure(ctx, "sharpness.face_laplacian_var").values["var"]
    soft = measure(blurred, "sharpness.face_laplacian_var").values["var"]
    assert soft < sharp / 10


def test_checkerboard_background_raises_cv(portraits, backend):
    ctx = portraits["portrait_second.jpg"]
    tiles = (np.indices((ctx.height, ctx.width)).sum(axis=0) // 24) % 2
    texture = np.repeat((tiles * 180 + 40).astype(np.uint8)[:, :, None], 3, axis=2)
    person = ctx.person_confidence[:, :, None] > 0.5
    busy = np.where(person, ctx.bgr, texture)
    plain = measure(ctx, "background.cv").values["cv"]
    noisy = measure(AnalysisContext(busy, backend), "background.cv").values["cv"]
    assert noisy > plain
    assert noisy > 0.4
