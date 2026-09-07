import numpy as np

from yolox.utils.oriented_boxes import (
    canonicalize_obb,
    obb_to_polygon,
    pairwise_oriented_iou,
    polygon_to_obb,
)


def test_canonicalize_angle_preserves_width_height_axes():
    box = canonicalize_obb(np.array([10, 20, 4, 8, -10], dtype=np.float32))
    np.testing.assert_allclose(box, [10, 20, 4, 8, 170], atol=1e-5)


def test_polygon_round_trip_preserves_geometry():
    box = np.array([23, 31, 15, 7, 137], dtype=np.float32)
    fitted = polygon_to_obb(obb_to_polygon(box))
    assert pairwise_oriented_iou(box[None], fitted[None])[0, 0] > 0.9999


def test_pairwise_oriented_iou_exact_cases():
    boxes = np.array([[10, 10, 8, 4, 30], [100, 100, 8, 4, 30]], dtype=np.float32)
    result = pairwise_oriented_iou(boxes, boxes[:1])
    np.testing.assert_allclose(result[:, 0], [1.0, 0.0], atol=1e-6)
