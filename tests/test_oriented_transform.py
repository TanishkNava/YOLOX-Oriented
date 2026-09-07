import numpy as np

from yolox.data.data_augment import (
    OrientedTrainTransform,
    apply_affine_to_obbs,
)
from yolox.utils.oriented_boxes import pairwise_oriented_iou


def test_oriented_transform_outputs_class_first_and_scales():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    targets = np.array([[10, 8, 12, 4, 30, 3]], dtype=np.float32)
    transform = OrientedTrainTransform(max_labels=2, flip_prob=0, hsv_prob=0)
    _, output = transform(image, targets, (40, 40))
    np.testing.assert_allclose(output[0], [3, 10, 8, 12, 4, 30], atol=1e-5)
    np.testing.assert_array_equal(output[1], np.zeros(6, dtype=np.float32))


def test_oriented_transform_mirror_refits_equivalent_box():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    targets = np.array([[10, 8, 12, 4, 30, 2]], dtype=np.float32)
    transform = OrientedTrainTransform(max_labels=1, flip_prob=1, hsv_prob=0)
    _, output = transform(image, targets, (20, 40))
    expected = np.array([[30, 8, 12, 4, 150]], dtype=np.float32)
    assert output[0, 0] == 2
    assert pairwise_oriented_iou(output[:, 1:6], expected)[0, 0] > 0.999


def test_affine_translation_preserves_angle_and_class():
    targets = np.array([[10, 8, 12, 4, 30, 7]], dtype=np.float32)
    matrix = np.array([[1, 0, 5], [0, 1, -2]], dtype=np.float32)
    output = apply_affine_to_obbs(targets, (40, 20), matrix, 1.0)
    expected = np.array([[15, 6, 12, 4, 30]], dtype=np.float32)
    assert output[0, 5] == 7
    assert pairwise_oriented_iou(output[:, :5], expected)[0, 0] > 0.999
