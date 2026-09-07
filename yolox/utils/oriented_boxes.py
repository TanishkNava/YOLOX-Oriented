#!/usr/bin/env python3
"""Geometry helpers for oriented bounding boxes.

The public representation used here is ``(cx, cy, w, h, angle_degrees)``.
Angles use OpenCV's rotated-rectangle convention and are canonicalized to
``[0, 180)``.  Width and height retain their supplied axes; no implicit
long-edge width convention is imposed.
"""

import cv2
import numpy as np


def canonicalize_obb(box, eps=1e-7):
    """Return one canonical OBB, preserving the input floating dtype if possible."""
    array = np.asarray(box)
    if array.shape != (5,):
        raise ValueError("an OBB must have shape (5,)")
    if not np.all(np.isfinite(array)):
        raise ValueError("an OBB must contain only finite values")

    out = array.astype(np.float64, copy=True)
    if out[2] <= eps or out[3] <= eps:
        raise ValueError("OBB width and height must be positive")
    out[4] %= 180.0
    return out.astype(array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32)


def canonicalize_obbs(boxes, eps=1e-7):
    """Vectorized :func:`canonicalize_obb` for an ``(N, 5)`` array."""
    array = np.asarray(boxes)
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError("OBBs must have shape (N, 5)")
    if len(array) == 0:
        return array.astype(
            array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32,
            copy=True,
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("OBBs must contain only finite values")
    if np.any(array[:, 2:4] <= eps):
        raise ValueError("OBB widths and heights must be positive")

    out = array.astype(np.float64, copy=True)
    out[:, 4] %= 180.0
    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32
    return out.astype(dtype)


def obb_to_polygon(box):
    """Convert one canonical-format OBB to four OpenCV corner points."""
    cx, cy, width, height, angle = canonicalize_obb(box).astype(np.float32)
    return cv2.boxPoints(((float(cx), float(cy)), (float(width), float(height)), float(angle)))


def obbs_to_polygons(boxes):
    """Convert ``(N, 5)`` OBBs to an ``(N, 4, 2)`` corner array."""
    boxes = canonicalize_obbs(boxes)
    if len(boxes) == 0:
        return np.empty((0, 4, 2), dtype=np.float32)
    return np.stack([obb_to_polygon(box) for box in boxes]).astype(np.float32)


def polygon_to_obb(polygon):
    """Fit an exact minimum-area OBB to exactly four two-dimensional points."""
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape == (8,):
        points = points.reshape(4, 2)
    if points.shape != (4, 2):
        raise ValueError("an oriented polygon must contain exactly four 2D points")
    if not np.all(np.isfinite(points)):
        raise ValueError("polygon points must be finite")
    if abs(float(cv2.contourArea(points))) <= 1e-7:
        raise ValueError("polygon must have positive area")
    (cx, cy), (width, height), angle = cv2.minAreaRect(points)
    return canonicalize_obb(np.array([cx, cy, width, height, angle], dtype=np.float32))


def polygons_to_obbs(polygons):
    """Fit OBBs to an ``(N, 4, 2)`` polygon array."""
    polygons = np.asarray(polygons)
    if polygons.ndim != 3 or polygons.shape[1:] != (4, 2):
        raise ValueError("polygons must have shape (N, 4, 2)")
    if len(polygons) == 0:
        return np.empty((0, 5), dtype=np.float32)
    return np.stack([polygon_to_obb(polygon) for polygon in polygons])


def _opencv_rect(box):
    cx, cy, width, height, angle = canonicalize_obb(box)
    return ((float(cx), float(cy)), (float(width), float(height)), float(angle))


def oriented_box_iou(box1, box2):
    """Compute exact geometric IoU for two OBBs with OpenCV."""
    box1 = canonicalize_obb(box1)
    box2 = canonicalize_obb(box2)
    area1 = float(box1[2] * box1[3])
    area2 = float(box2[2] * box2[3])
    _, intersection = cv2.rotatedRectangleIntersection(
        _opencv_rect(box1), _opencv_rect(box2)
    )
    intersection_area = (
        0.0
        if intersection is None
        else abs(float(cv2.contourArea(cv2.convexHull(intersection))))
    )
    union = area1 + area2 - intersection_area
    return 0.0 if union <= 0.0 else float(np.clip(intersection_area / union, 0.0, 1.0))


def pairwise_oriented_iou(boxes1, boxes2):
    """Compute exact pairwise IoU for two OBB arrays, returning shape ``(N, M)``."""
    boxes1 = canonicalize_obbs(boxes1)
    boxes2 = canonicalize_obbs(boxes2)
    result = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            result[i, j] = oriented_box_iou(box1, box2)
    return result


# Concise aliases commonly used by downstream oriented-detection code.
obb2poly = obb_to_polygon
poly2obb = polygon_to_obb
obb_iou = oriented_box_iou
pairwise_obb_iou = pairwise_oriented_iou
