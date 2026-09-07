#!/usr/bin/env python3
"""Validate oriented COCO annotations without loading YOLOX or pycocotools."""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_values(values, expected_length, field_name):
    if not isinstance(values, (list, tuple)) or len(values) != expected_length:
        raise ValueError(
            "{} must contain exactly {} numbers".format(field_name, expected_length)
        )
    if not all(_is_number(value) and math.isfinite(value) for value in values):
        raise ValueError("{} must contain only finite numbers".format(field_name))
    return np.asarray(values, dtype=np.float32)


def annotation_to_rbbox(annotation):
    """Return ``[cx, cy, w, h, angle]`` and its four corners.

    As in ``OrientedCOCODataset``, an explicit rbbox takes precedence over a
    segmentation when both fields are present.
    """
    if "rbbox" in annotation:
        box = _finite_values(annotation["rbbox"], 5, "rbbox")
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError("rbbox width and height must be positive")
        corners = cv2.boxPoints(
            ((float(box[0]), float(box[1])),
             (float(box[2]), float(box[3])),
             float(box[4]))
        )
        return box, corners

    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, (list, tuple)) and len(segmentation) == 8:
        coordinates = segmentation
    elif (
        isinstance(segmentation, (list, tuple))
        and len(segmentation) == 1
        and isinstance(segmentation[0], (list, tuple))
        and len(segmentation[0]) == 8
    ):
        coordinates = segmentation[0]
    else:
        raise ValueError(
            "orientation requires rbbox [cx,cy,w,h,angle] or one "
            "four-point segmentation"
        )

    corners = _finite_values(coordinates, 8, "segmentation").reshape(4, 2)
    area = abs(float(cv2.contourArea(corners)))
    if area <= 0:
        raise ValueError("segmentation must form a non-degenerate polygon")
    (cx, cy), (width, height), angle = cv2.minAreaRect(corners)
    if width <= 0 or height <= 0:
        raise ValueError("segmentation must have positive width and height")
    box = np.asarray([cx, cy, width, height, angle], dtype=np.float32)
    return box, corners


def _valid_id(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _record_id(record, kind, index, errors):
    value = record.get("id")
    if not _valid_id(value):
        errors.append("{} at index {} has a non-integer id".format(kind, index))
        return None
    return value


def validate_dataset(dataset, images_dir=None):
    """Validate a decoded oriented COCO object and return a list of errors."""
    errors = []
    if not isinstance(dataset, dict):
        return ["JSON root must be an object"]

    images = dataset.get("images")
    categories = dataset.get("categories")
    annotations = dataset.get("annotations")
    for name, value in (
        ("images", images),
        ("categories", categories),
        ("annotations", annotations),
    ):
        if not isinstance(value, list):
            errors.append("'{}' must be a list".format(name))
    if errors:
        return errors

    image_by_id = {}
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            errors.append("image at index {} must be an object".format(index))
            continue
        image_id = _record_id(image, "image", index, errors)
        if image_id is not None:
            if image_id in image_by_id:
                errors.append("duplicate image id {}".format(image_id))
            else:
                image_by_id[image_id] = image

        width, height = image.get("width"), image.get("height")
        if (
            not _is_number(width)
            or not _is_number(height)
            or not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0
            or height <= 0
        ):
            errors.append(
                "image {} width and height must be positive finite numbers".format(
                    image.get("id", index)
                )
            )

        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            errors.append(
                "image {} has a missing or invalid file_name".format(
                    image.get("id", index)
                )
            )
        elif images_dir is not None:
            image_path = os.path.join(images_dir, file_name)
            pixels = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if pixels is None:
                errors.append(
                    "image {} cannot be read: {}".format(
                        image.get("id", index), image_path
                    )
                )
            elif _is_number(width) and _is_number(height):
                actual_height, actual_width = pixels.shape[:2]
                if actual_width != width or actual_height != height:
                    errors.append(
                        "image {} dimensions are {}x{}, JSON declares {}x{}".format(
                            image.get("id", index),
                            actual_width,
                            actual_height,
                            width,
                            height,
                        )
                    )

    category_by_id = {}
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            errors.append("category at index {} must be an object".format(index))
            continue
        category_id = _record_id(category, "category", index, errors)
        if category_id is not None:
            if category_id in category_by_id:
                errors.append("duplicate category id {}".format(category_id))
            else:
                category_by_id[category_id] = category
        if not isinstance(category.get("name"), str) or not category["name"].strip():
            errors.append(
                "category {} has a missing or invalid name".format(
                    category.get("id", index)
                )
            )

    annotation_ids = set()
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            errors.append("annotation at index {} must be an object".format(index))
            continue
        annotation_id = _record_id(annotation, "annotation", index, errors)
        display_id = annotation.get("id", index)
        if annotation_id is not None:
            if annotation_id in annotation_ids:
                errors.append("duplicate annotation id {}".format(annotation_id))
            annotation_ids.add(annotation_id)

        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        image = image_by_id.get(image_id)
        if image is None:
            errors.append(
                "annotation {} references unknown image_id {}".format(
                    display_id, image_id
                )
            )
        if category_id not in category_by_id:
            errors.append(
                "annotation {} references unknown category_id {}".format(
                    display_id, category_id
                )
            )

        try:
            _, corners = annotation_to_rbbox(annotation)
        except (TypeError, ValueError, cv2.error) as error:
            errors.append(
                "annotation {} has invalid orientation: {}".format(display_id, error)
            )
            continue

        if image is None:
            continue
        width, height = image.get("width"), image.get("height")
        if not (_is_number(width) and _is_number(height) and width > 0 and height > 0):
            continue
        epsilon = 1e-4
        xs, ys = corners[:, 0], corners[:, 1]
        if (
            np.any(xs < -epsilon)
            or np.any(ys < -epsilon)
            or np.any(xs > float(width) + epsilon)
            or np.any(ys > float(height) + epsilon)
        ):
            errors.append(
                "annotation {} oriented box is outside image {} bounds "
                "(0..{}, 0..{})".format(display_id, image_id, width, height)
            )

    return errors


def load_and_validate(annotation_path, images_dir=None):
    """Load an annotation file and return ``(dataset, errors)``."""
    try:
        with open(annotation_path, "r", encoding="utf-8") as handle:
            dataset = json.load(handle)
    except (OSError, ValueError) as error:
        return None, ["cannot load {}: {}".format(annotation_path, error)]
    return dataset, validate_dataset(dataset, images_dir=images_dir)


def make_parser():
    parser = argparse.ArgumentParser(
        description="Validate oriented COCO JSON annotations."
    )
    parser.add_argument("annotations", help="path to the COCO annotation JSON")
    parser.add_argument(
        "--images-dir",
        help="optional image root; verifies readability and declared dimensions",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="maximum errors to print (default: 50; 0 prints all)",
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.max_errors < 0:
        print("error: --max-errors must be non-negative", file=sys.stderr)
        return 2
    dataset, errors = load_and_validate(args.annotations, args.images_dir)
    if errors:
        shown = errors if args.max_errors == 0 else errors[:args.max_errors]
        for error in shown:
            print("ERROR: {}".format(error), file=sys.stderr)
        if len(shown) < len(errors):
            print(
                "ERROR: {} additional error(s) not shown".format(
                    len(errors) - len(shown)
                ),
                file=sys.stderr,
            )
        print(
            "Validation failed with {} error(s).".format(len(errors)),
            file=sys.stderr,
        )
        return 1

    print(
        "Validation passed: {} image(s), {} annotation(s), {} category(s).".format(
            len(dataset["images"]),
            len(dataset["annotations"]),
            len(dataset["categories"]),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
