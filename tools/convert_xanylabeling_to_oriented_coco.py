#!/usr/bin/env python3
"""Convert X-AnyLabeling rotation JSON files to oriented COCO datasets."""

import argparse
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLASSES = (
    "curtain_closed",
    "curtain_open",
    "spray_machine",
    "person",
    "person_with_stick",
    "feed_machine",
)
FRAME_PATTERN = re.compile(r"frame_(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="X-AnyLabeling image/JSON directory")
    parser.add_argument("output", type=Path, help="output COCO dataset directory")
    parser.add_argument(
        "--block-size",
        type=int,
        default=300,
        help="number of source frame numbers per temporal block (default: 300)",
    )
    parser.add_argument(
        "--val-period",
        type=int,
        default=5,
        help="use one out of this many temporal blocks for validation (default: 5)",
    )
    parser.add_argument(
        "--val-block-index",
        type=int,
        default=4,
        help="validation block index within each period (default: 4)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy images instead of creating absolute symbolic links",
    )
    parser.add_argument(
        "--fit-outside-boxes",
        action="store_true",
        help="minimally move/scale rotated boxes that cross image boundaries",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output directory",
    )
    return parser.parse_args()


def polygon_area(points):
    return abs(
        sum(
            points[index][0] * points[(index + 1) % 4][1]
            - points[(index + 1) % 4][0] * points[index][1]
            for index in range(4)
        )
    ) / 2.0


def validate_points(points, annotation_path, shape_index):
    valid = (
        isinstance(points, list)
        and len(points) == 4
        and all(
            isinstance(point, list)
            and len(point) == 2
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in point
            )
            for point in points
        )
    )
    if not valid:
        raise ValueError(
            f"{annotation_path.name}: shape {shape_index} has invalid four-point geometry"
        )
    area = polygon_area(points)
    if area <= 0:
        raise ValueError(
            f"{annotation_path.name}: shape {shape_index} has zero polygon area"
        )
    return area


def fitted_rbbox(points, image_width, image_height):
    """Return an OpenCV rbbox fitted inside the image and whether it changed."""
    point_array = np.asarray(points, dtype=np.float32)
    outside = bool(
        np.any(point_array[:, 0] < 0)
        or np.any(point_array[:, 0] > image_width)
        or np.any(point_array[:, 1] < 0)
        or np.any(point_array[:, 1] > image_height)
    )
    (cx, cy), (width, height), angle = cv2.minAreaRect(point_array)
    if not outside:
        return [cx, cy, width, height, angle], False

    corners = cv2.boxPoints(((cx, cy), (width, height), angle))
    extent_x = float(np.max(np.abs(corners[:, 0] - cx)))
    extent_y = float(np.max(np.abs(corners[:, 1] - cy)))
    margin = 1e-2
    scale = min(
        1.0,
        max((image_width - 2 * margin) / (2 * extent_x), 0.0)
        if extent_x > 0
        else 1.0,
        max((image_height - 2 * margin) / (2 * extent_y), 0.0)
        if extent_y > 0
        else 1.0,
    )
    width *= scale
    height *= scale
    corners = cv2.boxPoints(((cx, cy), (width, height), angle))
    extent_x = float(np.max(np.abs(corners[:, 0] - cx))) + margin
    extent_y = float(np.max(np.abs(corners[:, 1] - cy))) + margin
    cx = min(max(float(cx), extent_x), image_width - extent_x)
    cy = min(max(float(cy), extent_y), image_height - extent_y)
    return [cx, cy, width, height, angle], True


def frame_number(path):
    match = FRAME_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"{path.name}: filename does not contain frame_<number>")
    return int(match.group(1))


def discover_annotations(source):
    annotations = [
        path
        for path in source.glob("*.json")
        if path.is_file() and FRAME_PATTERN.search(path.stem)
    ]
    if not annotations:
        raise ValueError(f"no frame annotation JSON files found in {source}")
    return sorted(annotations, key=frame_number)


def split_name(frame, block_size, val_period, val_block_index):
    block_index = frame // block_size
    return (
        "val2017"
        if block_index % val_period == val_block_index
        else "train2017"
    )


def create_image(source, destination, copy_images):
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def convert(args):
    if args.block_size <= 0 or args.val_period <= 1:
        raise ValueError("--block-size must be positive and --val-period must exceed 1")
    if not 0 <= args.val_block_index < args.val_period:
        raise ValueError("--val-block-index must be in [0, val-period)")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"{args.output} already exists; pass --force to replace it")
        shutil.rmtree(args.output)

    annotation_paths = discover_annotations(args.source)
    category_ids = {
        class_name: index + 1 for index, class_name in enumerate(DEFAULT_CLASSES)
    }
    categories = [
        {"id": category_id, "name": class_name, "supercategory": "object"}
        for class_name, category_id in category_ids.items()
    ]
    datasets = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train2017", "val2017")
    }
    counts = {split: Counter() for split in datasets}
    fitted_counts = Counter()
    annotation_id = 1

    for image_id, annotation_path in enumerate(annotation_paths, start=1):
        source_data = json.loads(annotation_path.read_text(encoding="utf-8"))
        frame = frame_number(annotation_path)
        split = split_name(
            frame, args.block_size, args.val_period, args.val_block_index
        )
        image_name = source_data.get("imagePath")
        image_path = args.source / image_name if isinstance(image_name, str) else None
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"{annotation_path.name}: missing image {image_name!r}")

        width = source_data.get("imageWidth")
        height = source_data.get("imageHeight")
        if not (
            isinstance(width, int)
            and isinstance(height, int)
            and width > 0
            and height > 0
        ):
            raise ValueError(f"{annotation_path.name}: invalid image dimensions")
        datasets[split]["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "frame_number": frame,
            }
        )

        for shape_index, shape in enumerate(source_data.get("shapes", [])):
            label = shape.get("label")
            if label not in category_ids:
                raise ValueError(f"{annotation_path.name}: unknown label {label!r}")
            if shape.get("shape_type") != "rotation":
                raise ValueError(
                    f"{annotation_path.name}: shape {shape_index} is not a rotation"
                )
            points = shape.get("points")
            area = validate_points(points, annotation_path, shape_index)
            rbbox, was_fitted = fitted_rbbox(points, width, height)
            if was_fitted and not args.fit_outside_boxes:
                raise ValueError(
                    f"{annotation_path.name}: shape {shape_index} crosses image bounds; "
                    "pass --fit-outside-boxes to correct it in the converted dataset"
                )
            if was_fitted:
                fitted_counts[label] += 1
            corners = cv2.boxPoints(
                ((rbbox[0], rbbox[1]), (rbbox[2], rbbox[3]), rbbox[4])
            )
            xs = corners[:, 0]
            ys = corners[:, 1]
            area = float(rbbox[2] * rbbox[3])
            datasets[split]["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[label],
                    "bbox": [
                        float(min(xs)),
                        float(min(ys)),
                        float(max(xs) - min(xs)),
                        float(max(ys) - min(ys)),
                    ],
                    "rbbox": [float(value) for value in rbbox],
                    "segmentation": [[value for point in points for value in point]],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            counts[split][label] += 1
            annotation_id += 1

    for split, dataset in datasets.items():
        if not dataset["images"]:
            raise ValueError(f"temporal split produced no {split} images")
        missing_classes = set(DEFAULT_CLASSES) - set(counts[split])
        if missing_classes:
            raise ValueError(
                f"{split} contains no examples for: {', '.join(sorted(missing_classes))}"
            )

    (args.output / "annotations").mkdir(parents=True)
    for split, dataset in datasets.items():
        image_dir = args.output / split
        image_dir.mkdir()
        image_names = {image["file_name"] for image in dataset["images"]}
        for image_name in sorted(image_names):
            create_image(
                args.source / image_name,
                image_dir / image_name,
                args.copy_images,
            )
        annotation_file = args.output / "annotations" / f"instances_{split}.json"
        annotation_file.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for split, dataset in datasets.items():
        print(
            f"{split}: {len(dataset['images'])} images, "
            f"{len(dataset['annotations'])} annotations, {dict(counts[split])}"
        )
    print(f"boxes fitted inside image bounds: {sum(fitted_counts.values())} {dict(fitted_counts)}")
    print(f"wrote oriented COCO dataset to {args.output}")


def main():
    args = parse_args()
    try:
        convert(args)
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
