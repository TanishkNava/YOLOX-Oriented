#!/usr/bin/env python3
"""Draw oriented COCO annotations on a deterministic sample of images."""

import argparse
import os
import random
import sys

import cv2
import numpy as np

try:
    from validate_oriented_dataset import annotation_to_rbbox, load_and_validate
except ImportError:
    try:
        from tools.validate_oriented_dataset import (
            annotation_to_rbbox,
            load_and_validate,
        )
    except ImportError:
        from yolox.tools.validate_oriented_dataset import (
            annotation_to_rbbox,
            load_and_validate,
        )


def _category_color(category_id):
    """Return a stable, high-contrast BGR color for a category id."""
    hue = int((category_id * 47) % 180)
    hsv = np.uint8([[[hue, 210, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_annotations(image, annotations, category_names, thickness=2):
    """Draw oriented boxes and class labels in-place and return the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, thickness * 0.25)
    for annotation in annotations:
        box, _ = annotation_to_rbbox(annotation)
        corners = cv2.boxPoints(
            ((float(box[0]), float(box[1])),
             (float(box[2]), float(box[3])),
             float(box[4]))
        )
        points = np.rint(corners).astype(np.int32).reshape((-1, 1, 2))
        category_id = annotation["category_id"]
        color = _category_color(category_id)
        cv2.polylines(
            image, [points], isClosed=True, color=color,
            thickness=thickness, lineType=cv2.LINE_AA
        )

        label = category_names[category_id]
        annotation_id = annotation.get("id")
        if annotation_id is not None:
            label = "{} #{}".format(label, annotation_id)
        anchor = np.rint(corners[np.argmin(corners[:, 1])]).astype(int)
        text_size, baseline = cv2.getTextSize(
            label, font, font_scale, max(1, thickness)
        )
        x = max(0, min(int(anchor[0]), image.shape[1] - text_size[0] - 1))
        y = max(text_size[1] + baseline, min(int(anchor[1]), image.shape[0] - 1))
        cv2.rectangle(
            image,
            (x, y - text_size[1] - baseline),
            (x + text_size[0], y + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            max(1, thickness),
            cv2.LINE_AA,
        )
    return image


def visualize_dataset(
    dataset, images_dir, output_dir, num_images=10, seed=0, thickness=2
):
    """Write annotated sampled images and return their output paths."""
    images = sorted(dataset["images"], key=lambda image: image["id"])
    sample_count = min(num_images, len(images))
    selected = random.Random(seed).sample(images, sample_count)

    annotations_by_image = {}
    for annotation in dataset["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
    for annotations in annotations_by_image.values():
        annotations.sort(key=lambda annotation: annotation["id"])

    category_names = {
        category["id"]: category["name"] for category in dataset["categories"]
    }
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for image_info in selected:
        source_path = os.path.join(images_dir, image_info["file_name"])
        image = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError("cannot read image: {}".format(source_path))
        draw_annotations(
            image,
            annotations_by_image.get(image_info["id"], []),
            category_names,
            thickness=thickness,
        )
        base_name = os.path.basename(image_info["file_name"])
        output_name = "{:012d}_{}".format(image_info["id"], base_name)
        output_path = os.path.join(output_dir, output_name)
        if not cv2.imwrite(output_path, image):
            raise OSError("cannot write image: {}".format(output_path))
        output_paths.append(output_path)
    return output_paths


def make_parser():
    parser = argparse.ArgumentParser(
        description="Visualize oriented COCO annotations on sampled images."
    )
    parser.add_argument("annotations", help="path to the COCO annotation JSON")
    parser.add_argument(
        "--images-dir", required=True, help="root directory for image file_name paths"
    )
    parser.add_argument(
        "--output-dir",
        default="oriented_visualizations",
        help="directory for rendered images (default: oriented_visualizations)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="number of images to sample (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="deterministic sampling seed (default: 0)"
    )
    parser.add_argument(
        "--thickness", type=int, default=2, help="box/text thickness (default: 2)"
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.num_images <= 0:
        print("error: --num-images must be positive", file=sys.stderr)
        return 2
    if args.thickness <= 0:
        print("error: --thickness must be positive", file=sys.stderr)
        return 2

    dataset, errors = load_and_validate(args.annotations, args.images_dir)
    if errors:
        for error in errors[:50]:
            print("ERROR: {}".format(error), file=sys.stderr)
        if len(errors) > 50:
            print(
                "ERROR: {} additional error(s) not shown".format(len(errors) - 50),
                file=sys.stderr,
            )
        print(
            "Visualization aborted: validation found {} error(s).".format(len(errors)),
            file=sys.stderr,
        )
        return 1
    if not dataset["images"]:
        print("Visualization aborted: dataset contains no images.", file=sys.stderr)
        return 1

    try:
        output_paths = visualize_dataset(
            dataset,
            args.images_dir,
            args.output_dir,
            num_images=args.num_images,
            seed=args.seed,
            thickness=args.thickness,
        )
    except (OSError, ValueError, cv2.error) as error:
        print("Visualization failed: {}".format(error), file=sys.stderr)
        return 1

    print(
        "Wrote {} visualization(s) to {}.".format(
            len(output_paths), os.path.abspath(args.output_dir)
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
