import json

import cv2
import numpy as np
import pytest

from tools.validate_oriented_dataset import validate_dataset
from yolox.data import OrientedCOCODataset


def _dataset(annotation):
    return {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 64, "height": 48}],
        "categories": [{"id": 4, "name": "object"}],
        "annotations": [{
            "id": 2,
            "image_id": 1,
            "category_id": 4,
            "bbox": [20, 15, 20, 10],
            "area": 200,
            "iscrowd": 0,
            **annotation,
        }],
    }


def _write_fixture(tmp_path, annotation):
    root = tmp_path / "dataset"
    (root / "annotations").mkdir(parents=True)
    (root / "train2017").mkdir()
    cv2.imwrite(str(root / "train2017" / "sample.jpg"), np.zeros((48, 64, 3), np.uint8))
    with (root / "annotations" / "instances_train2017.json").open("w") as stream:
        json.dump(_dataset(annotation), stream)
    return root


@pytest.mark.parametrize(
    "annotation",
    [
        {"rbbox": [30, 20, 20, 10, 25]},
        {"segmentation": [[20, 15, 40, 15, 40, 25, 20, 25]]},
    ],
)
def test_oriented_dataset_accepts_rbbox_and_polygon(tmp_path, annotation):
    root = _write_fixture(tmp_path, annotation)
    dataset = OrientedCOCODataset(
        data_dir=str(root),
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(48, 64),
    )
    labels = dataset.load_anno(0)
    assert labels.shape == (1, 6)
    assert labels[0, 5] == 0
    assert 0 <= labels[0, 4] < 180
    assert validate_dataset(_dataset(annotation)) == []


def test_oriented_dataset_rejects_missing_orientation(tmp_path):
    root = _write_fixture(tmp_path, {})
    with pytest.raises(ValueError, match="oriented annotations require"):
        OrientedCOCODataset(
            data_dir=str(root),
            json_file="instances_train2017.json",
            name="train2017",
        )
