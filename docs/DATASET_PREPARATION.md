# Dataset Preparation

Production Oriented YOLOX uses COCO JSON with mandatory oriented geometry for
every non-crowd annotation. It does not infer a missing angle from an
axis-aligned `bbox` and does not default a missing angle to zero.

## Accepted annotation forms

Keep the normal COCO identifiers and metadata. Add one of the following forms.

Direct rotated box:

```json
{
  "id": 1,
  "image_id": 10,
  "category_id": 3,
  "bbox": [80, 60, 40, 20],
  "rbbox": [100, 70, 40, 20, 35],
  "area": 800,
  "iscrowd": 0
}
```

`rbbox` is exactly `[cx, cy, w, h, angle_deg]`.

Four-point polygon:

```json
{
  "id": 2,
  "image_id": 10,
  "category_id": 3,
  "bbox": [80, 60, 40, 20],
  "segmentation": [[82, 64, 115, 55, 120, 76, 87, 85]],
  "area": 800,
  "iscrowd": 0
}
```

The loader accepts either one nested eight-number polygon, as above, or a flat
eight-number list:

```json
{"segmentation": [x1, y1, x2, y2, x3, y3, x4, y4]}
```

No other polygon length or multiple-polygon segmentation is accepted. Polygon
points are converted with `cv2.minAreaRect`.

## Angle and geometry rules

- The internal format is `[cx, cy, w, h, angle_deg]`.
- Angles use OpenCV rotated-rectangle orientation and are canonicalized modulo
  180 to `[0, 180)`.
- Width and height retain their corresponding axes; there is no long-edge
  width convention.
- Width and height must be positive.
- Rotated-box values and polygon coordinates must be finite.
- A polygon must have positive area.
- Missing or malformed oriented geometry raises `ValueError` during dataset
  initialization.

An axis-aligned object still needs an explicit orientation, for example:

```json
{"rbbox": [100, 70, 40, 20, 0]}
```

Do not add `angle` beside a four-number COCO `bbox`; that is not the training
loader's oriented annotation contract.

## Directory structure

The default experiment names expect:

```text
dataset/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   └── ...
└── val2017/
    └── ...
```

Configure `data_dir`, `train_ann`, `val_ann`, and `num_classes` in
`exps/custom/yolox_s_oriented.py` or a derived experiment.

## Validation checklist

Before training, verify:

1. `images`, `annotations`, and `categories` are valid COCO collections.
2. Every non-crowd annotation has one valid `rbbox` or one valid four-point
   `segmentation`.
3. Category IDs in annotations exist in `categories`.
4. Every referenced image exists under the configured split directory.
5. Box sizes and polygon areas are positive and all coordinates are finite.
6. The configured `num_classes` equals the number of trained categories.

Run the standalone validator and inspect rendered samples:

```bash
python tools/validate_oriented_dataset.py \
  /path/to/dataset/annotations/instances_train2017.json \
  --images-dir /path/to/dataset/train2017

python tools/visualize_oriented_dataset.py \
  /path/to/dataset/annotations/instances_train2017.json \
  --images-dir /path/to/dataset/train2017 \
  --output-dir oriented_visualizations
```

Dataset construction is also a strict format check:

```bash
python - <<'PY'
from yolox.data import OrientedCOCODataset

dataset = OrientedCOCODataset(
    data_dir="/path/to/dataset",
    json_file="instances_train2017.json",
    name="train2017",
)
print("validated annotations:", len(dataset))
PY
```

If this fails, correct the source annotation. Do not fill unknown orientations
with zero unless zero degrees is the verified ground truth.
