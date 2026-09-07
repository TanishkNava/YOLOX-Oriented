# Production Oriented YOLOX Implementation Guide

This guide describes the implemented core, not proposed pseudocode. The
upstream baseline is Megvii YOLOX commit
[`6ddff4824372906469a7fae2dc3206c7aa4bbaee`](https://github.com/Megvii-BaseDetection/YOLOX/commit/6ddff4824372906469a7fae2dc3206c7aa4bbaee).

## Implemented data path

`OrientedCOCODataset` requires rotated geometry for every non-crowd annotation.
It accepts either:

- `rbbox: [cx, cy, w, h, angle_deg]`; or
- `segmentation` containing exactly eight numbers for four points, either flat
  or as one nested polygon.

Polygons are fitted with `cv2.minAreaRect`. Boxes must be finite and have
positive width and height. Angles are canonicalized modulo 180 to `[0, 180)`;
width and height retain their supplied axes. A missing or invalid oriented
annotation fails dataset construction with `ValueError`. There is no
missing-angle fallback.

The layouts through the pipeline are:

- loaded dataset row: `[cx, cy, w, h, angle_deg, class]`;
- augmented/padded training target: `[class, cx, cy, w, h, angle_deg]`;
- decoded model row: `[cx, cy, w, h, angle_deg, obj_conf, class_probs...]`;
- post-NMS row: `[cx, cy, w, h, angle_deg, obj_conf, class_conf, class_id]`.

Mosaic, affine transforms, resizing, and horizontal flipping preserve oriented
geometry and canonicalize resulting angles.

## Model and training

`YOLOXHeadOriented` keeps the standard four-channel box regressors and adds a
separate one-channel `angle_preds` convolution at each feature level. Decoding
maps that channel through sigmoid to `[0, 180)`.

Regression uses transformed Gaussian KLD over the complete
`[cx, cy, w, h, angle_deg]` box. SimOTA is orientation-aware: candidate
similarity, dynamic-k selection, assignment cost, and classification target
quality use pairwise KLD similarity. Objectness and classification retain the
standard BCE losses. The KLD covariance calculations run in float32 for AMP
stability.

The production experiment is `exps/custom/yolox_s_oriented.py`, which derives
from `yolox.exp.oriented_yolox_base.Exp`. Set its inherited COCO configuration
for your dataset, especially:

```python
self.data_dir = "/path/to/dataset"
self.train_ann = "instances_train2017.json"
self.val_ann = "instances_val2017.json"
self.num_classes = 5
```

Expected layout:

```text
/path/to/dataset/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

Train:

```bash
python tools/train.py \
  -f exps/custom/yolox_s_oriented.py \
  -d 1 -b 32 --fp16
```

Fine-tune from standard YOLOX weights:

```bash
python tools/train.py \
  -f exps/custom/yolox_s_oriented.py \
  -c /path/to/yolox_s.pth \
  -d 1 -b 32 --fp16
```

The fine-tuning loader accepts matching keys and shapes, so the standard
backbone, neck, and existing head parameters load while `angle_preds` remains
newly initialized. `--resume` is different: it strictly restores model and
optimizer state and requires a matching oriented training checkpoint.

## Inference and rotated NMS

Decoded candidates are confidence-filtered using
`obj_conf * max(class_conf)`. Rotated NMS uses `cv2.dnn.NMSBoxesRotated`, with
a deterministic greedy fallback based on exact
`cv2.rotatedRectangleIntersection` IoU. Post-processing supports class-aware
and class-agnostic operation. Evaluation and the demo use class-aware NMS by
default; an experiment can set `class_agnostic_nms = True` when required.

Evaluate:

```bash
python tools/eval.py \
  -f exps/custom/yolox_s_oriented.py \
  -c YOLOX_outputs/yolox_s_oriented/best_ckpt.pth \
  -d 1 -b 32
```

Run image inference:

```bash
python tools/demo.py image \
  -f exps/custom/yolox_s_oriented.py \
  -c YOLOX_outputs/yolox_s_oriented/best_ckpt.pth \
  --path /path/to/image-or-directory \
  --device gpu --conf 0.25 --nms 0.45 \
  --save_result --output-json
```

Use `video` or `webcam` in place of `image` for those sources.
`--output-json` emits one line per image or frame:

```json
{
  "file_name": "frame.jpg",
  "detections": [
    {
      "cx": 100.0,
      "cy": 80.0,
      "w": 40.0,
      "h": 20.0,
      "angle": 35.0,
      "score": 0.91,
      "class_id": 2
    }
  ]
}
```

Coordinates are in the original image scale. Evaluation and demo load model
state strictly, so use an oriented checkpoint with the same architecture and
class count.

## Evaluator

`OrientedCOCOEvaluator` performs one-to-one class matching with exact OpenCV
rotated IoU and reports:

- rotated AP50;
- rotated AP50:95 using thresholds 0.50, 0.55, ..., 0.95 and 101-point AP;
- matched angle MAE at IoU 0.50 with 180-degree periodic error;
- optional per-class rotated AP50:95;
- average forward, NMS, and combined inference time.

This is an oriented evaluator, not `pycocotools.COCOeval` on axis-aligned
boxes.

## Scope

The implemented scope is dataset handling, geometry-safe augmentation, the
oriented head, KLD loss and assignment, rotated post-processing, evaluation,
and demo output. The three enhancement modules described by the associated
paper are not implemented, and this repository does not claim their reported
accuracy gains or deployment behavior.
