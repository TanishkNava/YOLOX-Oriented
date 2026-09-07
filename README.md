# Production Oriented YOLOX

This repository implements the production core for oriented object detection on
top of [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), pinned to
commit [`6ddff4824372906469a7fae2dc3206c7aa4bbaee`](https://github.com/Megvii-BaseDetection/YOLOX/commit/6ddff4824372906469a7fae2dc3206c7aa4bbaee).

Implemented functionality:

- oriented COCO loading with mandatory rotated geometry;
- canonical boxes `[cx, cy, w, h, angle_deg]`, with angles in `[0, 180)`;
- an angle branch added without changing upstream regression parameter shapes;
- transformed Gaussian KLD regression and orientation-aware SimOTA assignment;
- exact rotated IoU and class-aware or class-agnostic NMS using OpenCV;
- rotated AP50, rotated AP50:95, matched angle MAE, per-class AP, and timing;
- oriented image, video, and webcam demo output.

The scope is the production detection core. It does not implement the three
paper enhancement modules; results that depend on those modules are outside
this repository's claims.

## Installation

```bash
pip install -e .
```

## Dataset contract

Use normal COCO `images`, `annotations`, and `categories`. Every non-crowd
annotation must contain exactly one of:

```json
{"rbbox": [cx, cy, w, h, angle_deg]}
```

```json
{"segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]]}
```

The flat eight-number segmentation form is also accepted. Polygon input is
converted with `cv2.minAreaRect`. Width and height must be positive, all values
must be finite, and angles are canonicalized modulo 180 to `[0, 180)`. Missing
or malformed oriented geometry raises `ValueError`; it is never treated as
zero degrees. See [docs/DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md).

## Layouts

- Dataset raw row: `[cx, cy, w, h, angle_deg, class]`
- Training target: `[class, cx, cy, w, h, angle_deg]`
- Decoded head output: `[cx, cy, w, h, angle_deg, obj_conf, class_probs...]`
- Post-NMS detection: `[cx, cy, w, h, angle_deg, obj_conf, class_conf, class_id]`
- Public `format_oriented_detections` row: `[cx, cy, w, h, angle_deg, score, class_id]`
- `--output-json` detection: `{cx, cy, w, h, angle, score, class_id}`

## Train, evaluate, and run the demo

Configure dataset paths, annotation names, and `num_classes` in
`exps/custom/yolox_s_oriented.py` or a derived experiment.

```bash
python tools/train.py \
  -f exps/custom/yolox_s_oriented.py \
  -d 1 -b 32 --fp16
```

```bash
python tools/eval.py \
  -f exps/custom/yolox_s_oriented.py \
  -c YOLOX_outputs/yolox_s_oriented/best_ckpt.pth \
  -d 1 -b 32
```

```bash
python tools/demo.py image \
  -f exps/custom/yolox_s_oriented.py \
  -c YOLOX_outputs/yolox_s_oriented/best_ckpt.pth \
  --path /path/to/image.jpg \
  --device gpu --conf 0.25 --nms 0.45 \
  --save_result --output-json
```

`--output-json` writes one JSON object per image or frame to stdout. Coordinates
are restored to the original image scale.

## Pretrained weights

For fine-tuning, `tools/train.py -c <standard-yolox-checkpoint>` loads matching
keys and leaves the new `angle_preds` layers randomly initialized. This is the
expected compatibility path. Resume, evaluation, and demo use strict model
loading and therefore require an oriented checkpoint with matching classes and
architecture; a standard YOLOX checkpoint cannot be used directly there.

## Evaluation

Evaluation matches detections with exact OpenCV rotated IoU. It reports rotated
AP50, rotated AP50:95 over thresholds 0.50 through 0.95, matched angle MAE at
IoU 0.50, optional per-class rotated AP50:95, and forward/NMS timing.

## License

Apache License 2.0, following upstream YOLOX.
