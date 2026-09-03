# Oriented YOLOX - Rotated Object Detection

This is an implementation of **Oriented YOLOX** for rotated object detection, extending the original YOLOX architecture to support arbitrary object orientations. Built on top of [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

## Features

✅ **Oriented Bounding Boxes**: Predicts (x, y, w, h, θ) for rotated objects  
✅ **COCO Format Support**: Works with standard COCO JSON + images  
✅ **Custom Classes**: Optimized for Bird Shed detection (Vehicles, Persons, Sticks, Spray Machines, Feed Machines)  
✅ **Rotated NMS**: Handles overlapping rotated boxes correctly  
✅ **Backward Compatible**: Still detects horizontal objects (θ = 0°)  
✅ **Multiple Model Sizes**: Support for nano, tiny, small, medium, large, and x variants  

## Key Changes from Standard YOLOX

1. **Head Output**: Changed from `[x, y, w, h, obj_conf, class_probs]` to `[x, y, w, h, θ, obj_conf, class_probs]`
2. **Loss Function**: Added angle regression loss (smooth L1)
3. **NMS**: Implemented Rotated NMS using oriented box IoU
4. **Data Loading**: COCO dataset loader with angle extraction
5. **Inference**: Post-processing with angle handling

## Installation

```bash
git clone https://github.com/TanishkNava/YOLOX-Oriented.git
cd YOLOX-Oriented
pip install -e .
```

## Dataset Preparation

Your COCO JSON format should include angle in bbox:

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, w, h],
      "angle": 45.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "vehicle"},
    {"id": 2, "name": "person"},
    {"id": 3, "name": "stick"},
    {"id": 4, "name": "spray_machine"},
    {"id": 5, "name": "feed_machine"}
  ]
}
```

**Note**: If your COCO doesn't have angles yet, see [scripts/add_angles_to_coco.py](scripts/add_angles_to_coco.py) to add them.

## Quick Start

### 1. Prepare Dataset

```bash
export YOLOX_DATADIR=/path/to/your/datasets
ln -s /path/to/your/data ./datasets/BirdShed
```

### 2. Create Config

```bash
cp exps/default/yolox_s.py exps/custom/yolox_s_oriented_birdShed.py
# Edit the config (see exps/custom/README.md)
```

### 3. Train

```bash
python -m yolox.tools.train \
  -f exps/custom/yolox_s_oriented_birdShed.py \
  -d 1 -b 32 --fp16 -o
```

### 4. Evaluate

```bash
python -m yolox.tools.eval \
  -f exps/custom/yolox_s_oriented_birdShed.py \
  -c checkpoints/yolox_s_oriented.pth \
  -b 32 -d 1
```

### 5. Inference

```bash
python tools/demo.py image \
  -f exps/custom/yolox_s_oriented_birdShed.py \
  -c checkpoints/yolox_s_oriented.pth \
  --path /path/to/image.jpg \
  --conf 0.25 --save_result
```

## Will It Detect Horizontal Objects?

**YES!** ✅ Oriented YOLOX handles both horizontal AND rotated objects:

- **Horizontal objects** → Angle = 0° (or 180°)
- **Rotated objects** → Angle = θ (learned from data)
- The model naturally learns to predict 0° for axis-aligned boxes
- **No performance loss** on horizontal objects; often **better** since model has more flexibility

## Architecture Overview

```
Input Image
    ↓
  Backbone (CSPDarknet)
    ↓
  Neck (FPN)
    ↓
  Head (Decoupled)
    ├─→ Regression Branch: [x, y, w, h, θ]  ← NEW: Angle prediction
    ├─→ Objectness Branch: [obj_conf]
    └─→ Classification Branch: [class_probs]
    ↓
Decoder (Preprocessing)
    ↓
Rotated NMS
    ↓
Detections [(x, y, w, h, θ, conf, class_id), ...]
```

## File Structure

```
YOLOX-Oriented/
├── yolox/
│   ├── models/
│   │   ├── yolox.py               # Base model
│   │   ├── yolo_head_oriented.py  # Modified head for angle
│   │   └── darknet.py
│   ├── utils/
│   │   ├── rotated_nms.py         # Rotated NMS implementation
│   │   └── boxes.py               # Box operations (oriented)
│   ├── data/
│   │   ├── datasets/
│   │   │   └── coco_oriented.py   # COCO loader with angles
│   │   └── data_augment.py
│   └── tools/
│       ├── train.py
│       ├── eval.py
│       └── demo.py
├── exps/
│   ├── default/
│   │   └── yolox_s.py            # Base config
│   └── custom/
│       └── yolox_s_oriented_birdShed.py  # Your custom config
├── scripts/
│   ├── add_angles_to_coco.py      # Convert normal COCO to oriented COCO
│   ├── visualize_oriented_boxes.py # Visualize rotated bboxes
│   └── convert_dataset.py          # Dataset format conversion
└── README.md
```

## Key Implementation Details

### 1. Angle Representation
- **Range**: [0°, 180°) (symmetric representation)
- **Why**: Avoids 0° = 360° ambiguity; more stable for training
- **Loss**: Smooth L1 Loss on angle differences

### 2. Rotated NMS
- Uses **Oriented IoU** (IoU of rotated boxes)
- Handles arbitrary orientations correctly
- Applied after confidence filtering and NMS

### 3. Head Modification
```python
# Standard YOLOX:
outputs = [cls, obj, reg]  # reg = [x, y, w, h]

# Oriented YOLOX:
outputs = [cls, obj, reg]  # reg = [x, y, w, h, angle]
```

## Training Tips

1. **Learning Rate**: Start with base LR = 0.01, warmup for first 5 epochs
2. **Augmentation**: Enable random rotation, flip, mosaic
3. **Batch Size**: 32-64 per GPU (depends on image size)
4. **Epochs**: 300-500 for good convergence
5. **Weight Initialization**: Pretrain on standard YOLOX first, then fine-tune
6. **Angle Loss Weight**: Usually 1.0 (same as other regression losses)

## Evaluation Metrics

- **mAP@0.5:0.95**: Mean Average Precision with standard IoU threshold
- **mAP@0.5**: Coarse localization evaluation
- **Angle MAE**: Mean Absolute Error for angle predictions
- **Speed**: Inference time (ms/image)

## Troubleshooting

### High Loss on Angles
- Normalize angles to [0, 180) range
- Check if angle ground truth is correct
- Use smaller learning rate for angle branch

### Poor Detection Performance
- Verify dataset format matches COCO schema
- Check if angles are in correct range
- Visualize training data with `scripts/visualize_oriented_boxes.py`

### CUDA Out of Memory
- Reduce batch size (-b flag)
- Enable cache=False in config
- Use smaller model (nano/tiny)

## References

1. YOLOX: [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)
2. Oriented Object Detection: [https://arxiv.org/abs/2108.05699](https://arxiv.org/abs/2108.05699)
3. Rotated IoU: [https://arxiv.org/abs/2105.05396](https://arxiv.org/abs/2105.05396)

## Citation

```bibtex
@article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

## License

Apache License 2.0 (same as original YOLOX)

## Contact

For questions, issues, or contributions, please open an issue on GitHub.
