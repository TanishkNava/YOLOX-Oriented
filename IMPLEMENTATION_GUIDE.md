# Oriented YOLOX - Step-by-Step Implementation Guide

This guide walks you through implementing Oriented YOLOX from scratch on the Megvii YOLOX codebase.

## Phase 1: Setup & Understanding (Day 1)

### Step 1.1: Clone and Install Base YOLOX

```bash
cd /workspace
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -e .
pip install opencv-python pycocotools tensorboard
```

### Step 1.2: Understand YOLOX Architecture

Key files to review:
- `yolox/models/yolox.py` - Main model
- `yolox/models/yolo_head.py` - Detection head (standard)
- `yolox/data/datasets/coco.py` - Data loading

**What to understand**:
- YOLOX uses **decoupled head** (separate branches for cls, obj, reg)
- Output format: `(batch_size, num_anchors, num_classes + 5)` where 5 = [x, y, w, h, obj]
- **Anchor-free**: Directly predicts box coordinates
- **SimOTA**: Label assignment strategy

### Step 1.3: Prepare Your Dataset

Your COCO JSON needs angle information. Two options:

**Option A: If you have annotations in another format**
- Convert to COCO first using tool like [YOLO2COCO](https://github.com/RapidAI/YOLO2COCO)
- Then add angles manually

**Option B: If you have polygon/rotated annotations**
- Convert polygons to (cx, cy, w, h, θ) using OpenCV
- Add to COCO JSON

Example structure:
```python
# Your COCO should have 'angle' field:
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 50, 80],      # x, y, w, h
      "angle": 45.0,                   # NEW: angle in degrees [0, 180)
      "iscrowd": 0,
      "area": 4000
    }
  ]
}
```

## Phase 2: Implement Core Components (Days 2-3)

### Step 2.1: Create Oriented Head Module

Create `yolox/models/yolo_head_oriented.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolox.models.yolo_head import YOLOXHead
import numpy as np

class YOLOXHeadOriented(YOLOXHead):
    """
    YOLOX Head for Oriented Object Detection.
    Adds angle prediction to standard YOLOX head.
    
    Output: [x, y, w, h, angle, obj_conf, class_probs]
    """
    
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        super().__init__(
            num_classes=num_classes,
            width=width,
            strides=strides,
            in_channels=in_channels,
            act=act,
            depthwise=depthwise,
        )
        
        self.angle_prediction = True  # Flag for oriented detection
    
    def forward(self, xin):
        """
        Modified forward to include angle prediction.
        
        Args:
            xin: list of feature maps from FPN
            
        Returns:
            outputs: list of predictions at each scale
        """
        outputs = []
        
        for k, (cls_conv, obj_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.obj_convs, self.reg_convs, self.strides, xin)
        ):
            # Standard YOLOX branches
            cls_output = cls_conv(x)
            cls_output = self.cls_preds[k](cls_output)  # (B, num_classes, H, W)
            
            obj_output = obj_conv(x)
            obj_output = self.obj_preds[k](obj_output)  # (B, 1, H, W)
            
            reg_output = reg_conv(x)
            reg_output = self.reg_preds[k](reg_output)  # (B, 4, H, W)
            
            # NEW: Add angle prediction branch (reuse reg_output features)
            # Option 1: Shared feature extraction (recommended)
            angle_output = self.reg_preds[k](reg_output)  # (B, 1, H, W)
            
            # Reshape outputs
            batch_size = x.shape[0]
            height, width = x.shape[2:]
            
            cls_output = cls_output.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.num_classes
            )  # (B, H*W, num_classes)
            
            obj_output = obj_output.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 1
            )  # (B, H*W, 1)
            
            reg_output = reg_output.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 4
            )  # (B, H*W, 4)
            
            angle_output = angle_output.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 1
            )  # (B, H*W, 1)
            
            # Concatenate: [x, y, w, h, angle]
            reg_output = torch.cat([reg_output, angle_output], dim=-1)
            
            # Combine outputs
            output = torch.cat([reg_output, obj_output, cls_output], dim=-1)
            outputs.append(output)
        
        return outputs
    
    def get_losses(
        self,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
        device,
    ):
        """
        Compute losses including angle regression loss.
        """
        # Get standard YOLOX losses (obj + cls)
        loss_info = super().get_losses(
            x_shifts, y_shifts, expanded_strides, labels, outputs,
            origin_preds, dtype, device
        )
        
        # Add angle loss
        # Extract predicted angles
        pred_angles = outputs[..., 4:5]  # (B, num_anchors, 1)
        
        # Extract GT angles from labels
        # labels format: [image_idx, class_id, cx, cy, w, h, angle]
        gt_angles = labels[:, 6:7]  # (num_gt, 1)
        
        # Compute angle loss (smooth L1)
        angle_loss = self.angle_loss_fn(pred_angles, gt_angles)
        
        # Add to total loss
        loss_info['angle_loss'] = angle_loss
        loss_info['total_loss'] = (
            loss_info['loss_cls'] + 
            loss_info['loss_obj'] + 
            loss_info['loss_reg'] + 
            angle_loss * self.angle_loss_weight
        )
        
        return loss_info
    
    def angle_loss_fn(self, pred, target):
        """
        Smooth L1 loss for angle prediction.
        Handles angle wraparound at 180°.
        """
        # Normalize angles to [0, 180)
        pred = pred % 180.0
        target = target % 180.0
        
        # Compute angular difference
        diff = torch.abs(pred - target)
        diff = torch.min(diff, 180.0 - diff)  # Handle wraparound
        
        # Smooth L1 loss
        return F.smooth_l1_loss(diff, torch.zeros_like(diff))
```

### Step 2.2: Create Rotated NMS

Create `yolox/utils/rotated_nms.py`:

```python
import torch
import numpy as np
from shapely.geometry import Polygon

def rotated_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two rotated boxes.
    
    Args:
        box1: [x, y, w, h, angle] (center format)
        box2: [x, y, w, h, angle] (center format)
        
    Returns:
        iou: float, IoU value
    """
    # Convert to corner format
    poly1 = get_rotated_polygon(box1)
    poly2 = get_rotated_polygon(box2)
    
    # Calculate intersection and union
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        iou = intersection / union if union > 0 else 0.0
    except:
        iou = 0.0
    
    return iou

def get_rotated_polygon(box):
    """
    Convert rotated box to Shapely Polygon.
    
    Args:
        box: [x, y, w, h, angle] in degrees
        
    Returns:
        Polygon object
    """
    x, y, w, h, angle = box
    angle_rad = np.radians(angle)
    
    # Corner points of box (before rotation)
    corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ])
    
    # Rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Apply rotation and translation
    corners = corners @ R.T + np.array([x, y])
    
    return Polygon(corners)

def rotated_nms(dets, scores, iou_thr=0.5):
    """
    Apply rotated NMS.
    
    Args:
        dets: (N, 5) array, [x, y, w, h, angle]
        scores: (N,) array, detection scores
        iou_thr: float, NMS threshold
        
    Returns:
        keep: list of indices to keep
    """
    if len(dets) == 0:
        return []
    
    # Sort by score
    order = np.argsort(scores)[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = np.array([
            rotated_iou(dets[i], dets[j]) for j in order[1:]
        ])
        
        # Keep boxes with IoU < threshold
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]
    
    return keep
```

### Step 2.3: Create Oriented COCO Dataset Loader

Create `yolox/data/datasets/coco_oriented.py`:

```python
import numpy as np
from yolox.data.datasets import COCODataset

class COCODatasetOriented(COCODataset):
    """
    COCO dataset with angle annotations for oriented detection.
    
    Expected annotations format:
    {
        "id": int,
        "image_id": int,
        "category_id": int,
        "bbox": [x, y, w, h],
        "angle": float (degrees),  # NEW: angle in [0, 180)
        "area": float,
        "iscrowd": int
    }
    """
    
    def pull_item(self, index):
        """
        Extract image and annotations with angle information.
        """
        # Get standard COCO data
        img, targets = super().pull_item(index)
        
        if targets.shape[0] > 0:
            # Extract angles from annotations
            # targets format: [cls, cx, cy, w, h] from parent class
            # Need to append angle: [cls, cx, cy, w, h, angle]
            
            img_id = self.ids[index]
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            angles = []
            for ann in anns:
                # Get angle from annotation, default to 0 if not present
                angle = ann.get('angle', 0.0)
                angles.append(angle)
            
            if angles:
                angles = np.array(angles, dtype=np.float32).reshape(-1, 1)
                # Append angles to targets
                targets = np.concatenate([targets, angles], axis=1)
        
        return img, targets
    
    def load_annotations(self):
        """
        Load image and annotation indexes.
        Adds validation for angle field.
        """
        super().load_annotations()
        
        # Optional: Check if angles exist in dataset
        print(f"Loading {len(self.ids)} images with oriented annotations...")
        
        # Sample check
        if len(self.ids) > 0:
            sample_ann = self.coco.loadAnns(
                self.coco.getAnnIds(imgIds=self.ids[0])
            )
            if sample_ann and 'angle' in sample_ann[0]:
                print("✓ Angle annotations found in dataset")
            else:
                print("⚠ Warning: 'angle' field not found. Assuming 0° for all boxes.")
```

## Phase 3: Integration & Training (Days 4-5)

### Step 3.1: Create Custom Config File

Create `exps/custom/yolox_s_oriented_birdShed.py`:

```python
import os
from yolox.exp import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 5  # vehicle, person, stick, spray_machine, feed_machine
        self.depth = 0.33
        self.width = 0.50
        self.scale = (416, 416)
        self.random_size = (14, 26)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = True
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.print_interval = 100
        self.eval_interval = 10
        self.save_history_ckpt = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Oriented YOLOX specific
        self.use_oriented_head = True
        self.angle_loss_weight = 1.0
        self.nms_iou_thr = 0.45
        self.nms_score_thr = 0.01
    
    def get_model(self):
        from yolox.models import YOLOX
        from yolox.models.yolo_head_oriented import YOLOXHeadOriented
        
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        
        model = YOLOX(self.backbone, self.head, self.head_conv)
        # Replace head with oriented version
        model.head = YOLOXHeadOriented(
            num_classes=self.num_classes,
            width=self.width,
            strides=self.strides,
            in_channels=self.in_channels,
            act=self.act,
        )
        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        return model
    
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import COCODatasetOriented  # Use oriented version
        
        dataset = COCODatasetOriented(
            data_dir=os.path.join(self.data_dir, "train2017"),
            json_file="instances_train2017.json",
            name="train",
            img_size=self.input_size,
            preproc=self.preproc,
            cache=cache_img,
        )
        # ... rest of loader setup
```

### Step 3.2: Modify Training Script

Update `yolox/tools/train.py` to handle oriented annotations:

```python
# In the get_optimizer function or loss computation:

def criterion(outputs, labels):
    # Standard losses
    loss_cls = compute_cls_loss(outputs, labels)
    loss_obj = compute_obj_loss(outputs, labels)
    loss_reg = compute_reg_loss(outputs, labels)
    
    # NEW: Angle loss
    if model.head.angle_prediction:
        loss_angle = compute_angle_loss(
            outputs[..., 4:5],  # predicted angles
            labels[..., 6:7]    # GT angles
        )
        total_loss = loss_cls + loss_obj + loss_reg + loss_angle * angle_loss_weight
    else:
        total_loss = loss_cls + loss_obj + loss_reg
    
    return total_loss
```

### Step 3.3: Prepare & Run Training

```bash
# 1. Link dataset
export YOLOX_DATADIR=/path/to/datasets
ln -s /path/to/your/coco ./datasets/BirdShed

# 2. Verify data
python scripts/visualize_oriented_boxes.py --dataset_dir datasets/BirdShed/train2017

# 3. Train
python -m yolox.tools.train \
  -f exps/custom/yolox_s_oriented_birdShed.py \
  -d 1 \
  -b 32 \
  --fp16 \
  -o

# 4. Monitor training
tensorboard --logdir ./YOLOX_outputs
```

## Phase 4: Testing & Evaluation (Day 6)

### Step 4.1: Create Inference Script

Create `tools/demo_oriented.py`:

```python
import argparse
import cv2
import numpy as np
from yolox.utils.rotated_nms import rotated_nms

def draw_rotated_box(image, box, color=(0, 255, 0), thickness=2):
    """Draw rotated bbox on image."""
    x, y, w, h, angle = box
    angle_rad = np.radians(angle)
    
    # Get corners
    corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ])
    
    # Rotate
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = corners @ R.T + np.array([x, y])
    
    # Draw
    corners = corners.astype(int)
    cv2.polylines(image, [corners], True, color, thickness)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    
    # Load model and run inference
    # ... implementation
```

### Step 4.2: Evaluate on Test Set

```bash
python -m yolox.tools.eval \
  -f exps/custom/yolox_s_oriented_birdShed.py \
  -c checkpoints/yolox_s_oriented_best.pth \
  -b 32 -d 1 --conf 0.001
```

## Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| "angle field not in annotations" | Add angles to COCO JSON using `scripts/add_angles_to_coco.py` |
| Loss NaN after a few iterations | Check angle normalization, use smaller LR |
| Poor angle predictions | Increase `angle_loss_weight`, visualize training data |
| Memory error | Reduce batch size, disable cache |
| Inferior accuracy vs standard YOLOX | Add more training epochs, warmup longer, check data quality |

## Next Steps

1. ✅ Implement core modules
2. ✅ Integrate with YOLOX
3. ✅ Train on your dataset
4. ✅ Evaluate and optimize
5. 🔄 **Deploy to production** (export to ONNX, TensorRT, etc.)

Good luck! 🚀
