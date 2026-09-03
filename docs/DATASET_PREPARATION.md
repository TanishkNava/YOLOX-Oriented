# Dataset Preparation for Oriented YOLOX

## Converting Standard COCO to Oriented COCO

If your current COCO dataset doesn't have angle annotations, you need to add them.

### Method 1: From Polygon Annotations (Recommended)

If you have polygon-style rotated annotations:

```json
{
  "annotations": [
    {
      "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
      "bbox": [x, y, w, h],
      "category_id": 1
    }
  ]
}
```

Convert to oriented format:

```python
import json
import cv2
import numpy as np

def polygon_to_rotated_box(polygon):
    """
    Convert polygon to rotated box (cx, cy, w, h, angle).
    
    Args:
        polygon: list of [x1, y1, x2, y2, ..., x4, y4]
        
    Returns:
        (cx, cy, w, h, angle)
    """
    points = np.array(polygon).reshape(-1, 2)
    
    # Use cv2.minAreaRect to get oriented bounding box
    rect = cv2.minAreaRect(points.astype(np.float32))
    
    # rect = ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect
    
    # Normalize angle to [0, 180)
    if angle < 0:
        angle += 180
    if angle >= 180:
        angle -= 180
    
    return cx, cy, w, h, angle

def convert_coco_to_oriented(input_json, output_json):
    """Convert COCO JSON to oriented COCO JSON."""
    
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
    
    # Add angle to each annotation
    for ann in coco_data['annotations']:
        if 'segmentation' in ann and len(ann['segmentation']) > 0:
            # Use first polygon if multiple exist
            polygon = ann['segmentation'][0]
            cx, cy, w, h, angle = polygon_to_rotated_box(polygon)
            ann['angle'] = float(angle)
        else:
            # No segmentation, assume 0° (horizontal)
            ann['angle'] = 0.0
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"Converted COCO annotations saved to {output_json}")

# Usage
convert_coco_to_oriented('instances_train2017.json', 'instances_train2017_oriented.json')
```

### Method 2: Manual Annotation with Angle

Use annotation tools that support angle:
1. **LabelImg with rotation support**
2. **CVAT (Computer Vision Annotation Tool)**
3. **Supervisely**

Make sure your export includes angle field.

### Method 3: Programmatic Annotation

If you know object orientations from metadata:

```python
def add_angles_from_metadata(coco_json, metadata_file, output_json):
    """
    Add angles from external metadata file.
    
    Metadata format: {"image_id": {"object_id": angle_degrees}}
    """
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    for ann in coco['annotations']:
        img_id = str(ann['image_id'])
        obj_id = str(ann['id'])
        
        if img_id in metadata and obj_id in metadata[img_id]:
            ann['angle'] = metadata[img_id][obj_id]
        else:
            ann['angle'] = 0.0  # Default to horizontal
    
    with open(output_json, 'w') as f:
        json.dump(coco, f)
```

## Validation Checklist

Before training, verify your dataset:

```python
import json
import cv2
import numpy as np
from pathlib import Path

def validate_oriented_coco(json_path, img_dir):
    """Validate oriented COCO dataset."""
    
    with open(json_path) as f:
        coco = json.load(f)
    
    print("🔍 Validating Oriented COCO Dataset...\n")
    
    # Check 1: Required fields
    print("[1] Checking required fields...")
    required_fields = ['images', 'annotations', 'categories']
    for field in required_fields:
        if field not in coco:
            print(f"  ❌ Missing '{field}'")
            return False
        print(f"  ✓ '{field}' found ({len(coco[field])} items)")
    
    # Check 2: Annotations have angle
    print("\n[2] Checking angle annotations...")
    missing_angles = 0
    angle_range = [float('inf'), float('-inf')]
    
    for ann in coco['annotations']:
        if 'angle' not in ann:
            missing_angles += 1
            ann['angle'] = 0.0  # Add default
        else:
            angle = ann['angle']
            angle_range[0] = min(angle_range[0], angle)
            angle_range[1] = max(angle_range[1], angle)
    
    print(f"  Annotations with angle: {len(coco['annotations']) - missing_angles}")
    if missing_angles > 0:
        print(f"  ⚠ Missing angles (set to 0°): {missing_angles}")
    print(f"  Angle range: [{angle_range[0]:.1f}°, {angle_range[1]:.1f}°]")
    
    # Check 3: Image files exist
    print("\n[3] Checking image files...")
    missing_images = []
    for img in coco['images']:
        img_path = Path(img_dir) / img['file_name']
        if not img_path.exists():
            missing_images.append(img['file_name'])
    
    if missing_images:
        print(f"  ❌ Missing images: {len(missing_images)}")
        for f in missing_images[:5]:
            print(f"    - {f}")
    else:
        print(f"  ✓ All {len(coco['images'])} image files found")
    
    # Check 4: Annotation validity
    print("\n[4] Checking annotation validity...")
    invalid_anns = 0
    
    for ann in coco['annotations']:
        # Check bbox
        if 'bbox' not in ann or len(ann['bbox']) != 4:
            invalid_anns += 1
            continue
        
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            invalid_anns += 1
            continue
        
        # Check angle range
        angle = ann.get('angle', 0.0)
        if not (0 <= angle < 180):
            ann['angle'] = angle % 180  # Normalize
    
    print(f"  Valid annotations: {len(coco['annotations']) - invalid_anns}")
    if invalid_anns > 0:
        print(f"  ⚠ Invalid annotations: {invalid_anns}")
    
    print("\n✅ Validation complete!")
    return True

# Usage
validate_oriented_coco(
    'datasets/BirdShed/annotations/instances_train2017.json',
    'datasets/BirdShed/train2017'
)
```

## Dataset Statistics

After preparation, generate statistics:

```python
def dataset_statistics(json_path):
    """Generate dataset statistics."""
    with open(json_path) as f:
        coco = json.load(f)
    
    print("📊 Dataset Statistics\n")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Classes: {len(coco['categories'])}")
    
    # Distribution by class
    print("\nClass Distribution:")
    class_counts = {}
    for ann in coco['annotations']:
        cat_id = ann['category_id']
        class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
    
    for cat in coco['categories']:
        count = class_counts.get(cat['id'], 0)
        pct = 100 * count / len(coco['annotations'])
        print(f"  {cat['name']}: {count} ({pct:.1f}%)")
    
    # Angle distribution
    angles = [ann.get('angle', 0) for ann in coco['annotations']]
    print(f"\nAngle Statistics:")
    print(f"  Min: {min(angles):.1f}°")
    print(f"  Max: {max(angles):.1f}°")
    print(f"  Mean: {np.mean(angles):.1f}°")
    print(f"  Median: {np.median(angles):.1f}°")

from pathlib import Path
import numpy as np

dataset_statistics('datasets/BirdShed/annotations/instances_train2017.json')
```

## Expected Directory Structure

```
datasets/BirdShed/
├── train2017/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── val2017/
│   ├── image_100.jpg
│   └── ...
└── annotations/
    ├── instances_train2017.json  # With 'angle' field
    └── instances_val2017.json    # With 'angle' field
```

## Quick Start

```bash
# 1. Prepare base COCO
mkdir -p datasets/BirdShed/{train2017,val2017,annotations}
cp /your/images/train/* datasets/BirdShed/train2017/
cp /your/images/val/* datasets/BirdShed/val2017/

# 2. Add angles to annotations
python scripts/add_angles_to_coco.py \
  --input instances_train2017.json \
  --output datasets/BirdShed/annotations/instances_train2017.json \
  --from_polygons  # or --from_metadata metadata.json

# 3. Validate
python scripts/validate_dataset.py \
  --json datasets/BirdShed/annotations/instances_train2017.json \
  --image_dir datasets/BirdShed/train2017

# 4. Visualize
python scripts/visualize_oriented_boxes.py \
  --json datasets/BirdShed/annotations/instances_train2017.json \
  --image_dir datasets/BirdShed/train2017 \
  --num_samples 10
```

Done! Your dataset is ready for training. 🎉
