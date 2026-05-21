# AI Vision System — Training Workflow

## Overview

This document defines the standardized workflow for training, validating, and deploying custom object detection models within the AI Vision System.

The objective is to ensure:
- Consistency across all models
- Reproducible training pipelines
- Production-ready performance
- Scalable model management

---

## System Architecture

The system is built around **modular model profiles**, where each detectable object or defect is trained and managed independently.

Each model profile includes:
- Dataset (images and labels)
- Trained model weights
- Configuration metadata

### Example Profiles

```
yellow_daifuku
mouse
box_label
damaged_part
```

This design enables:
- Independent iteration and improvement
- Controlled deployment of models
- Simplified debugging and maintenance

---

## Data Collection

### Command

```powershell
python training\collect_images.py
```

### Output Location

```
data/datasets/<object_name>/images/train/
```

### Data Collection Guidelines

To ensure robust model performance:
- Capture images from multiple angles and orientations
- Include varying lighting conditions (bright, low-light, shadows)
- Use different distances and camera perspectives
- Include diverse backgrounds and environments
- Collect data under real production conditions

**Principle:** Model performance is directly correlated with dataset diversity and quality.

---

## Image Annotation

### Command

```powershell
python training\label_images.py
```

### Controls

| Key | Function |
|-----|----------|
| Mouse Drag | Draw bounding box |
| S | Save label |
| N | Next image |
| U | Undo last annotation |
| Q | Quit |

---

### Label Format (YOLO)

Each image must have a corresponding `.txt` label file using YOLO format:

```
<class_id> <center_x> <center_y> <width> <height>
```

#### Example

```
0 0.523438 0.412500 0.234375 0.187500
```

### Annotation Requirements

- All values must be space-separated
- Coordinates must be normalized (range: 0 to 1)
- Label filenames must match image filenames exactly

```
mouse_001.jpg
mouse_001.txt
```

---

## Dataset Preparation

### Train / Validation Split

Split dataset into:
- Training set (used for learning)
- Validation set (used for evaluation)

### Recommended Ratio

```
80% Training
20% Validation
```

### Directory Structure

```
data/datasets/<object_name>/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

---

## Dataset Configuration

Each dataset must define a `data.yaml` configuration file.

### Example

```yaml
path: data/datasets/mouse
train: images/train
val: images/val

names:
  0: mouse
```

---

## Model Training

### Command

```powershell
python training\train_my_items.py
```

### Output

```
runs/<model_name>/weights/best.pt
```

#### Example

```
runs/mouse/weights/best.pt
```

---

## Model Packaging

After training, create a standardized model profile.

### Directory Structure

```
models/mouse/
├── best.pt
├── classes.txt
├── config.json
```

### classes.txt

```
mouse
```

### config.json

```json
{
  "model_name": "mouse",
  "input_size": 640,
  "confidence_threshold": 0.35,
  "target_class_id": 0
}
```

---

## Model Activation

Update the active model configuration:

**File:**
```
app/config.py
```

### Example

```python
ACTIVE_MODEL_PROFILE = "mouse"
TARGET_CLASSES = [0]
```

---

## Inference / Detection

### Option 1

```powershell
scripts\run_app.bat
```

### Option 2

```powershell
python -m app.main
```

---

## Model Optimization

### Data-Level Improvements
- Increase dataset size
- Include negative samples (images without target objects)
- Add edge cases and failure scenarios
- Ensure consistent and accurate labeling

### Training-Level Improvements
- Retrain frequently with updated data
- Monitor validation performance
- Adjust hyperparameters if required

### Deployment Testing
Validate performance under:
- Different lighting conditions
- Operational environments
- Real production workflows

---

## Dataset Size Guidelines

| Dataset Size | Use Case |
|-------------|----------|
| 20–50 images | Initial prototyping |
| 100–300 images | Baseline production |
| 300+ images | High reliability production |

---

## End-to-End Example Workflow

```powershell
# Step 1: Collect images
python training\collect_images.py

# Step 2: Label images
python training\label_images.py

# Step 3: Split dataset (80/20)

# Step 4: Configure data.yaml

# Step 5: Train model
python training\train_my_items.py

# Step 6: Package model
# models/<model_name>/

# Step 7: Activate model
# app/config.py

# Step 8: Run inference
python -m app.main
```

---

## Best Practices

- Validate all annotations before training
- Maintain consistent dataset structure
- Version datasets and models where possible
- Track model performance across iterations
- Avoid training with noisy or incorrect labels

---

## Summary

A high-performing model depends on:
- High-quality, diverse data
- Accurate annotations
- Consistent workflow execution

Following this process ensures reliable, scalable, and production-ready AI vision models.