---

# YOLOv8 Road Object Detection

This repository demonstrates how to fine-tune the [YOLOv8](https://github.com/ultralytics/ultralytics) object detection model to detect various road objects and signs on Polish roads. The approach is inspired by [this tutorial](https://medium.com/@mikolaj.kolek/fine-tuning-yolo-for-road-sign-and-object-detection-on-polish-roads-71a366e9a876) and uses the **BDD100K** (Berkeley DeepDrive) dataset, which provides a large number of labeled driving images.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Data Processing](#data-processing)
5. [Training](#training)
6. [Inference](#inference)
7. [Results and Logging](#results-and-logging)
8. [License](#license)
9. [References](#references)

---

## Introduction

- **Goal**: Detect key elements on the road—such as cars, pedestrians, traffic lights, and road signs—in images and videos.
- **Model**: [YOLOv8](https://github.com/ultralytics/ultralytics) from Ultralytics.
- **Detected Classes**: 
  ```
  [
    'car', 
    'different-traffic-sign', 
    'green-traffic-light', 
    'motorcycle', 
    'pedestrian', 
    'pedestrian-crossing', 
    'prohibition-sign', 
    'red-traffic-light', 
    'speed-limit-sign', 
    'truck', 
    'warning-sign'
  ]
  ```

The repository contains code to:
- Pre-process and convert dataset annotations (BDD100K) into YOLO format.
- Train a custom YOLOv8 model (`fine_tuned_yolov8s.pt`).
- Run real-time or offline inference on videos/images with the trained model.

---

## Project Structure

A typical layout in this repository is as follows:

```
.
├── notebooks
│   ├── data
│   │   └── bdd100k.names        # Class names
│   └── data_processing.ipynb    # Jupyter Notebook for data preparation & annotation conversion
├── poetry.lock
├── pyproject.toml
├── README.md                    # This README file
└── road_detection_model
    ├── data_finetune.yaml       # YOLO configuration for fine-tuning
    ├── data.yaml                # YOLO configuration for baseline training
    ├── __init__.py
    ├── live.py                  # Script for real-time detection (e.g., from an .mp4 file)
    ├── Models
    │   ├── fine_tuned_yolov8s.pt
    │   ├── pre_trained_yolov8s.pt
    │   └── yolov8n.pt
    ├── runs
    │   ├── detect
    │   │   ├── train
    │   │   ├── train2
    │   │   ├── val
    │   │   └── val2
    │   └── fine_tuning
    │       ├── train
    │       ├── train2
    │       └── train3
    ├── test_images
    │   └── test_film.mp4        # Example video for testing inference
    ├── train.py                 # Script for training
    └── validate.py              # Script for validation
```

### Key Files and Folders

- **notebooks/**  
  Contains Jupyter notebooks and auxiliary data needed for processing annotations.

- **road_detection_model/**  
  Main folder for:
  - **YOLO config files** (`data.yaml`, `data_finetune.yaml`).
  - **Trained models** (inside `Models/`).
  - **Scripts** (`train.py`, `validate.py`, `live.py`).
  - **runs/** subfolders for training and detection logs.

- **test_images/**  
  Contains test media files (e.g., `test_film.mp4`).

---

## Dataset

We utilize the [BDD100K Dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/) which includes:

- **100k_images_train.zip**  
- **100k_images_val.zip**  
- **100k_images_test.zip**  
- **bddk100k_det_20_labels_trainval.zip** (Annotation files)

**Note**: Each ZIP file is quite large, so ensure you have sufficient disk space and bandwidth.

1. **Download** the dataset from [BDD100K official site](https://dl.cv.ethz.ch/bdd100k/data/).
2. **Unzip** the images (train/val/test) into a local directory.
3. **Obtain** the annotation files (JSON format) for bounding boxes.

---

## Data Processing

Before training, we need to convert the original BDD100K annotations (in JSON format) into YOLO-compatible text files. A sample workflow is as follows:

1. **Open** `notebooks/data_processing.ipynb`.
2. **Set** the paths to your BDD100K images and annotation files.
3. **Run** the notebook cells to:
   - Parse the JSON annotations.
   - Create YOLO `.txt` label files with the bounding box coordinates (normalized).
   - Organize images and labels into train/val/test directories.

The `bdd100k.names` file lists the class labels relevant to your training.

After you set up eveything with this you can swap the created dataset set into your files adjusted for your training.

---

## Training

We use the Ultralytics YOLOv8 environment for training:

1. **Install** dependencies. For example, if you're using [Poetry](https://python-poetry.org/) (Optionally):
   ```bash
   poetry install
   poetry shell
   ```
   Or if using pip:
   ```bash
   pip install ultralytics comet_ml
   ```

2. **Configure** your training parameters. In `data.yaml` or `data_finetune.yaml`, set:
   - `train`: path to your training images/labels
   - `val`: path to your validation images/labels
   - `names`: list of class labels
   - **Hyperparameters** (optional) for epochs, batch size, etc.

3. **Run** the training script:
   ```bash
   cd road_detection_model
   python train.py --config data_finetune.yaml --epochs 50 --batch-size 16
   ```
   *(Adapt the command-line arguments as needed.)*

4. **Models**:
   - `yolov8n.pt` is a smaller, faster YOLOv8 model.
   - `yolov8s.pt` is a slightly larger model with better accuracy.
   - **fine_tuned_yolov8s.pt** is your custom-trained model.

During or after training, you can monitor metrics such as **mAP**, **precision**, and **recall**.

---

## Inference

Once training is complete, you can test the model on:
- **Static images** (individual frames) 
- **Video streams** (e.g., a dashcam video, `test_film.mp4`)

### Using `live.py` for Video

```bash
cd road_detection_model
python live.py --weights Models/fine_tuned_yolov8s.pt --source test_images/test_film.mp4
```

- **--weights**: Path to the trained YOLOv8 model.
- **--source**: Path to the video file or a webcam index (e.g., `0` for default webcam).

The script will display detections in real-time or write the annotated output to a new video file (depending on how you implement it).

<video src="./demo/visualisation.mp4" controls width="640" height="360">
</video>

---

## Results and Logging

- All training artifacts and logs are stored in **`runs/`** directories:
  - **`runs/detect/train`**, **`runs/detect/val`**
  - **`runs/fine_tuning/train`**, **`runs/fine_tuning/train2`**, etc.

- If using **[Comet ML](https://www.comet.ml/)**, you can also log and visualize:
  - Metrics (loss, mAP, precision, recall)
  - Confusion matrices
  - Example predictions
  Simply add your **Comet ML** API key to your environment or script configuration.

---

## License

This project is governed by the licenses of:
- **BDD100K** dataset — see their [official license](https://doc.bdd100k.com/license.html).
- **Ultralytics YOLOv8** — see [license details here](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

Feel free to modify or distribute this code under the terms specified by the original licenses.

---

## References

- [BDD100K Dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Mikołaj Kolek — Fine-tuning YOLO article](https://medium.com/@mikolaj.kolek/fine-tuning-yolo-for-road-sign-and-object-detection-on-polish-roads-71a366e9a876)
- [Comet ML for experiment tracking](https://www.comet.ml/)

---