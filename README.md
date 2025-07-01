
---

# Master Thesis

This thesis focuses on intelligent Matrix headlights in automobiles. The goal is to identify the best possible solution for detecting the most crucial elements on the road for drivers, leveraging various models.

Currently, this repository contains several approaches, including:

* **Distinguishing between day and night.**
* **Recognizing different weather conditions** (snow, rain, fog) based on diverse image sets.
* **Detecting important road elements** such as traffic signs, pedestrians, vehicles, and the roadway itself. Three solutions are used here:

  * **A solution using the YOLOv5 model.**
  * **A solution using the YOLOv8 model.**
  * **A solution leveraging an ONNX-based model.**
* **Segmentation of the road with different approaches:**

  * **SegFormer**
  * **YOLOP**
  * **U-Net + ResNet-34**
  * **DeepLabV3+ResNet-50/100**
* **A full working simulator** that combines all components with a mechanism for steering the matrix headlights.
* **Future work**: possible prototype design and deployment concepts.

Each approach is accompanied by a dedicated `.ipynb` notebook or `.py` file in each approach subfolder, which explains the steps taken in each case.

## Key subdirectories:

* **Approches/Adverse_weather_conditions_detection:**
Contains code and notebooks for classifying fog, rain, snow, and night conditions using the ACDC dataset. The README explains the dataset layout, training workflow and provides sample results.

* **Approches/Final_simulator_and_future_implementation:**
Implements a simulator that merges object detection, road segmentation, and weather classifiers to adjust simulated matrix headlights in real time. The README outlines its system overview, processing pipeline, example outputs, and future hardware integration ideas.
The simulator.py script loads YOLOv5, a U-Net model, and four ResNet-18 weather classifiers, then processes video frames to control simulated headlight beams:

```python
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True).to(device).eval()
unet_model = load_unet_model(r"D:\night_segmentation_new\unet34_night_final_resized.pth", num_classes=3, device=device)
rain_model = models.resnet18(pretrained=False)
fog_model = models.resnet18(pretrained=False)
night_model = models.resnet18(pretrained=False)
snow_model = models.resnet18(pretrained=False)
```

* **Approches/ONNX_road_lines_car_detection:**
Provides instructions for using the HybridNets ONNX/TFLite project for simultaneous road and vehicle detection, including environment setup and running the detection.

* **Approches/Road_segmentation:**
Contains benchmarking of several segmentation models (SegFormer, DeepLabV3+ResNet50, U-Net, YOLOP, an OpenCV baseline). The README summarizes datasets, model performances and includes sample plots.
Training scripts for U-Net and DeepLabV3 show how datasets are loaded, losses computed, and metrics logged to Comet ML.

* **Approches/Yolov5_v8_car_sign_people_road_detection:**
Holds code/notebooks for object detection using YOLOv5 and YOLOv8. The README describes dataset creation, training procedures, and a comparison of YOLOv5 vs. YOLOv8 performance.
Scripts such as train v8.py and live v8.py provide training and inference routines for YOLOv8.

* **demo:**
Contains workflow diagrams and GIFs used in the README to illustrate the processing pipeline and simulator in action.

---

## Workflow Pipelines & Experiment Tracking

<table>
  <tr>
    <td align="center"><b>Workflow Pipeline (v2)</b><br>
      <img src="./demo/workflow_master_thesis.png" width="320" alt="Workflow Pipeline v2"/>
    </td>
    <td align="center"><b>Workflow Pipeline (v1)</b><br>
      <img src="./demo/Workflow.png" width="320" alt="Workflow Pipeline v1"/>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Comet ML Experiment Tracking</b><br>
      <img src="./demo/comet.png" width="640" alt="Comet ML Tracking"/>
    </td>
  </tr>
</table>

---

# Computers' Specs

## Laptop

### CPU and Overall Memory Status

| **Parameter**                  | **Value**                           |
| ------------------------------ | ----------------------------------- |
| MaxClockSpeed (MHz)            | 2592                                |
| Processor Name                 | Intel® Core™ i7-9750H CPU @ 2.60GHz |
| Number of Cores                | 6                                   |
| Number of Logical Processors   | 12                                  |
| **System Model:**              | ROG Strix G531GW\_G531GW            |
| **OS Name:**                   | Microsoft Windows 11 Pro            |
| **OS Version:**                | 10.0.26100 N/A Build 26100          |
| Total Physical Memory (MB)     | 16,234                              |
| Available Physical Memory (MB) | 4,241                               |
| Virtual Memory: Max Size (MB)  | 32,469                              |
| Virtual Memory: Available (MB) | 14,070                              |
| Virtual Memory: In Use (MB)    | 18,399                              |
| VRam (MB)                      | 8,006                               |

### Installed GPU Adapters

| AdapterRAM (GB) | DriverVersion  | Name                    | VideoModeDescription      |
| --------------- | -------------- | ----------------------- | ------------------------- |
| 1.00            | 26.20.100.6911 | Intel® UHD Graphics 630 | 1920 × 1080, 32-bit color |
| 4.00            | 32.0.15.7640   | NVIDIA GeForce RTX 2070 | 1920 × 1080, 32-bit color |

---

## PC

### CPU Specifications and Overall Memory Status

| **Parameter**                  | **Value**                  |
| ------------------------------ | -------------------------- |
| MaxClockSpeed (MHz)            | 2500                       |
| Processor Name                 | Intel® Core™ i5-14400F     |
| Number of Cores                | 10                         |
| Number of Logical Processors   | 16                         |
| **System Manufacturer:**       | LENOVO                     |
| **System Model:**              | 90UU00L7PL                 |
| **OS Name:**                   | Microsoft Windows 11 Pro   |
| **OS Version:**                | 10.0.26100 N/A Build 26100 |
| Total Physical Memory (MB)     | 32,490                     |
| Available Physical Memory (MB) | 19,139                     |
| Virtual Memory: Max Size (MB)  | 50,922                     |
| Virtual Memory: Available (MB) | 29,176                     |
| Virtual Memory: In Use (MB)    | 21,746                     |
| VRam (MB)                      | 7,949                      |

### Installed GPU Adapter

| AdapterRAM (GB) | DriverVersion | Name                       | VideoModeDescription      |
| --------------- | ------------- | -------------------------- | ------------------------- |
| 4.00            | 32.0.15.6103  | NVIDIA GeForce RTX 4060 Ti | 1920 × 1080, 32-bit color |

---

# Simulator in action
<table align = "center">
  <tr>
    <td align="center"><b>Simulator in action</b><br>
      <img src="./demo/gif_master_thesis.gif" width="800" height="320" alt="YOLOv5 Inference GIF"/>
    </td>
  </tr>
</table>