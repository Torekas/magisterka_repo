
---

# Using the ACDC Dataset for Weather Condition Classification

## 1. Downloading the Dataset

This part of the project uses the [**ACDC** (Adverse Conditions Dataset with Correspondences)](https://acdc.vision.ee.ethz.ch/download), which contains urban driving images under various weather conditions: **fog**, **rain**, **snow**, **night**, and **clear** (reference).

Download the file: **`rgb_anon_trainvaltest.zip`**
and extract it into your chosen directory.

---

## 2. Directory Structure

After unzipping and arranging the data, your main folder (`data_root/`) should look like this:

```
data_root/
├── fog/
│   ├── train/
│   │   ├── fog/
│   │   └── clear/
│   ├── val/
│   │   ├── fog/
│   │   └── clear/
│   └── test/
│       ├── fog/
│       └── clear/
├── rain/
│   ├── train/
│   │   ├── rain/
│   │   └── clear/
│   ├── val/
│   │   ├── rain/
│   │   └── clear/
│   └── test/
│       ├── rain/
│       └── clear/
├── snow/
│   ├── train/
│   │   ├── snow/
│   │   └── clear/
│   ├── val/
│   │   ├── snow/
│   │   └── clear/
│   └── test/
│       ├── snow/
│       └── clear/
└── night/
    ├── train/
    │   ├── night/
    │   └── clear/
    ├── val/
    │   ├── night/
    │   └── clear/
    └── test/
        ├── night/
        └── clear/
```

Each condition folder includes **train**, **val**, and **test** splits.
Inside each split you will find images for the target condition (e.g. `fog/`) and corresponding reference `clear/` images.

---

## 3. Model Training & Evaluation

The core workflow is implemented in [`adverse_weather_detection.ipynb`](adverse_weather_detection.ipynb).

**Key points:**

* The code is fully commented and structured for clarity and reproducibility.
* Each weather condition is treated as a binary classification problem: *condition* vs. *clear*.
* All major steps are automated: data loading, augmentation, model training, evaluation, and metrics reporting.

**Main Steps:**

1. **Preprocessing & Augmentation**: Image data is augmented for better generalization. Separate transforms for train/eval.
2. **Model Definition**: Uses a pretrained ResNet-18 backbone with a custom last layer.
3. **Training and Validation**: Trained separately for each condition to distinguish it from clear weather. Logs progress.
4. **Testing & Model Saving**: Evaluates on test set; model weights saved as `model_<condition>_new.pth`.
5. **Visualization & Metrics**: Generates bar plots, loss curves, and exports summary tables.

---

## 4. Sample Images

<table>
  <tr>
    <td align="center"><b>Fog</b><br><img src="./demo/fog.png" width="320" alt="Example: Fog"/></td>
    <td align="center"><b>Rain</b><br><img src="./demo/rain.png" width="320" alt="Example: Rain"/></td>
  </tr>
  <tr>
    <td align="center"><b>Snow</b><br><img src="./demo/snow.png" width="320" alt="Example: Snow"/></td>
    <td align="center"><b>Night</b><br><img src="./demo/night.png" width="320" alt="Example: Night"/></td>
  </tr>
</table>

Reference (clear) images are included for direct comparison in each subfolder.

---

## 5. Results & Next Steps

<table>
  <tr>
    <td>
      <b>Metrics Table</b><br>
      <img src="./demo/ACDC_table.png" width="340" alt="Metrics Table"/>
    </td>
    <td>
      <b>Accuracy Comparison</b><br>
      <img src="./demo/accuracy_comparison_ACDC.png" width="340" alt="Accuracy Comparison"/>
    </td>
  </tr>
</table>

**Next Steps:**

* Adjust model architecture, hyperparameters, or augmentations for better performance.
* Extend the dataset (e.g., add KITTI or PVDN) for improved robustness.
* Optimize training (e.g., more powerful GPU, mixed precision).

---

## 6. Running the Code

To train and evaluate models:

1. Install requirements (`PyTorch`, `torchvision`, `pandas`, `plotly`, etc.).
2. Adjust the `data_root` path in `adverse_weather_detection.ipynb` to match your dataset location.

---

## 7. File Overview

* `adverse_weather_detection.ipynb` — Main notebook for preprocessing, training, evaluation, and reporting.

  * *See in-code comments for details on every function and step.*

---
