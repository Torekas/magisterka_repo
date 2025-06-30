Absolutely! Here’s your **edited README**, integrating clear references to the code, describing its structure and main logic, and making it easy for users or collaborators to understand how the provided code connects to the dataset and workflow. This is tailored for GitHub and suitable for users who want to get started with the project or understand the codebase.

---

# Using the ACDC Dataset for Weather Condition Classification

## 1. Downloading the Dataset

This project uses the [**ACDC** (Adverse Conditions Dataset with Correspondences)](https://acdc.vision.ee.ethz.ch/download), which contains urban driving images under various weather conditions:

* **fog**
* **rain**
* **snow**
* **night**
* **clear** (reference)

Download the file:
**`rgb_anon_trainvaltest.zip`**
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

The core training, evaluation, and result visualization workflow is implemented in [`adverse_weather_detection.ipynb`](adverse_weather_detection.ipynb).
**Key points:**

* The code is **fully commented** and structured for clarity and reproducibility.
* Each weather condition is treated as a binary classification problem: *condition* vs. *clear*.
* All major steps are automated: data loading, augmentation, model training, evaluation, and metrics reporting.

### Main Steps (as implemented in the code):

1. **Preprocessing & Augmentation**

   * Image data is augmented during training for better generalization.
   * Separate transforms for training and evaluation are used.

2. **Model Definition**

   * Uses a pretrained ResNet-18 as the backbone.
   * The last layer is replaced for binary classification.

3. **Training and Validation**

   * For each weather condition (`fog`, `rain`, `snow`, `night`), the model is trained to distinguish between the condition and clear weather.
   * Progress (losses, accuracy) is logged and saved.

4. **Testing & Model Saving**

   * After training, the model is evaluated on the test set for each condition.
   * Model weights are saved as `model_<condition>_new.pth`.

5. **Visualization & Metrics**

   * Generates accuracy comparison bar plots (`accuracy_comparison.html`).
   * Plots validation loss curves for all conditions (`val_loss_comparison.html`).
   * Exports a styled summary table of quantitative metrics (`metrics_table.html` and `metrics_summary.csv`).

---

## 4. Sample Images

Below are example images (either generated or sample previews) for each category:

### Fog:

<img src="./demo/fog.png" width="640" height="360" alt="Example: Fog" />

### Rain:

<img src="./demo/rain.png" width="640" height="360" alt="Example: Rain" />

### Night:

<img src="./demo/night.png" width="640" height="360" alt="Example: Night" />

### Snow:

<img src="./demo/snow.png" width="640" height="360" alt="Example: Snow" />

Reference (clear) images are included for direct comparison.

---

## 5. Results & Next Steps

* **Model performance** (accuracy, loss) for each weather condition is reported in the outputs below.
<img src="./demo/ACDC_table.png" width="340" height="100" alt="Example: Snow" />
<img src="./demo/accuracy_comparison_ACDC.png" width="640" height="360" alt="Example: Snow" />
* **Next Steps:**

  * Adjust the model architecture, hyperparameters, or data augmentation for better performance.
  * Extend the dataset (e.g., using KITTI or PVDN) for improved robustness.
  * Optimize training (e.g., use a more powerful GPU, mixed precision).

---

## 6. Running the Code

To train and evaluate the models on your own machine:

1. Install requirements (PyTorch, torchvision, pandas, plotly, etc.).
2. Adjust the `data_root` path in `adverse_weather_detection.ipynb` to match your dataset location.

---

## 7. File Overview

* `adverse_weather_detection.ipynb` — Main script for preprocessing, training, evaluation, and reporting.
  *See in-code comments for detailed explanation of every function and step.*

---
