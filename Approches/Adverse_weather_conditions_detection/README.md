
---

# Using the ACDC Dataset for Weather Condition Classification

## 1. Downloading the Dataset
This project is based on the [**ACDC** (Adverse Conditions Dataset with Correspondences)](https://acdc.vision.ee.ethz.ch/download).  
- File: **`rgb_anon_trainvaltest.zip`**  
- The dataset contains images under various weather conditions: **fog**, **rain**, **snow**, **night**, and **clear** (reference).

## 2. Directory Structure
After unzipping and arranging the data, your main folder (`data_root/`) might look like this:

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

### Explanation
- **fog**, **rain**, **snow**, **night** – these folders contain images under each respective weather condition.  
- Each folder includes **train**, **val**, and **test** subfolders to enable consistent training, validation, and testing splits.  
- Within the **train**, **val**, and **test** folders, there are two subfolders: one for the target weather condition (e.g., `fog/`) and one for reference images (e.g., `clear/`).

## 3. Training the Model
We use **PyTorch** to train the model. In a sample Jupyter notebook (`.ipynb`):

1. **Import libraries** (PyTorch, torchvision, numpy, etc.).  
2. **Load images** from the above structure using a custom DataLoader or existing image-loading utilities.  
3. **Define the network architecture** (e.g., a CNN) or use a pretrained model (ResNet, EfficientNet, MobileNet, etc.).  
4. **Train** the model:  
   - Set hyperparameters (number of epochs, batch size, learning rate).  
   - Define the loss function (e.g., `CrossEntropyLoss`) and optimizer (e.g., `Adam`).  
   - Implement the training and validation loop.  

5. **Save the model**:  
   - After training completes, you can save the model to a `.pth` or `.pt` file, for example:
     ```python
     weather_classification = ['fog', 'night', 'rain', 'snow']
     torch.save(model.state_dict(), f'{weather_classification}_model.pth')
     ```
6. **Evaluate** the model on the test set to gauge accuracy in distinguishing different weather conditions.

## 4. Sample Images
Below are example images (either generated or sample previews) for each category:

### Fog:
<img src="./demo/fog.png" width="640" height="360" alt="Example: Fog" />

### Rain:
<img src="./demo/rain.png" width="640" height="360" alt="Example: Clear" />

### Night:
<img src="./demo/night.png" width="640" height="360" alt="Example: Night" />

### Snow:
<img src="./demo/snow.png" width="640" height="360" alt="Example: Snow" />

To the comparison there are files with clear weather to see the difference.


## 5. Conclusions and Next Steps
- **Model architecture adjustment**: Depending on your results (accuracy, precision, recall), you may fine-tune your network or try a different architecture.  
- **Dataset extension**: You can include other datasets (e.g., KITTI, PVDN) to improve generalization.  
- **Performance optimization**: Consider training on a more powerful GPU, using Mixed Precision Training, or other optimizations.  

---
