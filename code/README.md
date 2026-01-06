
# X-ray Image Density Distribution Prediction System Based on Deep Learning

This project implements the functionality of using deep learning models to predict the density distribution of X-ray images, including UNet series segmentation models and density prediction models.

## Directory Structure

```text
code/
├── README.md                   # Project documentation
├── main.py                     # Main program, contains model training and testing functions
├── dataset.py                  # Dataset processing class
├── metrix.py                   # Evaluation metrics calculation (IoU, Dice, Hausdorff distance, etc.)
├── plot.py                     # Plotting and visualization functions
├── ML_density.ipynb            # Jupyter Notebook for density prediction model training
├── iou_example.py              # IoU calculation example
├── nets/                       # Network model definitions
│   ├── Unet_pp.py              # UNet++ model
│   ├── attention_Unet.py       # Attention UNet model
│   └── resNet.py               # ResNet UNet model
└── pre/                        # Preprocessing and deployment related scripts
    ├── Model_location_UNet.py  # UNet model deployment script
    ├── Model_location_density.py # Density prediction deployment script
    └── preprocessing.py        # Data preprocessing script
```

## Features

1. Supports multiple UNet architectures (UNet++, Attention UNet, ResNet UNet)
2. Image segmentation and density prediction functionality
3. Multiple evaluation metrics (IoU, Dice coefficient, Hausdorff distance)
4. Visualization result output

## Environment Configuration

### Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install scikit-image
pip install pillow
pip install pandas
pip install joblib
pip install scipy
pip install xgboost
pip install lightgbm
pip install tqdm
```

### Recommended Configuration

- Python 3.7+
- PyTorch 1.9+
- CUDA 10.2+ (for GPU acceleration)

## Dataset Requirements

### Segmentation Model Dataset

The dataset should be organized according to the following structure:

```text
datasets/
└── tryx/
    ├── training/
    │   ├── images/            # Training images
    │   └── 1st_manual/        # Training labels (masks)
    └── test/
        ├── images/            # Test images
        └── 1st_manual/        # Test labels (masks)
```

- **Image Format:** Supports common image formats (JPG, PNG, etc.)
- **Mask Format:** Binary images (0 or 255)
- **Image Size:** It is recommended to keep them consistent; the code will automatically resize them to 576×576.

### Density Prediction Model Dataset

- **Input:** Excel file containing features such as RGB values, dose, thickness, etc.
- **Output:** Density value
- **File Path:** `./database/hist_data_with_density_noisy2.xlsx`

## Training Guide

### 1. Segmentation Model Training

Run the main program to train the model:

```bash
# Train UNet++ model
python main.py --arch unet++ --epoch 20 --batch_size 2

# Train Attention UNet model
python main.py --arch Attention_UNet --epoch 20 --batch_size 2

# Train ResNet UNet model
python main.py --arch resnet34_unet --epoch 20 --batch_size 2
```

### Training Parameters

| Parameter           | Description                                                  | Default       |
| ------------------- | ------------------------------------------------------------ | ------------- |
| `--action`          | Operation type: `train`, `test`, `train&test`                | `train&test`  |
| `--arch`            | Model architecture: `unet++`, `Attention_UNet`, `resnet34_unet` | -             |
| `--epoch`           | Number of training epochs                                    | `1`           |
| `--batch_size`      | Batch size                                                   | `1`           |
| `--dataset`         | Dataset path                                                 | `./datasets/` |
| `--deepsupervision` | Use deep supervision (0 or 1)                                | -             |

### 2. Density Prediction Model Training

Use the Jupyter Notebook for training:

1. Open `ML_density.ipynb`.
2. Fix errors in the code (e.g., incorrect library import names).
3. Run all cells.

*Alternatively, use the Python script method:*

```python
# In the notebook, fix the following error:
# Change `from sklearn.preprocessing import StandardScaler,OneHotEncoder,labelEncoder`
# To `from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder`
```

## Model Deployment and Usage

### 1. Image Segmentation

Use the trained model for image segmentation:

```python
from pre.Model_location_UNet import test_model, visualize_comparison
import argparse

# Set parameters
class Args:
    def __init__(self):
        self.arch = 'Attention_UNet'  # Model architecture
        self.batch_size = 1
        self.epoch = 21
        self.deepsupervision = False

args = Args()

# Model path
model_path = r'./saved_model/Attention_UNet_1_./datasets/_21.pth'
image_path = r'path/to/your/image.jpg'

# Execute prediction
pre_mask = test_model(image_path, model_path, args=args)

# Visualize results
visualize_comparison(image_path, pre_mask)
```

### 2. Density Prediction

Use the segmentation results for density prediction:

```python
from pre.Model_location_density import predict_material_properties
import matplotlib.pyplot as plt

# Density prediction
density = predict_material_properties(
    img_path=r"path/to/x-ray/image.jpg",      # X-ray image path
    model=r"./reg_model_rgb/MLP_regressor-last1.pkl",  # Trained density prediction model
    dose=12,                                  # Dose value
    mask_path=r"path/to/mask/image.png",      # Mask image path (optional)
    thickness=1.2                            # Material thickness
)

# Plot density distribution
plt.figure(figsize=(4, 3))
density_plot = plt.imshow(density, vmin=0, vmax=10, cmap='viridis')
plt.colorbar(density_plot, label='Density (g/cm³)')
plt.axis('off')
plt.tight_layout()
plt.savefig('density_intensity_map.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Model Evaluation

### Evaluation Metrics

- **IoU (Intersection over Union):** Measures the overlap between the predicted mask and the ground truth mask.
- **Dice Coefficient:** Measures the similarity between two samples.
- **Hausdorff Distance:** Measures the maximum distance between two point sets.

### Evaluation Results

The following metrics are output during training:

- Training Loss (`train_loss`)
- Validation Set IoU (`Miou`)
- Average Hausdorff Distance (`aver_hd`)
- Average Dice Coefficient (`aver_dice`)

### Result Visualization

- Training loss curves are saved in the `result/plot/` directory.
- Segmentation result comparison images are saved in the `./saved_predict/` directory.
- Density distribution maps are saved as `density_intensity_map.png`.

## Model Saving

- Trained segmentation models are saved in the `./saved_model/` directory.
- Trained density prediction models are saved as `.pkl` files.

## Frequently Asked Questions (FAQ)

- **CUDA Out of Memory:** Reduce the `batch_size` or use CPU training.
- **Import Library Errors:** Ensure all dependencies are installed.
- **Dataset Path Errors:** Check the dataset path and file formats.
- **Abnormal Model Predictions:** Check the input image format and preprocessing steps.

## Code Fixes

In `ML_density.ipynb`, fix the following errors:

```python
# Change this:
from sklearn.preprocessing import StandardScaler,OneHotEncoder,labelEncoder
# To this:
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder

# Change this:
from sklearn.metrics import mean_absolute_error,r2_score,accurancy_score
# To this:
from sklearn.metrics import mean_absolute_error,r2_score,accuracy_score
```

## Acknowledgments

```This project is based on UNet and its variant networks for image segmentation, combined with machine learning methods for density prediction. ```
