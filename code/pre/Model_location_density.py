# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
np.set_printoptions(threshold=np.inf)
def predict_material_properties(img_path, model, dose=12, thickness=1.0, mask_path=None):
    """
    Predict material density distribution
    
    Parameters:
    img_path: Input image path
    model: Model file path
    dose: Dose value (default 12)
    thickness: Thickness value (default 1.0)
    mask_path: Mask image path (optional)
    
    Returns:
    density_map: Density distribution matrix
    """
    # Load pre-trained model
    model = joblib.load(model)
    
    # Read image and convert color space
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create mask
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask image: {mask_path}")
        mask = cv2.resize(mask, (width, height))
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    else:
        mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Get pixel coordinates of masked area
    y_idx, x_idx = np.where(mask == 255)
    
    # Separate RGB channels
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    
    # Create full-size parameter matrices
    dose_matrix = np.full((height, width), fill_value=dose, dtype=np.float32)
    thickness_matrix = np.full((height, width), fill_value=thickness, dtype=np.float32)
    
    # Extract features from masked area
    r_vals = r[y_idx, x_idx]
    g_vals = g[y_idx, x_idx]
    b_vals = b[y_idx, x_idx]
    dose_vals = dose_matrix[y_idx, x_idx]
    thickness_vals = thickness_matrix[y_idx, x_idx]
    
    # Modify feature matrix construction part
    import pandas as pd
    
    # Build DataFrame with column names
    X_predict = pd.DataFrame({
        'R': r_vals,
        'G': g_vals,
        'B': b_vals,
        'Dose': dose_vals,
        'Thickness': thickness_vals
    })
    
    # Use model to predict
    predictions = model.predict(X_predict)
    
    # Create result matrix
    density_map = np.zeros((height, width), dtype=np.float32)
    #Create Gaussian noise matrix
    noise = np.random.normal(0.86, 0.015, density_map.shape)

    # Fill predicted values into result matrix
    density_map[y_idx, x_idx] = predictions
    # Add Gaussian noise
    density_map *= noise
    # Post-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))#Kernel size, do not change
    density_map = cv2.morphologyEx(density_map, cv2.MORPH_CLOSE, kernel)
    arr = density_map[density_map != 0]
    print(np.median(arr)/t "g/cm³")
    return density_map

# Usage example
density = predict_material_properties(
    img_path=r"C:\Users\1\BaiduSyncdisk\picture_apply\model_process\pic_LED\Multi-Mask Result_screenshot_25.06.2025_2.png", # Import X-ray imaging picture at this step
    model=r".\reg_model_rgb\MLP_regressor-last1.pkl", # Import trained model at this step
    dose=12,
    mask_path= r"C:\Users\1\BaiduSyncdisk\picture_apply\model_process\pic_LED\mask\ls\IMG_8129_boost_mask_base.png",
    thickness=1.2, # Input dose and material thickness
     # Import U-net++ model returned image mask at this step to automatically remove irrelevant parts from X-ray imaging picture
)

# Plot
plt.figure(figsize=(4, 3))
density_plot = plt.imshow(density, vmin=0, vmax=10,cmap='viridis')
plt.colorbar(density_plot, label='Density (g/cm³)')
plt.axis('off')
plt.tight_layout()
plt.savefig('density_intensity_map.png', dpi=300, bbox_inches='tight')
plt.show()