import argparse
import logging
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from nets.resNet import resnet34_unet
from nets.Unet_pp import NestedUNet
from nets.attention_Unet import Attention_Unet
from PIL import Image
from torch.utils.data import DataLoader

# Ensure proper font display
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]


def get_iou(mask, pred):
    """
    Calculate IoU metric
    """
    intersection = np.logical_and(mask, pred)
    union = np.logical_or(mask, pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def get_dice(mask, pred):
    """
    Calculate Dice coefficient
    """
    intersection = np.logical_and(mask, pred)
    dice = 2 * np.sum(intersection) / (np.sum(mask) + np.sum(pred))
    return dice


def adaptive_resize(image, target_size):
    """
    Resize image while maintaining aspect ratio and perform edge padding
    """
    h, w = image.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Edge padding
    delta_h = target_size[0] - new_h
    delta_w = target_size[1] - new_w
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def getModel(args):
    """
    Get model based on parameters
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(num_channels=3, pretrained=False).to(device)
    elif args.arch == 'unet++':
        args.deepsupervision = True
        model = NestedUNet(args, 3, 1).to(device)
    elif args.arch == 'Attention_UNet':
        model = Attention_Unet(3, 1).to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {args.arch}")
    
    return model


def test_model(image_path, model_path, mask_path=None, args=None):
    """
    Test model and return prediction results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = getModel(args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = cv2.imread(image_path)
    image = adaptive_resize(image, (576, 576))
    image = image.astype('float32') / 255
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'forward') and len(list(model.forward.__code__.co_varnames)) > 1:
            pred = model(tensor)[0]
        else:
            pred = model(tensor)
        
    pred = pred.squeeze().cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8) * 255
    
    if mask_path is not None:
        mask = cv2.imread(mask_path, 0)
        mask = adaptive_resize(mask, (576, 576))
        mask = (mask > 127).astype(np.uint8) * 255
        
        iou = get_iou(mask, pred)
        dice = get_dice(mask, pred)
        print(f'IoU: {iou:.4f}, Dice: {dice:.4f}')
    
    return pred


def visualize_comparison(image_path, pred_mask, save_dir=r'./results'):
    """
    Visualize comparison between original image and predicted mask
    
    Parameters:
        image_path: Path to original image
        pred_mask: Predicted mask matrix (0-255 range)
        save_dir: Directory to save results
    """
    try:
        raw_img = cv2.imread(image_path)
        processed_img = adaptive_resize(raw_img, (576, 576))
        
        pred_mask_normalized = pred_mask.astype(np.float32) / 255

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(pred_mask_normalized, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        plt.imshow(pred_mask_normalized, cmap='jet', alpha=0.3)
        plt.title('Overlay Effect')
        plt.axis('off')
        
        base_name = os.path.basename(image_path).split('.')[0]
        save_path = os.path.join(save_dir, f'{base_name}_comparison.png')
        
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        cv2.imwrite(os.path.join(save_dir, f'{base_name}_processed.png'), processed_img)
        cv2.imwrite(os.path.join(save_dir, f'{base_name}_mask.png'), pred_mask)
        
        plt.close('all')
        print(f"Visualization results saved to {save_dir}")
        
    except Exception as e:
        print(f'Visualization failed: {str(e)}')
        plt.close('all')


class IOUMetric:
    """
    Tool class for calculating IoU
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc, iu


if __name__ == "__main__":
    # Parameter settings
    class Args:
        def __init__(self):
            self.arch = 'Attention_UNet'
            self.dataset = 'tryPCB'
            self.batch_size = 1
            self.epoch = 21
            self.deepsupervision = False
    
    args = Args()
    
    # Set model, image, and mask paths
    model_path = r'C:\Users\admin\BaiduSyncdisk\u-net_model\Attention_UNet_3_driveEye_21.pth'
    image_path = r'C:\Users\admin\BaiduSyncdisk\picture_apply\model_process\pic4\IMG_8303_boost1.JPG'
    mask_path = r'J:\BaiduNetdiskDownload\UNets-master\datasets\tryx\test\1st_manual\edges_19.tif'
    
    # Perform test and visualization
    if os.path.exists(model_path):
        pre_mask = test_model(image_path, model_path, mask_path=mask_path, args=args)
        visualize_comparison(image_path, pre_mask)
    else:
        print(f"Model file not found: {model_path}")
