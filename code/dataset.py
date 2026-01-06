import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio

class tryPCB(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r"D:\U-net\DRS--main\DRS--main\code\datasets\tryx"
        self.pics, self.masks = self.getDataPath()
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'\training\images\*')
        self.train_mask_paths = glob(self.root + r'\training\1st_manual\*')
        self.val_img_paths = glob(self.root + r'\test\images\*')
        self.val_mask_paths = glob(self.root + r'\test\1st_manual\*')
        self.test_img_paths = self.val_img_paths
        self.test_mask_paths = self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths
    def adaptive_resize(self, image, target_size):
        # Maintain the aspect ratio for scaling
        h, w = image.shape[:2]
        scale = min(target_size[0]/h, target_size[1]/w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Edge filling
        delta_h = target_size[0] - new_h
        delta_w = target_size[1] - new_w
        top, bottom = delta_h//2, delta_h - (delta_h//2)
        left, right = delta_w//2, delta_w - (delta_w//2)
        
        return cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=0)
    def __getitem__(self, index):
        # imgx,imgy=(576,576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        #print(pic_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]
        # pic = cv2.resize(pic,(imgx,imgy))
        pic = self.adaptive_resize(pic,(576, 576))
        
        # mask = cv2.resize(mask, (imgx, imgy))
        mask = self.adaptive_resize(mask, (576, 576))
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # 确保mask是单通道的
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.aug:
            if random.uniform(0, 1) > 0.5:
                pic = pic[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                pic = pic[::-1, :, :].copy()
                mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        # return img_x, img_y,pic_path,mask_path
        return img_x, img_y.squeeze(), pic_path, mask_path

    def __len__(self):
        return len(self.pics)
