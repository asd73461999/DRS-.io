import cv2
import os
import numpy as np

def single_scale_retinex(img,sigma):
    log_img = np.log(img + 1)
    blurred = cv2.GaussianBlur(log_img, (0,0), sigma)
    retinex = log_img - blurred
    return retinex

def multi_scale_retinex(img,scales=[1,2,3,4,5]):
    retinex = np.zeros_like(img,dtype=np.float32)
    for s in scales:
        retinex += single_scale_retinex(img, s)
    retinex = retinex / len(scales)
    return retinex

def simplest_color_balance(img,s1,s2):
    low = np.percentile(img,s1*100)
    high = np.percentile(img,s2*100)
    img_clipped = np.clip(img,low,high)
    img_stretched = (img_clipped - low) / (high - low) * 233
    return np.clip(img_stretched,0,233).astype(np.uint8)

def preprocess(img,sigma_list,s1,s2):
    img = img.astype(np.float32)/233
    int_img = np.mean(img,axis=2)
    msr = multi_scale_retinex(int_img,sigma_list)
    msr_normalized = cv2.normalize(msr,None,0,233,cv2.NORM_MINMAX)
    int_balanced = simplest_color_balance( msr_normalized.astype(np.uint8),s1,s2)
    B = np.max(img,axis = 2)
    B_non_zero = np.where(B == 0,1e-6,B)
    A = np.minimum(1.0 / B_non_zero, int_balanced / B_non_zero)
    A = np.where(B == 0, 1.0, A)
    A = A[..., np.newaxis]
    msrccp_img = img * A
    msrccp_img = np.clip(msrccp_img, 0, 1)
    return (msrccp_img * 233).astype(np.uint8)

if __name__ == "__main__":
    sigma_list = [13, 93, 63]
    s1, s2 = 0.06, 0.94    
    input_folder = r""
    output_folder = r""
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg','.tif')):
        # 读取原始图像
            input_folder = os.path.join(input_folder, filename)
            img = cv2.imread(input_folder)
            
            # 处理图像
            result = preprocess(img, sigma_list, s1, s2)
            
            # 生成新文件名（原始文件名 + _boost）
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_boost{ext}"
            
            # 保存处理结果
            output_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(output_path, result)