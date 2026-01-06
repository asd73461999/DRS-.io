import numpy as np
import cv2
from PIL import Image, ImageDraw

def calculate_iou_simple(mask1, mask2):
    """
    计算两个二值图像的交并比(IoU)
    
    参数:
    mask1: 第一个二值图像数组 (0表示背景, 1表示前景)
    mask2: 第二个二值图像数组 (0表示背景, 1表示前景)
    
    返回:
    iou: 交并比值
    """
    # 计算交集
    intersection = np.logical_and(mask1, mask2)
    
    # 计算并集
    union = np.logical_or(mask1, mask2)
    
    # 计算交集和并集的像素数量
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    
    # 避免除零错误
    if union_sum == 0:
        return 1.0  # 如果两个图像都为空，则IoU为1
    
    # 计算IoU
    iou = intersection_sum / union_sum
    return iou

def create_sample_masks():
    """
    创建示例图像用于演示
    """
    # 创建一个200x200的图像
    size = (200, 200)
    
    # 创建第一个掩码 - 圆形
    mask1 = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask1)
    draw.ellipse((50, 50, 150, 150), fill=255)
    mask1_array = np.array(mask1) // 255  # 转换为0/1值
    
    # 创建第二个掩码 - 矩形
    mask2 = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask2)
    draw.rectangle((100, 100, 180, 180), fill=255)
    mask2_array = np.array(mask2) // 255  # 转换为0/1值
    
    return mask1_array, mask2_array

def demonstrate_iou():
    """
    演示IoU计算过程
    """
    # 创建示例图像
    mask1, mask2 = create_sample_masks()
    
    # 计算IoU
    iou = calculate_iou_simple(mask1, mask2)
    
    print("交并比(IoU)计算示例:")
    print(f"掩码1的形状: {mask1.shape}")
    print(f"掩码2的形状: {mask2.shape}")
    print(f"掩码1中前景像素数: {np.sum(mask1)}")
    print(f"掩码2中前景像素数: {np.sum(mask2)}")
    
    # 计算交集和并集
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    print(f"交集像素数: {np.sum(intersection)}")
    print(f"并集像素数: {np.sum(union)}")
    print(f"IoU值: {iou:.4f}")
    
    # 可视化结果
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(mask1, cmap='gray')
        axes[0, 0].set_title('掩码1 (圆形)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask2, cmap='gray')
        axes[0, 1].set_title('掩码2 (矩形)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(intersection, cmap='gray')
        axes[1, 0].set_title('交集')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(union, cmap='gray')
        axes[1, 1].set_title('并集')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'交并比(IoU) = {iou:.4f}')
        plt.tight_layout()
        plt.savefig('iou_visualization.png')
        print("可视化结果已保存为 iou_visualization.png")
        plt.show()
    except ImportError:
        print("注意: 未安装matplotlib，无法显示可视化结果")

def calculate_iou_from_probability_map(prob_map, ground_truth, threshold=0.5):
    """
    从概率图计算IoU
    
    参数:
    prob_map: 概率图 (0-1之间的浮点数)
    ground_truth: 真实标签 (0/1二值图)
    threshold: 阈值，用于将概率图转换为二值图
    
    返回:
    iou: 交并比值
    """
    # 将概率图转换为二值图
    prediction = (prob_map > threshold).astype(np.int32)
    
    # 计算IoU
    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    
    if union_sum == 0:
        return 1.0
    
    iou = intersection_sum / union_sum
    return iou

def demonstrate_probability_iou():
    """
    演示从概率图计算IoU
    """
    # 创建示例真实标签
    gt = np.zeros((100, 100))
    gt[30:70, 30:70] = 1  # 正方形
    
    # 创建示例概率图 (模拟神经网络输出)
    prob_map = np.zeros((100, 100))
    # 在正方形区域内设置较高的概率
    prob_map[35:65, 35:65] = 0.9
    # 在边界区域设置中等概率
    prob_map[25:35, 25:75] = 0.6
    prob_map[65:75, 25:75] = 0.6
    prob_map[35:65, 25:35] = 0.6
    prob_map[35:65, 75:85] = 0.6
    
    # 使用不同阈值计算IoU
    thresholds = [0.3, 0.5, 0.7]
    
    print("\n从概率图计算IoU示例:")
    print(f"真实标签前景像素数: {np.sum(gt)}")
    print(f"概率图平均值: {np.mean(prob_map):.4f}")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 显示真实标签
        axes[0, 0].imshow(gt, cmap='gray')
        axes[0, 0].set_title('真实标签')
        axes[0, 0].axis('off')
        
        # 显示概率图
        im = axes[0, 1].imshow(prob_map, cmap='viridis')
        axes[0, 1].set_title('概率图')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # 显示不同阈值下的二值化结果和IoU
        for i, threshold in enumerate(thresholds):
            prediction = (prob_map > threshold).astype(np.int32)
            iou = calculate_iou_from_probability_map(prob_map, gt, threshold)
            
            axes[0, i+2].imshow(prediction, cmap='gray')
            axes[0, i+2].set_title(f'阈值={threshold}\n预测结果')
            axes[0, i+2].axis('off')
            
            # 显示交集和并集
            intersection = np.logical_and(prediction, gt)
            union = np.logical_or(prediction, gt)
            
            axes[1, i*2].imshow(intersection, cmap='gray')
            axes[1, i*2].set_title(f'阈值={threshold}\n交集')
            axes[1, i*2].axis('off')
            
            axes[1, i*2+1].imshow(union, cmap='gray')
            axes[1, i*2+1].set_title(f'阈值={threshold}\n并集, IoU={iou:.4f}')
            axes[1, i*2+1].axis('off')
            
        plt.tight_layout()
        plt.savefig('probability_iou.png')
        print("概率图IoU计算结果已保存为 probability_iou.png")
        plt.show()
        
    except ImportError:
        print("注意: 未安装matplotlib，无法显示可视化结果")
        
        # 只计算数值结果
        for threshold in thresholds:
            iou = calculate_iou_from_probability_map(prob_map, gt, threshold)
            print(f"阈值={threshold}时, IoU={iou:.4f}")

if __name__ == "__main__":
    print("=== 交并比(IoU)计算示例 ===\n")
    
    # 演示基本IoU计算
    demonstrate_iou()
    
    # 演示从概率图计算IoU
    demonstrate_probability_iou()
    
    print("\n=== 程序执行完成 ===")