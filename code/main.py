import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from nets.resNet import resnet34_unet
from nets.Unet_pp import NestedUNet
import os
from matplotlib import pyplot as plt
from dataset import *
from metrix import *
from metrix import IOUMetric
from plot import *
from nets.attention_Unet import Attention_Unet
from nets.resNet import resnet34_unet
from nets.Unet_pp import  NestedUNet
import numpy as np
from PIL import Image
import cv2

#   returm parameters
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="train&test",help = "train/test/train&test")
    parser.add_argument("--arch",'-a',metavar="ARCH",default="unet++",help="resnet34_unet/unet++/Attention_UNet")
    parser.add_argument("--dataset",type=str,default='./datasets/',help="dataset path")
    parser.add_argument('--deepsupervision', default=0, type=int)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--log_dir", default='result/log', help="log dir")
    parser.add_argument("--threshold",type=float,default=None)
    args = parser.parse_args()
    return args  # 添加返回语句

# return args log
def getlog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = os.path.join(dirname,'log.txt')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(filename=filename,level=logging.DEBUG,format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    return logging

def getModel(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(num_channels=3, pretrained=False).to(device)
    elif args.arch == 'unet++':
        class Args:
            def __init__(self, deepsupervision):
                self.deep_supervision = bool(deepsupervision)
        temp_args = Args(args.deepsupervision)
        model = NestedUNet(temp_args, 3, 1).to(device)
    elif args.arch == 'Attention_UNet':
        model = Attention_Unet(3, 1).to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {args.arch}")
    return model

def getDataset(args):
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])
    
    y_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True))  # 将3通道转换为单通道
    ])
    
    train_dataset = tryPCB("train", transform=x_transforms, target_transform=y_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = tryPCB("val", transform=x_transforms, target_transform=y_transforms)
    val_dataloaders = DataLoader(val_dataset, batch_size=1)
    test_dataset = tryPCB("test", transform=x_transforms, target_transform=y_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=1)
    
    return train_dataloaders, val_dataloaders, test_dataloaders  # 添加返回语句


def val(model,best_iou,val_dataloaders):
    model= model.eval()
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        #print(num)
        for x, _,pic,mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            
            # 创建IOUMetric实例并使用其方法
            iou_metric = IOUMetric(2)
            hd_total += get_hd(mask[0], img_y)
            miou_total += iou_metric.get_iOu(mask[0], img_y)  #获取当前预测图的miou，并加到总miou中
            dice_total += get_dice(mask[0],img_y)
            if i < num:  # 修正条件判断逻辑
                i += 1   #处理验证集下一张图
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} &gt; best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} &gt; best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========&gt;save best model!')
            best_iou = aver_iou
            print('===========&gt;save best model!')
            save_dir = './saved_model'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_filename = str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth'
            model_path = os.path.join(save_dir, model_filename)
            model_parent_dir = os.path.dirname(model_path)
            if not os.path.exists(model_parent_dir):
                os.makedirs(model_parent_dir)
                
            torch.save(model.state_dict(), model_path)
        return best_iou,aver_iou,aver_dice,aver_hd
# train
def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_iou,aver_iou,aver_dice,aver_hd = 0,0,0,0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y, _, _ in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output.squeeze(1), labels.float())
                loss /= len(outputs)
            else:
                output = model(inputs)
                loss = criterion(output.squeeze(1), labels.float())
            if threshold!=None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        best_iou,aver_iou,aver_dice,aver_hd = val(model,best_iou,val_dataloader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        
        # test(test_dataloaders, save_predict=False, args=args)  # 使用当前训练的模型进行测试
        
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice',iou_list, dice_list)
    metrics_plot(args,'hd',hd_list)
    return model

# test
def test(test_dataloaders, save_predict=True, args=None):  # 修改参数列表
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        # 修改: 确保预测保存目录存在
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model_path = r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth'
    # 修改: 确保模型保存目录存在
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if os.path.exists(model_path):
        model = getModel(args)  # 创建新模型实例
        model.load_state_dict(torch.load(model_path, map_location='cpu'))  # 载入训练好的模型
        print("Loaded trained model from", model_path)
    else:
        print("Model file not found at", model_path)
        return
    model.eval()

    plt.ion() #开启动态模式
    with torch.no_grad():
        print('testing........')
        i=0   #验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(test_dataloaders)  #验证集图片的总数
        for pic,mask,pic_path,mask_path in test_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            iou_metric = IOUMetric(2)
            iou_metric.add_batch(predict.astype(np.int32), mask.squeeze().numpy().astype(np.int32))
            _, _, iou, _, _ = iou_metric.evaluate()
            # 修复: 确保iou是标量值
            if isinstance(iou, np.ndarray):
                iou = np.nanmean(iou)
            miou_total += iou  #获取当前预测图的miou，并加到总miou中
            hd_total += get_hd(mask.squeeze().numpy(), predict)
            dice = get_dice(mask.squeeze().numpy(), predict)
            dice_total += dice

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            print(pic_path[0])
            # 检查文件是否存在
            if os.path.exists(pic_path[0]):
                img = cv2.imread(pic_path[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax1.imshow(img)
                ax1.axis('off')

            else:
                print(f"Warning: File not found - {pic_path[0]}")
                # 创建一个空白图像占位
                ax1.imshow(np.zeros((256, 256, 3)), cmap='cool')
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            ax2.imshow(predict,cmap='cool')
            ax2.axis('off')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            ax3.axis('off')
            print(mask_path[0])
            # 检查文件是否存在
            if os.path.exists(mask_path[0]):
                mask_img = cv2.imread(mask_path[0], cv2.IMREAD_GRAYSCALE)
                ax3.imshow(mask_img, cmap='Grays_r')
            else:
                print(f"Warning: Mask file not found - {mask_path[0]}")
                # 创建一个空白图像占位
                ax3.imshow(np.zeros((256, 256)), cmap='Greys_r')
            if save_predict == True:
                if args.dataset == 'tryx':
                    mask_filename = os.path.basename(mask_path[0])
                    mask_name, _ = os.path.splitext(mask_filename)
                    saved_predict = os.path.join(dir, mask_name + '.tif')
                    # 将plt.savefig替换为cv2.imwrite
                    fig.canvas.draw()
                    image_array = np.array(fig.canvas.renderer._renderer)
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(saved_predict, image_bgr)
                else:
                    # mask_filename = os.path.basename(mask_path[0])
                    # saved_predict = os.path.join(dir, mask_filename)
                    # plt.savefig(saved_predict,bbox_inches='tight',dpi=300)
                    mask_filename = os.path.basename(mask_path[0])
                    mask_name, _ = os.path.splitext(mask_filename)
                    saved_predict = os.path.join(dir, mask_name + '.tif')
                    # 将plt.savefig替换为cv2.imwrite
                    fig.canvas.draw()
                    image_array = np.array(fig.canvas.renderer._renderer)
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(saved_predict, image_bgr)
            plt.show()  # 确保图像显示
            print('iou={},dice={}'.format(iou,dice))
            plt.close(fig)  # 关闭当前图像以释放内存
            if i < num:
                i += 1   #处理验证集下一张图

        print('Miou=%f,aver_hd=%f,dv=%f' % (miou_total/num,hd_total/num,dice_total/num))
        logging.info('Miou=%f,aver_hd=%f,dv=%f' % (miou_total/num,hd_total/num,dice_total/num))
        #print('M_dice=%f' % (dice_total / num))
        return model

if __name__ =="__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getlog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True, args=args)
