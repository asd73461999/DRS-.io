import matplotlib.pyplot as plt
import os
def loss_plot(args,loss):
    num = args.epoch
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    # 修复: 确保路径中的特殊字符不会导致文件名无效
    safe_arch = str(args.arch).replace('/', '_').replace('\\', '_')
    safe_dataset = str(args.dataset).replace('/', '_').replace('\\', '_')
    save_loss = plot_save_path + safe_arch + '_' + str(args.batch_size) + '_' + safe_dataset + '_' + str(args.epoch) + '_loss.jpg'
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)
    plt.close()  # 关闭图形以释放内存

def metrics_plot(arg,name,*args):
    num = arg.epoch
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    # 修复: 确保路径中的特殊字符不会导致文件名无效
    safe_arch = str(arg.arch).replace('/', '_').replace('\\', '_')
    safe_dataset = str(arg.dataset).replace('/', '_').replace('\\', '_')
    save_metrics = plot_save_path + safe_arch + '_' + str(arg.batch_size) + '_' + safe_dataset + '_' + str(arg.epoch) + '_'+name+'.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)
    plt.close()  # 关闭图形以释放内存