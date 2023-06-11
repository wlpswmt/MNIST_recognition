
import torch
import torch.nn as nn
#import torch.nn as nn
#from MNIST.pth import *
#import glob
import cv2
#import torch.nn.functional as F
#from torch.autograd import Variable
#from torchvision import datasets, transforms
import numpy as np
#import torchvision
#from skimage import io, transform
#from MNIST import *

from openvino.inference_engine import IECore,IENetwork
from time import time
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1：输入通道为 1，输出通道为 16，卷积核大小 为 5
        # 使用 Relu 激活函数
        # 使用最大值池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        # 卷积层2：输入通道为 16，输出通道为 32，卷积核大小 为 5
        # 使用 Relu 激活函数
        # 使用最大值池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                padding = 2
        ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        # 输出层，全连接层，输入大小 32 * 7 * 7， 输出大小 10
        self.layer_out = nn.Linear(32 * 7 * 7, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        self.out = self.layer_out(x)
        return self.out

 
ie = IECore()
for device in ie.available_devices:
    print(device)
DEVICE = 'CPU'#这里使用的CPU 但在树莓派上应该使用神经棒

model_xml = 'F:/VsCodeFiles/Pytorch/NumberRco/model50.xml'#文件地址要对
#'/Downloads/test2/main.xml'
model_bin = 'F:/VsCodeFiles/Pytorch/NumberRco/model50.bin'
#'/Downloads/test2/main.bin'


#读取IR模型文件
net = ie.read_network(model=model_xml, weights=model_bin)

#准备输入输出张量
print("Preparing input blobs")
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
net.batch_size = 1

#载入模型到AI推断计算设备
print("Loading IR to the plugin...")
exec_net = ie.load_network(network=net, num_requests=1, device_name=DEVICE)
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()  # 读取帧数 ret返回是否读取成功
    cv2.namedWindow('source',cv2.WINDOW_NORMAL) # 可以随意调整
    # 在视频方块上绘制矩形框 用于标定用于输入网络中的图像
    cv2.rectangle(frame,(220,150),(320,250),(0,0,255),2)
    # 展示原图
    # cv2.imshow("source", frame)
    source = frame
    # 图像灰度化处理
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)    # 转换为单通道

    frame = cv2.medianBlur(frame, 5)
    res, frame = cv2.threshold(frame, 90, 255, cv2.THRESH_BINARY_INV)   # 反二值化处
    roiImg = frame[150:220,250:320]  # 感兴趣的区域 使用索引进行拆分
    cv2.imshow("gray", frame)
    
    # 调整像素块的大小
    frame = cv2.resize(roiImg, (380, 380))
    cv2.imshow("28*28", roiImg)   # 展示我们感兴趣的区块
    cv2.waitKey(100)       # 施加延迟
    
    img = cv2.resize(roiImg, (28, 28))    # 将输入的图像转换成28*28大小
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 换成单通道
    img= torch.from_numpy(img)  # 转换成tensor 便于框架模型计算
    
    # 将矩形升维度 从而可以用来输入到网站用于计算
    testimg = torch.unsqueeze(img , dim=0)  
    testimg = torch.unsqueeze(testimg, dim=0)
    # 将图像尺寸转换成float32的类型  从而可以送进模型进行训练
   
    testimg = testimg.to(torch.float32)
    start = time()
    res = exec_net.infer(inputs={input_blob:testimg})
    end = time()
    print("Infer Time:{}s".format((end-start)))
    model = CNN()  # 初始化模型网络
    model.load_state_dict(torch.load('F:/VsCodeFiles/Pytorch/NumberRco/model50.pth')) 
    
    predimg = model(testimg)                           # 导入模型中进行计算
    _, pred = torch.max(predimg.data, dim=1)           # 将预测数据最大化
    org1 = (260,150)
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    fontScale1 = 2
    color1 = (0,0,255)
    thickness1 = 2
    cv2.putText(source,str(int(pred.data[0])),org1, font1, fontScale1, color1, thickness1, cv2.LINE_AA)
    cv2.imshow("source", source)
    print('the predict num is', int(pred.data[0]))     # 输出预测数字




# 打开摄像头


    

# image1 = cv2.imread(image_file,0)
# print(type(image1))

#执行推理

# image = cv2.resize(image1, (w, h))

# cv2.imshow("Detection results", )
# cv2.waitKey(0)
# cv2.destroyAllWindows()
   
