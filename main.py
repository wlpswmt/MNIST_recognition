import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
# 分批次训练，一批 100 个训练数据
BATCH_SIZE = 100
# 所有训练数据训练 3 次
# 学习率设置为 0.0001
LEARN_RATE = 0.0001
 
# 架加载数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()  # 将下载的文件转换成pytorch认识的tensor类型，且将图片的数值大小从（0-255）归一化到（0-1）
)
 
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
 
test_data = torchvision.datasets.MNIST(
    root='./dataset/mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()  # 将下载的文件转换成pytorch认识的tensor类型，且将图片的数值大小从（0-255）归一化到（0-1）
)
 
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle = True)
 
 
# 定义CNN神经网络
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
# 实例化CNN，并将模型放在 GPU 上训练
model = CNN().to(device)
# 使用交叉熵损失，同样，将损失函数放在 GPU 上
loss_fn = nn.CrossEntropyLoss().to(device)
# 使用 Adam 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
 
def train(epoch):
    running_loss = 0.0        #每一轮训练重新记录损失值
    for batch_idx, data in enumerate(train_loader, 0):    #提取训练集中每一个样本
        inputs, target = data        
        inputs, target = inputs.to(device), target.to(device)  # 这里的数据（原数据）也要迁移到GPU
        # outputs输出为0-9的概率  256*10
        outputs = model(inputs)              #代入模型
        loss = loss_fn(outputs, target)    #计算损失值
        loss.backward()                      #反向传播计算得到每个参数的梯度值
        optimizer.step()                     #梯度下降参数更新
        optimizer.zero_grad()                #将梯度归零
        running_loss += loss.item()          #损失值累加
    
        if batch_idx % 300 == 299:           #每300个样本输出一下结果
            print('[%d,%5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0          # (训练轮次，  该轮的样本次，  平均损失值)
    return running_loss

def test():
    correct = 0
    total = 0
    with torch.no_grad():            #执行计算，但不希望在反向传播中被记录
        for data in test_loader:     #提取测试集中每一个样本
            images, labels = data
            images, labels = images.to(device), labels.to(device)# 这里的数据（原数据）也要迁移到GPU
            # outputs输出为0-9的概率  256*10
            outputs = model(images)  #带入模型
            # torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示）
            # 第二个值是value所在的index（也就是predicted）
            _, pred = torch.max(outputs.data, dim=1)    #获得结果中的最大值
            total += labels.size(0)                     #测试数++
            correct += (pred == labels).sum().item()    #将预测结果pred与标签labels对比，相同则正确数++
        print('%d %%' % (100 * correct / total))    #输出正确率
        
        

# torch.save(model, './MNIST.pth')

# #模型转换
# model.eval()
# dummy_input = torch.randn(1,1,28,28)
# torch.onnx.export(model,(dummy_input),'model.onnx')
    # 这两个数组主要是为了画图
lossy = []        #定义存放纵轴数据（损失值）的列表
epochx = []       #定义存放横轴数据（训练轮数）的列表
    
for epoch in range(100):    #训练 n 轮 可以自己进行 修改 和 调整
    epochx.append(epoch)   #将本轮轮次存入epochy列表
    lossy.append(train(epoch))  #执行训练，将返回值loss存入lossy列表  
    test()                 #每轮训练完都测试一下正确率
path = "F:/VsCodeFiles/Pytorch/NumberRco/model100.pth"
    #torch.save(model,path)
torch.save(model.state_dict(),path)   # 保存模型
# model.eval()
# dummy_input = torch.randn(1,1,28,28)
# torch.onnx.export(model,(dummy_input),'best.onnx')
# model = torch.load("F:/VsCodeFiles/Pytorch/NumberRco/model20.pth")  # 加载模型
 

# 模型转换为.onnx形式


#     #可视化一下训练过程
# plt.plot(epochx, lossy)
# plt.grid()
# plt.show()