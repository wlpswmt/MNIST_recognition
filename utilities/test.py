import torch

print(torch.backends.cudnn.version()) #查看cudnn版本号
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("GPU 编号: {}".format(device))
