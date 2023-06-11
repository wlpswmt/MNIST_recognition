
# 将模型格式转换成onnx格式 从而可以使用openvino进行模型推理 

import torch


model = torch.load("F:/VsCodeFiles/Pytorch/NumberRco/model50.pth")  # 加载模型

dummy_input = torch.randn(1,1,28,28)
torch.onnx.export(model,(dummy_input),'best.onnx')