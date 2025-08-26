import torch
print(torch.__version__)  # 输出 PyTorch 版本
print(torch.version.cuda)  # 输出 PyTorch 编译时使用的 CUDA 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用（True/False）


from peft import LoraModel 
# 原始模块
import torch.nn as nn
# orig_linear = nn.Linear(10, 5)

# # 包装成 ModulesToSaveWrapper
# wrapped = ModulesToSaveWrapper(orig_linear)

# # 访问原始模块
# print(wrapped.base_module)  # 原来的 nn.Linear(10, 5)

t = torch.tensor([[True, False], [False, True]])
print(~t)
