import torch
import torch.utils
import torch.utils.cpp_extension
print(torch.utils.cpp_extension.CUDA_HOME)
print(torch.cuda.is_available())