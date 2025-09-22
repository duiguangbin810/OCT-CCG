# import torch
# print(torch.__version__)  # 确认是否为2.3.0
# print(torch.__file__)     # 查看实际调用的PyTorch安装路径

import torch
print("PyTorch版本:", torch.__version__)
print("torch.library中的属性:", dir(torch.library))
print("是否存在custom_op:", hasattr(torch.library, "custom_op"))