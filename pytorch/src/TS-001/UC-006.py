import torch
from  gt.core import Executable

@Executable("Batched Input", op_type="conv2d")
def batched_input():
    x = torch.randn(2, 3, 32, 32, requires_grad=True)  # Batch size = 2
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
    y = conv(x)
    return [x, conv.weight, conv.bias], y