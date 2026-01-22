import torch
from  gt.core import Executable

@Executable("Padded Convolution", op_type="conv2d")
def padded_convolution():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    y = conv(x)
    return [x, conv.weight, conv.bias], y