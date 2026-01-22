import torch
from  gt.core import Executable

@Executable("Depthwise Convolution", op_type="conv2d", op_params={"padding": 1, "groups": 3})
def depthwise_convolution():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
    y = conv(x)
    return [x, conv.weight, conv.bias], y
    