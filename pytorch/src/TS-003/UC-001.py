import torch
import torch.nn as nn
from  gt.core import Executable


@Executable("MNIST flatten", op_type="flatten", op_params={"start_dim": 1, "end_dim": -1})
def image_flatten():
    flatten = nn.Flatten()  # default start_dim=1, end_dim=-1
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    y = flatten(x)
    print(f"Shape {y.shape}") #(2, 784)
    return [x], y

@Executable("Flatten with with custom start dim", op_type="flatten", op_params={"start_dim": 1, "end_dim": -1})
def flatten_with_custom_start_dim():
    flatten = nn.Flatten(start_dim=1)
    input_tensor = torch.randn(2, 3, 4)
    output_tensor = flatten(input_tensor)
    print(output_tensor.shape)  # 3*4 = 12
    return [input_tensor], output_tensor



@Executable("Flatten single sample", op_type="flatten", op_params={"start_dim": 1, "end_dim": -1})
def flatten_single_sample():
    flatten = nn.Flatten()  # default start_dim=1, end_dim=-1
    input_tensor = torch.randn(1, 3, 3)
    output_tensor = flatten(input_tensor)
    print(output_tensor.shape)  # (1, 9)
    return [input_tensor], output_tensor


@Executable("Flatten preserve batch dim", op_type="flatten", op_params={"start_dim": 1, "end_dim": -1})
def flatten_preserve_batch_dim():
    flatten = nn.Flatten()  # default start_dim=1, end_dim=-1
    input_tensor = torch.randn(10, 5, 2, 2)
    output_tensor = flatten(input_tensor)
    # self.assertEqual(output_tensor.shape, (10, 20))  # 5*2*2 = 20
    return [input_tensor], output_tensor

@Executable("Flatten no batch dim", op_type="flatten", op_params={"start_dim": 1, "end_dim": -1})
def flatten_no_batch_dim():
    flatten = nn.Flatten()  # default start_dim=1, end_dim=-1
    input_tensor = torch.randn(3, 4)
    output_tensor = flatten(input_tensor)
    # self.assertEqual(output_tensor.shape, (3, 4))  # with start_dim=1, 2D input stays same
    return [input_tensor], output_tensor


  