import torch
import math
import numpy as np

def target_function1(x):
    """
    Generate the target function values.

    Args:
        x (torch.Tensor): Input tensor of values.

    Returns:
        torch.Tensor: Target function values.
    """
    return (torch.sin(x * (2 * math.pi)) + torch.randn(x.size()) * math.sqrt(0.04)).view(-1, 1)

def target_function2(x, y=1.2):
    """
    Generate the target function values, which has multiple 'optimal' points.

    Args:
        x (torch.Tensor): Input tensor of values.

    Returns:
        torch.Tensor: Target function values.
    """
    y = torch.tensor(y, dtype=x.dtype)  # Convert y to tensor
    out = (
        -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2)))
        - torch.exp(0.5 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y)))
        + torch.exp(torch.tensor(1.0)) + 20
    )
    return -out

def measure(x, noise=0.01):
    """
    Generate the target function values with some noise.

    Args:
        x (torch.Tensor): Input tensor of values.

    Returns:
        torch.Tensor: Target function values.
    """
    return target_function2(x) + noise * torch.randn(len(x))
