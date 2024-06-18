import torch
import math

def target_function(x):
    """
    Generate the target function values.

    Args:
        x (torch.Tensor): Input tensor of values.

    Returns:
        torch.Tensor: Target function values.
    """
    return (torch.sin(x * (2 * math.pi)) + torch.randn(x.size()) * math.sqrt(0.04)).view(-1, 1)