import torch
import math
import matplotlib.pyplot as plt
from target_function import target_function
from model import FancyGPWithPriors
from optimizer import get_next_points
from test_parameters import Test_parameters

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = -torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

test_x = torch.linspace(0, 1, 51)
test_y = -torch.sin(test_x * (2 * math.pi)) + torch.randn(test_x.size()) * math.sqrt(0.04)
noise, lengthscale, outputscale = 12., 0.5, 2.

def main():
    """
    Main function to run the GP model training and optimization.

    Args:
        None

    Returns:
        None
    """
    # Parameters for the GP model
    noise = 12.0
    lengthscale = 0.5
    outputscale = 2.0
    
    # Run the Test_parameters function with specified parameters
    Test_parameters(train_x, train_y, noise, lengthscale, outputscale)

if __name__ == "__main__":
    main()
