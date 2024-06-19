import torch
import math
import matplotlib.pyplot as plt
from target_function import target_function2
from target_function import measure
from model import FancyGPWithPriors
from optimizer import get_next_points
from test_parameters import Test_parameters

X_bounds = torch.tensor([-2.0, 2.0])
train_x = torch.empty(2).uniform_(X_bounds[0], X_bounds[1])
train_y = measure(train_x, noise = 0.01)

test_x = torch.linspace(-2, 2, 200)
test_y = measure(test_x, noise = 0.01)
# noise, lengthscale, outputscale = 0.01, 0.05, 0.1
# noise, lengthscale, outputscale = 12., 0.5, 2.


def main():
    """
    Main function to run the GP model training and optimization.

    Args:
        None

    Returns:
        None
    """
    # Parameters for the GP model
    noise = 0.01
    lengthscale = 0.05
    outputscale = 0.1
    
    # Run the Test_parameters function with specified parameters
    Test_parameters(train_x, train_y, noise, lengthscale, outputscale)

if __name__ == "__main__":
    main()
