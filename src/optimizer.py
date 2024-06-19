import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

def get_next_points(init_x, init_y, best_init_y, model, likelihood, bounds, batch_size):
    """
    Optimize the acquisition function to get the next points for evaluation.

    Args:
        init_x (torch.Tensor): Initial training inputs.
        init_y (torch.Tensor): Initial training targets.
        best_init_y (float): Best initial target value.
        model (FancyGPWithPriors): Gaussian Process model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): Likelihood function.
        bounds (torch.Tensor): Bounds for the input space.
        batch_size (int): Number of points to sample in each iteration.

    Returns:
        tuple: New candidate points, acquisition values, updated model, and acquisition function.
    """
    model_bo = SingleTaskGP(
        train_X=init_x, train_Y=init_y,
        covar_module=model.covar_module,
        likelihood=likelihood,
        input_transform=Normalize(d=1),
        outcome_transform=Standardize(m=1)
    )

    best_value = best_init_y
    # For analytical, it can change to minimize problem
    EI = ExpectedImprovement(model_bo, best_f=best_value, maximize=True)
    
    new_point_mc, ac_values = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=batch_size,
        num_restarts=20,
        raw_samples=100,
        options={},
    )

    return new_point_mc, ac_values, model_bo, EI
