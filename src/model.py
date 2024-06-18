import torch
import gpytorch

class FancyGPWithPriors(gpytorch.models.ExactGP):
    """
    A Gaussian Process model with priors on lengthscale and outputscale.

    Args:
        train_x (torch.Tensor): Training inputs.
        train_y (torch.Tensor): Training targets.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): Likelihood function.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(FancyGPWithPriors, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
        outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior
        )

        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean
        self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x):
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            gpytorch.distributions.MultivariateNormal: Multivariate normal distribution of predictions.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
