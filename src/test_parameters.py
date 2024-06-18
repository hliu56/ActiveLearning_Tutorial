import torch
import gpytorch
import time
import matplotlib.pyplot as plt
from model import FancyGPWithPriors
from optimizer import get_next_points
from target_function import target_function

def Test_parameters(train_x, train_y, noise, lengthscale, outputscale):
    """
    Train the GP model and test parameters.

    Args:
        noise (float): Noise level for the GP model.
        lengthscale (float): Lengthscale for the GP kernel.
        outputscale (float): Outputscale for the GP kernel.

    Returns:
        None
    """
    print('Initial model parameters')
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
    )
    
    model = FancyGPWithPriors(train_x, train_y, likelihood)
    
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(noise),
        'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
        'covar_module.outputscale': torch.tensor(outputscale),
    }
    
    model.initialize(**hypers)
    print(
        model.likelihood.noise_covar.noise.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.covar_module.outputscale.item()
    )
    
    model_init_p1 = round(model.likelihood.noise_covar.noise.item(), 2)
    model_init_p2 = round(model.covar_module.base_kernel.lengthscale.item(), 1)
    model_init_p3 = round(model.covar_module.outputscale.item(), 1)
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        observed_pred_untrain = likelihood(model(test_x))
    
    with torch.no_grad():
        lower_init, upper_init = observed_pred_untrain.confidence_region()
    
    training_iter = 50
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
    
    with torch.no_grad():  
        lower, upper = observed_pred.confidence_region()
    
    init_x = train_x.unsqueeze(1)
    init_y = train_y.unsqueeze(1)
    best_init_y = train_y.min().item()
    bounds = torch.tensor([[0.],[1.]])
    
    start_time = time.time()
    
    n_runs = 10
    
    for i in range(n_runs):
        print(f'Iterations: {i}')
        new_candidates, ac_values, model_final, ac_func = get_next_points(init_x, init_y, best_init_y, model, likelihood, bounds, batch_size=1)
        new_results = target_function(new_candidates)
        print(f'New candidates: {new_candidates}')
        init_x = torch.cat([init_x, new_candidates])
        init_y = torch.cat([init_y, new_results])
    
        best_init_y = init_y.min().item()
        print(best_init_y)
    
    end_time = time.time()
    running_time = end_time - start_time
    
    print("Script execution time: {:.2f} min".format(round(running_time/60, 2)))
    
    xx = torch.linspace(0, 1, 200).unsqueeze(-1)
    x = init_x
    y = init_y
    
    with torch.no_grad():
        posterior = model_final.posterior(X=xx.unsqueeze(1))
        
    ymean, yvar = posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)
    eci_vals = ac_func(xx.unsqueeze(1))
    
    fig, axes = plt.subplots(1, 4, figsize=(28, 5))
    
    ax = axes[0]
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    ax.plot(test_x.numpy(), observed_pred_untrain.mean.numpy(), 'b')
    ax.fill_between(test_x.numpy(), lower_init.numpy(), upper_init.numpy(), alpha=0.1)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'noise_level: {model_init_p1}, lengthscale: {model_init_p2}, outputscale: {model_init_p3}')
    
    ax = axes[1]
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.1)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'noise_level: {round(model.likelihood.noise_covar.noise.item(), 2)}, lengthscale: {round(model.covar_module.base_kernel.lengthscale.item(), 1)}, outputscale: {round(model.covar_module.outputscale.item(), 1)}')
    
    ax = axes[2]
    ax.plot(xx[:, 0].cpu(), ymean[:, 0].cpu(), "b")
    ax.fill
