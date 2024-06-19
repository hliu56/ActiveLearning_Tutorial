import os
import torch
import numpy as np
import gpytorch
import time
import matplotlib.pyplot as plt
from model import FancyGPWithPriors
from optimizer import get_next_points
from target_function import target_function2
from target_function import measure

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
        test_x = torch.linspace(-2, 2, 200)
        observed_pred_untrain = likelihood(model(test_x))

    with torch.no_grad():
        lower_init, upper_init = observed_pred_untrain.confidence_region()

    training_iter = 20

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
    best_init_y = train_y.max().item()
    bounds = torch.tensor([[-2.],[2.]])

    start_time = time.time()

    num_steps = 50  # number of Bayesian optimization steps

    # Ensure the Results folder exists
    os.makedirs('Results', exist_ok=True)

    for e in range(num_steps):
        print("\nStep {}/{}".format(e + 1, num_steps))
        
        # Bayesian optimization step
        new_candidates, ac_values, model_final, ac_func = get_next_points(init_x, init_y, best_init_y, model, likelihood, bounds, batch_size=1)
        new_results = measure(new_candidates)
        
        init_x = torch.cat([init_x, new_candidates])
        init_y = torch.cat([init_y, new_results])

        best_init_y = init_y.max().item()
        print("Best y value:", best_init_y)
        
        # Generate plot for this iteration
        xx = torch.linspace(-2, 2, 200).unsqueeze(-1)
        x = init_x.numpy()
        y = init_y.numpy()
        
        with torch.no_grad():
            posterior = model_final.posterior(X=xx.unsqueeze(1))
            
        ymean = posterior.mean.squeeze(-1).numpy()
        yvar = posterior.variance.squeeze(-1).numpy()
        acq = ac_func(xx.unsqueeze(1)).detach().numpy()
        
        # Get bounds for uncertainty
        lower_b = ymean - 1.96 * np.sqrt(yvar)
        upper_b = ymean + 1.96 * np.sqrt(yvar)
        
        # Compute true function values
        true_function_values = target_function2(xx).numpy()
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(14, 5.5))
        
        ax1.scatter(x[:-1], y[:-1], marker='x', c='k', label="Observations", s=64)
        ax1.fill_between(xx.squeeze().numpy(), lower_b.squeeze(), upper_b.squeeze(), color='r', alpha=0.3, label="Model uncertainty")
        ax1.plot(xx.squeeze().numpy(), ymean, lw=2, c='b', label='Posterior mean')
        ax1.plot(xx.squeeze().numpy(), true_function_values, lw=2, c='g', label='True function')
        ax1.set_xlabel("$X$", fontsize=16)
        ax1.set_ylabel("$y$", fontsize=16)
        ax1.legend(loc='best', fontsize=10)
        ax1.set_title(f'Step {e + 1}')
        
        idx = acq.argmax()
        ax2.plot(xx.squeeze().numpy(), 200*acq, lw=2, c='orangered', label='Acquisition function')
        ax2.scatter(xx.squeeze().numpy()[idx], acq[idx], s=70, c='orangered', label='Next point to measure')
        ax2.plot(xx.squeeze().numpy(), ymean, lw=2, c='b', label='Posterior mean')
        ax2.plot(xx.squeeze().numpy(), true_function_values, lw=2, c='g', label='True function')
        ax2.set_xlabel("$X$", fontsize=16)
        ax2.set_ylabel("Acquisition Value", fontsize=16)
        ax2.legend(loc='best', fontsize=10)
        ax2.set_title(f'Step {e + 1}')
        
        plt.savefig(f'Results/iteration_{e + 1}.png')
        # plt.show()
        plt.close(fig)  # Close the figure to free up memory

    end_time = time.time()
    running_time = end_time - start_time

    print("Script execution time: {:.2f} min".format(round(running_time/60, 2)))

    # Final plot after all iterations
    xx = torch.linspace(-2, 2, 200).unsqueeze(-1)
    x = init_x.numpy()
    y = init_y.numpy()

    with torch.no_grad():
        posterior = model_final.posterior(X=xx.unsqueeze(1))
        
    ymean = posterior.mean.squeeze(-1).numpy()
    yvar = posterior.variance.squeeze(-1).numpy()
    acq = ac_func(xx.unsqueeze(1)).detach().numpy()

    # Get bounds for uncertainty
    lower_b = ymean - 1.96 * np.sqrt(yvar)
    upper_b = ymean + 1.96 * np.sqrt(yvar)

    # Compute true function values
    true_function_values = target_function2(xx).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(14, 5.5))

    ax1.scatter(x, y, marker='x', c='k', label="Observations", s=64)
    ax1.fill_between(xx.squeeze().numpy(), lower_b.squeeze(), upper_b.squeeze(), color='r', alpha=0.3, label="Model uncertainty")
    ax1.plot(xx.squeeze().numpy(), ymean, lw=2, c='b', label='Posterior mean')
    ax1.plot(xx.squeeze().numpy(), true_function_values, lw=2, c='g', label='True function')
    ax1.set_xlabel("$X$", fontsize=16)
    ax1.set_ylabel("$y$", fontsize=16)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_title('Final Iteration')

    idx = acq.argmax()
    ax2.plot(xx.squeeze().numpy(), 200*acq, lw=2, c='orangered', label='Acquisition function')
    ax2.scatter(xx.squeeze().numpy()[idx], acq[idx], s=70, c='orangered', label='Next point to measure')
    ax2.plot(xx.squeeze().numpy(), ymean, lw=2, c='b', label='Posterior mean')
    ax2.plot(xx.squeeze().numpy(), true_function_values, lw=2, c='g', label='True function')
    ax2.set_xlabel("$X$", fontsize=16)
    ax2.set_ylabel("Acquisition Value", fontsize=16)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_title('Final Iteration')

    plt.savefig('Results/final_iteration.png')
    # plt.show()
    plt.close(fig)
