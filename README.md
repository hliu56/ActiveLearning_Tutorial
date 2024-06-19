# ActiveLearning_Tutorial

## Table of Contents
1. [Foundations](#foundations)
    - [Gaussian Process Model](#gaussian-process-model)
    - [Kernel Function](#kernel-function)
    - [Acquisition Function](#acquisition-function)
2. [Installation](#installation)
3. [Usage](#usage)

## Foundations

### Gaussian Process Model

A Gaussian Process (GP) is a collection of random variables, any finite number of which have a joint Gaussian distribution. It is a powerful tool for regression and classification tasks in machine learning.

**Bayes' Rule:**

Bayes' rule is fundamental in probabilistic models, including Gaussian Processes. It allows us to update the probability estimate for a hypothesis as more evidence or information becomes available.

**Formulation:**

A Gaussian Process is fully specified by its mean function $\( m(\mathbf{x}) \)$ and covariance function $\( k(\mathbf{x}, \mathbf{x}') \)$:

$$
m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]
$$

$$
k(\mathbf{x}, \mathbf{x}') = \mathbb{E}[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}'))]
$$

Given a dataset $D = \{(\mathbf{x}_i, y_i) | i = 1, \ldots, n\}$, where $\mathbf{x}_i$ is the input and $y_i$ is the output, the goal is to predict the output $y_*$ at a new input $\mathbf{x}_*$.

Given a dataset $\( D = \{(\mathbf{x}_i, y_i) | i = 1, \ldots, n\} \)$, where $\( \mathbf{x}_i \)$ is the input and $\( y_i \)$ is the output, the goal is to predict the output $\( y_* \)$ at a new input $\( \mathbf{x}_* \)$.

**References:**
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. The MIT Press.

### Kernel Function

The kernel function (or covariance function) defines the covariance between pairs of random variables. It encodes our assumptions about the function we want to learn.

Commonly used kernel functions include:

- **Squared Exponential (RBF) Kernel:**

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left( -\frac{|\mathbf{x} - \mathbf{x}'|^2}{2 \ell^2} \right)
$$

  where $\(\sigma_f^2\)$ is the variance and $\(\ell\)$ is the length-scale.

- **Matérn Kernel:**

$$
k(\mathbf{x}, \mathbf{x}') = \frac{1}{\Gamma(\nu)2^{\nu-1}} \left( \frac{\sqrt{2\nu} |\mathbf{x} - \mathbf{x}'|}{\ell} \right)^\nu K_\nu \left( \frac{\sqrt{2\nu} |\mathbf{x} - \mathbf{x}'|}{\ell} \right)
$$

  where $\(K_\nu\)$ is a modified Bessel function and $\(\nu\)$ is a parameter that controls the smoothness.

### Acquisition Function

In Bayesian Optimization, the acquisition function is used to determine the next point to sample. It balances exploration (sampling where the model uncertainty is high) and exploitation (sampling where the model prediction is high).

Common acquisition functions include:

- **Expected Improvement (EI):**

$$
EI(\mathbf{x}) = \mathbb{E}[\max(0, f(\mathbf{x}) - f(\mathbf{x}^+))]
$$

  where $\( f(\mathbf{x}^+) \)$ is the best observed value.

- **Upper Confidence Bound (UCB):**

$$
UCB(\mathbf{x}) = \mu(\mathbf{x}) + \kappa \sigma(\mathbf{x})
$$

  where $\( \mu(\mathbf{x}) \)$ is the mean prediction and $\( \sigma(\mathbf{x}) \)$ is the standard deviation.

**References:**
- Brochu, E., Cora, V. M., & De Freitas, N. (2010). *A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning*.

## Installation

To install the necessary dependencies for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ActiveLearning_Tutorial.git
