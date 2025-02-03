import jax.numpy as jnp 
from scipy.special import comb
from scipy.stats import norm


def bernoulli_likelihood(y, n, thetas):
    """
    Compute the Bernoulli likelihood for given parameters.
    
    Args:
        y (int): Number of successes
        n (int): Number of trials
        thetas (array): Array of probability values
    
    Returns:
        array: Likelihood values for each theta
    """
    # Compute binomial coefficient (n choose y)
    binom_coef = comb(n, y)

    # Compute likelihood using the Bernoulli formula
    return binom_coef * jnp.power(thetas, y) * jnp.power(1 - thetas, n - y)


def bernoulli_MLE(y, n):
    """
    Compute the maximum likelihood estimate for a Bernoulli distribution.
    
    Args:
        n (int): Number of trials
        y (int): Number of successes
    
    Returns:
        float: Maximum likelihood estimate
    """
    # The MLE for a Bernoulli distribution is the number of successes divided by the number of trials
    return y / n


def confidence_intervals(MLE, N1):
    """
    Calculate confidence intervals for a Bernoulli parameter estimate.
    
    Args:
        MLE (float): Maximum likelihood estimate
        N1 (int): Sample size
        confidence_level (float): Confidence level (default: 0.95 for 95%)
    
    Returns:
        tuple: Lower and upper bounds of the confidence interval
    """
    # z-score based on confidence level
    z = 1.96
    
    # Calculate the margin of error using the standard formula
    # sqrt(p * (1-p) / n) where p is the MLE and n is sample size
    conf_interval = z * jnp.sqrt(MLE * (1 - MLE) / N1)

    upper = MLE + conf_interval
    lower = MLE - conf_interval
    
    # Return the lower and upper bounds of the confidence interval
    return lower, upper


import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import beta, binom

def plot_bayesian_inference(y, N, a_0=1, b_0=1):
    """
    Plots the prior, likelihood, and posterior distributions for a Beta-Binomial Bayesian inference.

    Args:
        y (int): Number of observed successes
        N (int): Number of trials
        a_0 (float, optional): Prior shape parameter a (default: 1)
        b_0 (float, optional): Prior shape parameter b (default: 1)
    
    Returns:
        dict: A dictionary containing the MAP estimate and posterior mean.
    """
    
    # Grid for plotting in [0,1]
    thetas = jnp.linspace(0, 1, 1000)

    # Prior distribution
    beta_prior = beta.pdf(thetas, a_0, b_0)

    # Likelihood distribution
    beta_likelihood = binom.pmf(y, N, thetas)

    # Posterior parameters
    a = a_0 + y
    b = b_0 + N - y

    # Posterior distribution
    beta_posterior = beta.pdf(thetas, a, b)

    # MAP estimate (mode of posterior)
    MAP_index = jnp.argmax(beta_posterior)
    MAP_theta = thetas[MAP_index]
    print(f"MAP estimate theta: {MAP_theta:.4f}")
    # Posterior mean
    post_mean_theta = a / (a + b)
    print(f"Posterior mean theta: {post_mean_theta:.4f}")

    upper_lower = confidence_intervals(post_mean_theta, N)

    # Plot prior, likelihood, and posterior
    plt.figure(figsize=(12, 8))
    plt.plot(thetas, beta_prior, label=r'Prior $p(\theta)$')
    plt.plot(thetas, beta_likelihood, label=r'Likelihood $p(y|\theta)$')
    plt.plot(thetas, beta_posterior, label=r'Posterior $p(\theta|y)$')
    plt.axvline(MAP_theta, color='red', label='MAP', linestyle='--')
    plt.axvline(post_mean_theta, color='green', linestyle=':', label=r'Posterior mean $\mathbb{E}[\theta \mid y]$')
    plt.axvline(upper_lower[0], color='purple', linestyle='-.', label='95% CI')
    plt.axvline(upper_lower[1], color='purple', linestyle='-.')
    plt.title(rf'Prior, likelihood, and posterior for $y={y}$ and $N={N}$ and $\alpha$={a_0}, $\beta$={b_0}')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.legend()
    plt.show()

    # Return results
    return {
        "MAP_estimate": float(MAP_theta),
        "Posterior_mean": float(post_mean_theta)
    }


def stars(n=100):
    print("*" * n)



if __name__ == "__main__":
    # Example usage
    results = plot_bayesian_inference(y=1, N=7)
    print(f"MAP estimate: {results['MAP_estimate']:.4f}")
    print(f"Posterior mean: {results['Posterior_mean']:.4f}")