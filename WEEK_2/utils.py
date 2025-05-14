
import pylab as plt
import pandas as pd
import jax.numpy as jnp
from jax import random
import seaborn as snb
import numpy as np

from scipy.stats import binom as binom_dist

sigmoid = lambda x: 1./(1 + jnp.exp(-x))
log_npdf = lambda x, m, v: -(x-m)**2/(2*v) - 0.5*jnp.log(2*jnp.pi*v)

class LogisticRegression(object):

    def __init__(self, x, y, N, sigma2_alpha=1., sigma2_beta=1.):
        # data
        self.x = x
        self.y = y
        self.N = N

        # hyperparameters
        self.sigma2_alpha = sigma2_alpha
        self.sigma2_beta = sigma2_beta

    def f(self, x, alpha, beta):
        """ implements eq. (3). Output must have the same shape as x """
        output = alpha + beta * x
        return output
        
    def theta(self, x, alpha, beta):
        """ implements eq. (2). Output must have the same shape as x """
        z = self.f(x, alpha, beta)
        return sigmoid(z)

    def log_prior(self, alpha, beta):
        """ implements log. of eq. (8). Output must have the same shape as alpha and beta """
        return log_npdf(alpha, 0, self.sigma2_alpha) + log_npdf(beta, 0, self.sigma2_beta)

    def log_likelihood(self, alpha, beta):
        """ implements log. of eq. (5). Output must have the same shape as alpha and beta """
        theta = self.theta(self.x, alpha, beta)
        loglik = jnp.sum(binom_dist.logpmf(self.y, self.N, theta), axis=0)
        print(f"Shape of loglik {loglik.shape}")
        #print(f"Shape of alpha {alpha.ndim}")
        #print(f"Shape of beta {beta.ndim}")
        return loglik

    def log_joint(self, alpha, beta):
        return self.log_prior(alpha, beta).squeeze() + self.log_likelihood(alpha, beta).squeeze()
    


## instantiate model
#model = LogisticRegression(x, y, N)
#
## some sanity checks to help verify your implementation and make it compatible with the rest of the exercise
#assert jnp.allclose(model.theta(x=2, alpha=2, beta=-0.5), 0.7310585786300049), "The value of output of the theta-function was different than expected. Please check your code."
#assert jnp.allclose(model.log_prior(1., 1.), -2.8378770664093453), "The value of the output of the log_prior function was different than expected. Please check your code."
#assert jnp.allclose(model.log_likelihood(-1.,2.), -95.18926297085957), "The value of the output of the log_likelihood function was different than expected. Please check your code."
#
#assert model.theta(jnp.linspace(-3, 3, 10), 1, 1).shape == (10,), "The shape of the output of the theta-function was different than expected. Please check your code."
#assert model.log_prior(jnp.linspace(-3, 3, 10), jnp.linspace(-3, 3, 10)).shape == (10,), "The shape of the output of the log_prior-function was different than expected. Please check your code."
#assert model.log_likelihood(jnp.linspace(-3, 3, 10)[:, None], jnp.linspace(-3, 3, 10)[:, None]).shape == (10, 1),  "The shape of the output of the log_likelihood-function was different than expected. Please check your code."



# Interactive testing
if __name__ == "__main__":
    # ---------------------------------------------------
    # Create a reproducible example:
    # ---------------------------------------------------
    # We will create some example data.

    # 1. Define our input feature x (for example, 5 data points)
    x = jnp.array([0., 1., 2., 3., 4.])

    # 2. Define the number of trials for the binomial likelihood:
    N = 10  # each observation is the result of 10 trials

    # 3. Choose "true" parameters for simulation
    alpha_true = -1.0
    beta_true = 2.0

    # 4. Compute the probability for each x using the logistic function:
    theta_true = sigmoid(alpha_true + beta_true * x)
    print("True probabilities (theta):", theta_true)

    # 5. Simulate some observed outcomes y from a binomial distribution.
    # We use numpy's random generator for reproducibility.
    rng = np.random.default_rng(seed=0)
    y = jnp.array(rng.binomial(n=N, p=np.array(theta_true)))
    print("Simulated observed outcomes (y):", y)

    # ---------------------------------------------------
    # Instantiate the model with the simulated data:
    model = LogisticRegression(x, y, N)

    # ---------------------------------------------------
    # Now, let us run through each function and see what happens:

    # A. Compute theta for a given set of parameters:
    alpha_example = 1.0
    beta_example = -0.5
    theta_values = model.theta(x, alpha_example, beta_example)
    print("\nOutput of theta(x, alpha=1.0, beta=-0.5):")
    print(theta_values)

    # B. Compute the log prior for given parameters:
    log_prior_val = model.log_prior(alpha_example, beta_example)
    print("\nLog prior for alpha=1.0 and beta=-0.5:")
    print(log_prior_val)

    # C. Compute the log likelihood for given parameters:
    # Here, our x and y (simulated) are used.
    log_likelihood_val = model.log_likelihood(alpha_example, beta_example)
    print("\nLog likelihood for alpha=1.0 and beta=-0.5:")
    print(log_likelihood_val)

    # D. Compute the joint log probability (log_prior + log_likelihood):
    log_joint_val = model.log_joint(alpha_example, beta_example)
    print("\nJoint log probability (log_prior + log_likelihood) for alpha=1.0 and beta=-0.5:")
    print(log_joint_val)
