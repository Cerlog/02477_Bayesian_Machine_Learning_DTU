import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from jax import random
from scipy.optimize import minimize
import pylab as plt
from scipy.stats import multivariate_normal as mvn
def metropolis(log_target, num_params, tau, num_iter, theta_init=None, seed=0):    
    """ Runs a Metropolis-Hastings sampler 

        Arguments:
        log_target:         function for evaluating the log target distribution, i.e. log \tilde{p}(theta). The function expect a parameter of size num_params.
        num_params:         number of parameters of the joint distribution (integer)
        tau:                standard deviation of the Gaussian proposal distribution (positive real)
        num_iter:           number of iterations (integer)
        theta_init:         vector of initial parameters (np.array with shape (num_params) or None)        
        seed:               seed (integer)

        returns
        thetas              np.array with MCMC samples (np.array with shape (num_iter+1, num_params))
    """ 

    # set initial key
    key = random.PRNGKey(seed)

    # if there is no theta init, it starts with zero
    # theta init is \theta^{k-1}
    if theta_init is None:

        theta_init = jnp.zeros((num_params))
        print(f"Shape of theta init {theta_init.shape} is the same as a number of parameters {num_params}")
    # prepare lists 
    thetas = [theta_init] # list to store all samples, starting with the initial 
    accepts = [] # list of accepted proposals 
    log_p_theta = log_target(theta_init) # log probability of the initial position 

    for k in range(num_iter):
        
        

        # update keys: key_proposal for sampling proposal distribution and key_accept for deciding whether to accept or reject.
        key, key_proposal, key_accept = random.split(key, num=3)

        ##############################################
        # Your solution goes here
        ##############################################
        # Get the current state theta^{(k)} from the end of the list 
        theta_current = thetas[-1] # current position of the chain, \theta^{k-1}
        
        # 1. Propose a new state theta' ~ q(theta' | theta_current)
        # Draw from N(theta_current, tau^2 * I)
        noise = random.normal(key_proposal, shape=(num_params))
        theta_proposal = theta_current + tau * noise
        
        log_p_theta_proposal = log_target(theta_proposal)
        
        log_acceptance_ratio = log_p_theta_proposal - log_p_theta 
        
        u = random.uniform(key_accept)
        
        accept = jnp.log(u) < jnp.minimum(0.0, log_acceptance_ratio)
        
        if accept:
            thetas.append(theta_proposal)
            log_p_theta = log_p_theta_proposal
            accepts.append(1.)
        else: 
            thetas.append(theta_current)
            accepts.append(0)
        
        

        ##############################################
        # End of solution
        ##############################################

    print('Acceptance ratio: %3.2f' % jnp.mean(jnp.array(accepts)))

    # return as np.array
    thetas = jnp.stack(thetas)

    # check dimensions and return
    assert thetas.shape == (num_iter+1, num_params), f'The shape of thetas was expected to be ({num_iter+1}, {num_params}), but the actual shape was {thetas.shape}. Please check your code.'
    return thetas