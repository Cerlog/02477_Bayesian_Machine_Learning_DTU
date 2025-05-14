
def squared_exponential(tau, kappa, lengthscale):
    return kappa**2*jnp.exp(-0.5*tau**2/lengthscale**2)

class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun, kappa=1., lengthscale=1.0):
        """
        Initialize the Stationary Isotropic Kernel with a given kernel function.
        
        The argument kernel_fun must be a function of three arguments:
        
        kernel_fun(||tau||, kappa, lengthscale), for example:
        squared_exponential = lambda tau, kappa, lengthscale: kappa**2 * np.exp(-0.5 * tau**2 / lengthscale**2)
        
        The kernel function models the covariance between points based on their distance,
        ensuring that closer points have higher similarity.
        
        Arguments:
            kernel_fun  -- Function defining the kernel.
            kappa       -- Magnitude (positive scalar, default 1.0), representing overall variance.
            lengthscale -- Characteristic lengthscale (positive scalar, default 1.0), controlling smoothness.
        """
        self.kernel_fun = kernel_fun
        self.kappa = kappa
        self.lengthscale = lengthscale

    def contruct_kernel(self, X1, X2, kappa=None, lengthscale=None, jitter=1e-8):
        """
        Compute and return the NxM kernel matrix between the two sets of input X1 (shape NxD) and X2 (MxD) using 
        the stationary and isotropic covariance function specified by self.kernel_fun.
    
        The kernel function is applied to the pairwise distances between input points:
        
        K[i, j] = kernel_fun(||X1_i - X2_j||, kappa, lengthscale)
        
        where:
        - ||X1_i - X2_j|| represents the Euclidean distance between two points.
        - kappa controls the overall variance.
        - lengthscale determines how quickly correlations decay with distance.
    
        Arguments:
            X1          -- NxD matrix of input points.
            X2          -- MxD matrix of input points.
            kappa       -- Magnitude (positive scalar, default is self.kappa).
            lengthscale -- Characteristic lengthscale (positive scalar, default is self.lengthscale).
            jitter      -- Non-negative scalar to stabilize computations (default 1e-8), used to ensure numerical stability.
        
        Returns:
            K           -- NxM kernel matrix representing covariances between input points.
        """
        # Extract dimensions 
        N, M = X1.shape[0], X2.shape[0]

        # Prepare hyperparameters
        kappa = self.kappa if kappa is None else kappa
        lengthscale = self.lengthscale if lengthscale is None else lengthscale

        # Compute pairwise distances (vectorized approach for efficiency)
        X1_expanded = X1[:, None, :]  # Shape: (N, 1, D)
        X2_expanded = X2[None, :, :]  # Shape: (1, M, D)
        pairwise_diff = X1_expanded - X2_expanded  # Shape: (N, M, D)
        pairwise_dist = jnp.linalg.norm(pairwise_diff, axis=2)  # Shape: (N, M)

        # Apply kernel function
        K = self.kernel_fun(pairwise_dist, kappa, lengthscale)

        # Add jitter to diagonal if X1 and X2 are the same set (stabilizes Cholesky decomposition)
        if jnp.array_equal(X1, X2):
            K = K + jnp.eye(N) * jitter

        # Alternative implementation 
        ##### # STEP 1: Compute all pairwise distances and apply kernel function
        ##### K = jnp.zeros((N, M))
        ##### for i in range(N):
        #####     for j in range(M):
        #####         # Compute pairwise distance using Euclidean norm
        #####         diff = X1[i] - X2[j]
        #####         distance = jnp.linalg.norm(diff)
        #####         # Apply kernel function to compute covariance
        #####         k_value = self.kernel_fun(distance, kappa, lengthscale)
        #####         # Update kernel matrix entry
        #####         K = K.at[i, j].set(k_value)
        ##### 
        ##### # STEP 2: Add jitter if X1 and X2 are identical (for numerical stability)
        ##### if jnp.array_equal(X1, X2):
        #####     identity = jnp.eye(N)
        #####     jitter_matrix = identity * jitter
        #####     K = K + jitter_matrix

        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but got {K.shape}."
        return K
    
class GaussianProcessRegression(object):

    def __init__(self, X, y, kernel, kappa=1., lengthscale=1., sigma=1/2, jitter=1e-8):
        """  
        Arguments:
            X                -- NxD input points
            y                -- Nx1 observed values 
            kernel           -- must be instance of the StationaryIsotropicKernel class
            jitter           -- non-negative scaler
            kappa            -- magnitude (positive scalar)
            lengthscale      -- characteristic lengthscale (positive scalar)
            sigma            -- noise std. dev. (positive scalar)
        """
        self.X = X
        self.y = y
        self.N = len(X)
        self.kernel = kernel
        self.jitter = jitter
        self.set_hyperparameters(kappa, lengthscale, sigma)
        self.check_dimensions()

    def check_dimensions(self):
        assert self.X.ndim == 2, f"The variable X must be of shape (N, D), however, the current shape is: {self.X.shape}"
        N, D = self.X.shape

        assert self.y.ndim == 2, f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"
        assert self.y.shape == (N, 1), f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"
        

    def set_hyperparameters(self, kappa, lengthscale, sigma):
        self.kappa = kappa
        self.lengthscale = lengthscale
        self.sigma = sigma

    def posterior_samples(self, key, Xstar, num_samples):
        """Generate samples from the posterior p(f^*|y, x^*) for each of the inputs in Xstar
        
        Implements sampling from the multivariate Gaussian:
        f_* ~ N(μ_{*|X}, Σ_{*|X})
        
        Arguments:
            key              -- jax random key for controlling the random number generator
            Xstar            -- PxD prediction points
            num_samples      -- number of samples to generate
        
        Returns:
            f_samples        -- numpy array of (P, num_samples) containing num_samples for each of the P inputs in Xstar
        """
        # Get posterior mean and covariance at test points
        mu, Sigma = self.predict_f(Xstar)
        
        # Use generate_samples to draw samples from this distribution:
        # f_* ~ N(μ_{*|X}, Σ_{*|X})
        f_samples = generate_samples(key, mu, Sigma, num_samples)

        assert (f_samples.shape == (len(Xstar), num_samples)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, {num_samples}), but actual shape was {f_samples.shape}. Please check implementation"
        return f_samples
        
    def predict_y(self, Xstar):
        """Returns the posterior distribution of y^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Implements the equations:
        μ_{y_*} = μ_{*|X}
        Σ_{y_*} = Σ_{*|X} + σ²I
        
        This differs from predict_f by adding observation noise σ² to the covariance.
        
        Arguments:
        Xstar       -- PxD prediction points
        
        Returns:
        mu          -- Px1 mean vector (μ_{y_*})
        Sigma_y     -- PxP covariance matrix (Σ_{y_*})
        """
        # Get posterior distribution of latent function f*
        mu, Sigma_ = self.predict_f(Xstar)
        
        # Add observation noise to covariance: Σ_{y_*} = Σ_{*|X} + σ²I
        Sigma_y = Sigma_ + self.sigma**2 * jnp.eye(len(Xstar))
        
        return mu, Sigma_y

    def predict_f(self, Xstar):
        """Returns the posterior distribution of f^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Implements the equations:
        μ_{*|X} = K_{X,*}^T (K_{XX} + σ²I)^{-1} y
        Σ_{*|X} = K_{*,*} - K_{X,*}^T (K_{XX} + σ²I)^{-1} K_{X,*}
        
        Arguments:
        Xstar -- PxD prediction points
        
        Returns:
        mu    -- Px1 mean vector (μ_{*|X})
        Sigma -- PxP covariance matrix (Σ_{*|X})
        """
        # Compute kernel matrices
        K_X_X = self.kernel.contruct_kernel(self.X, self.X, self.kappa, self.lengthscale, self.jitter)  # K_{XX}
        K_star_star = self.kernel.contruct_kernel(Xstar, Xstar, self.kappa, self.lengthscale, self.jitter)  # K_{*,*}
        K_x_star = self.kernel.contruct_kernel(self.X, Xstar, self.kappa, self.lengthscale)  # K_{X,*}
        
        # Add noise to training covariance: K_{XX} + σ²I
        K_sigma = K_X_X + self.sigma**2 * jnp.eye(len(self.X))
        
        # Compute (K_{XX} + σ²I)^{-1} y efficiently without explicit inversion
        alpha = jnp.linalg.solve(K_sigma, self.y)
        
        # Compute mean: μ_{*|X} = K_{X,*}^T (K_{XX} + σ²I)^{-1} y
        mu = K_x_star.T @ alpha
        
        # Compute (K_{XX} + σ²I)^{-1} K_{X,*} efficiently
        v = jnp.linalg.solve(K_sigma, K_x_star)
        
        # Compute covariance: Σ_{*|X} = K_{*,*} - K_{X,*}^T (K_{XX} + σ²I)^{-1} K_{X,*}
        Sigma = K_star_star - K_x_star.T @ v

        # sanity check for dimensions
        assert (mu.shape == (len(Xstar), 1)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, 1), but actual shape was {mu.shape}. Please check implementation"
        assert (Sigma.shape == (len(Xstar), len(Xstar))), f"The shape of the posterior Sigma seems wrong. Expected ({len(Xstar)}, {len(Xstar)}), but actual shape was {Sigma.shape}. Please check implementation"

        return mu, Sigma
    
    def log_marginal_likelihood(self, kappa, lengthscale, sigma):
        """Evaluate the log marginal likelihood p(y|X,θ) given the hyperparameters
        
        Implements the equation:
        log p(y|X,θ) = -1/2 * (y^T (K + σ²I)^{-1} y + log|K + σ²I| + n*log(2π))
        
        Efficient implementation uses Cholesky decomposition for numerical stability.
        
        Arguments:
            kappa       -- magnitude parameter (positive scalar) 
            lengthscale -- characteristic lengthscale (positive scalar)
            sigma       -- noise std. dev. (positive scalar)
        
        Returns:
            log_marginal -- scalar value of log marginal likelihood
        """
        # Compute kernel matrix with current hyperparameters
        K = self.kernel.contruct_kernel(self.X, self.X, kappa, lengthscale)  # K
        
        # Add noise variance: C = K + σ²I
        C = K + sigma**2 * jnp.identity(self.N)
        
        # Compute Cholesky decomposition: C = LL^T
        L = jnp.linalg.cholesky(C)
        
        # Solve L*v = y efficiently
        v = jnp.linalg.solve(L, self.y)
        
        # Compute log determinant term: log|C| = 2*sum(log(diag(L)))
        # Note: Factor of 2 is accounted for in the constant term
        logdet_term = jnp.sum(jnp.log(jnp.diag(L)))
        
        # Compute quadratic term: 1/2 * y^T C^{-1} y = 1/2 * ||v||²
        quad_term = 0.5 * jnp.sum(v**2)
        
        # Constant term: -1/2 * n * log(2π)
        const_term = -0.5 * self.N * jnp.log(2*jnp.pi)
        
        # Combine terms: log p(y|X,θ) = const_term - logdet_term - quad_term
        return const_term - logdet_term - quad_term
def generate_samples(key, m, K, num_samples, jitter=0):
    """
    Generate samples from a multivariate Gaussian distribution N(m, K)
    - where m is the mean vector of shape (N,)
    - K is the covariance matrix of shape (N, N)

    
    Uses the Cholesky decomposition method to transform standard normal samples:
    
    1. If z ~ N(0, I) (standard normal), and if K = L L^T (Cholesky decomposition), then:
       f = m + Lz ~ N(m, K)
    
    2. Adding jitter to the diagonal elements for numerical stability:
       K_jitter = K + jitter * I
       where I is the identity matrix of the same shape as K.

    3. Computing Cholesky decomposition:
       L = cholesky(K_jitter)
    
    4. Generating samples:
       f = m + Lz
    
    Arguments:
        key          -- JAX random key for controlling the random number generator.
        m            -- Mean vector (shape (N,)).
        K            -- Covariance matrix (shape NxN).
        num_samples  -- Number of samples to generate (positive integer).
        jitter       -- Amount of jitter to add to the diagonal for numerical stability.
    
    Returns:
        f_samples    -- Matrix containing the samples (shape N x num_samples).
    """
    # Get dimension from covariance matrix
    N = len(K)

    # Generate standard normal samples z ~ N(0, I)
    zs = random.normal(key, shape=(N, num_samples))
    print(f"The shape of z's: {zs.shape}")

    # Add jitter to diagonal elements for numerical stability
    K_jitter = K + jnp.eye(N) * jitter
    

    # Compute the Cholesky decomposition: K = LL^T
    L = jnp.linalg.cholesky(K_jitter)


################ CODE BELOW IS FOR DEBUGGING PURPOSES ################
    print(f"Shape of the mean, mu {m.shape}")
    # printing the reshaped mean vector after broadcasting
    m_reshaped = m.reshape(-1, 1)                           # shape (N, 1)
    print(f"Shape of mu after broadcasting {m_reshaped.shape}")
    print(f"Shape of L: {L.shape}")
################ CODE ABOVE IS FOR DEBUGGING PURPOSES ################

    # transform the samples: f = m + L*z
    f_samples = m_reshaped + L @ zs

    print(f"Shape of f_samples {f_samples.shape}")
    # Verify output dimensions
    assert f_samples.shape == (N, num_samples), f"Incorrect sample shape: expected ({N}, {num_samples}), got {f_samples.shape}"
    
    return f_samples
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



def metropolis_for_chains(log_target, num_params, tau, num_iter, theta_init=None, seed=0):    
    """ Runs a Metropolis-Hastings sampler 
    
        Arguments:
        log_target:         function for evaluating the log target distribution, i.e. log \tilde{p}(theta). The function expect a parameter of size num_params.
        num_params:         number of parameters of the joint distribution (integer)
        tau:                standard deviation of the Gaussian proposal distribution (positive real)
        num_iter:           number of iterations (integer)
        theta_init:         vector of initial parameters (jnp.array with shape (num_params) or None)        
        seed:               seed (integer)

        returns
        thetas              jnp.array with MCMC samples (jnp.array with shape (num_iter+1, num_params))
    """ 
    
    # set initial key
    key = random.PRNGKey(seed)

    if theta_init is None:
        theta_init = jnp.zeros((num_params))
    
    # prepare lists 
    thetas = [theta_init]
    accepts = []
    log_p_theta = log_target(theta_init)
    
    for k in range(num_iter):

        # update keys: key_proposal for sampling proposal distribution and key_accept for deciding whether to accept or reject.
        key, key_proposal, key_accept = random.split(key, num=3)

        # get the last value for theta and generate new proposal candidate
        theta_cur = thetas[-1]
        theta_star = theta_cur + tau*random.normal(key_proposal, shape=(num_params, ))
        
        # evaluate the log density for the candidate sample
        log_p_theta_star = log_target(theta_star)

        # compute acceptance probability
        log_r = log_p_theta_star - log_p_theta
        A = min(1, jnp.exp(log_r))
        
        # accept new candidate with probability A
        if random.uniform(key_accept) < A:
            theta_next = theta_star
            log_p_theta = log_p_theta_star
            accepts.append(1)
        else:
            theta_next = theta_cur
            accepts.append(0)

        thetas.append(theta_next)


        
    print('Acceptance ratio: %3.2f' % jnp.mean(jnp.array(accepts)))
        
    # return as jnp.array
    thetas = jnp.stack(thetas)

    # check dimensions and return
    assert thetas.shape == (num_iter+1, num_params), f'The shape of thetas was expected to be ({num_iter+1}, {num_params}), but the actual shape was {thetas.shape}. Please check your code.'
    return thetas, accepts

def metropolis_multiple_chains(log_target, num_params, num_chains, tau, num_iter, theta_init, seeds, warm_up=0):
    """
    Runs multiple Metropolis-Hastings (MH) Markov Chain Monte Carlo (MCMC) chains in parallel.

    **Overview:**
      This function orchestrates running multiple independent MH chains, where each chain
      is initialized with a different starting point and seed. The sampler uses the given
      log_target function to compute the log of the joint probability of the target distribution.
      The candidate proposals for each chain are generated from a Gaussian (Normal) distribution 
      centered at the current state with variance 'tau'. After iterating, a burn-in period (if any)
      is discarded from each chain's samples.

    **Mathematical Background:**
      The MH algorithm proposes a new candidate state $\theta^{*}$ based on the current state $\theta^{(t)}$
      using a proposal density:
          \[
          \theta^{*} \sim \mathcal{N}(\theta^{(t)}, \tau)
          \]
      The acceptance probability $\alpha$ is computed via the ratio of the target densities:
          \[
          \alpha = \min\left(1, \frac{p(\theta^{*})}{p(\theta^{(t)})}\right)
          \]
      In log-space (for numerical stability), this becomes:
          \[
          \alpha = \min\left(1, \exp\left(\log p(\theta^{*}) - \log p(\theta^{(t)})\right)\right)
          \]
      If the candidate $\theta^{*}$ is accepted (with probability $\alpha$), it becomes the next state;
      otherwise, the chain remains at $\theta^{(t)}$.

    **Input Arguments:**
      - log_target:
          A function that computes the log joint probability $ \log p(\theta) $ of the target distribution.
          It should accept a parameter vector (or scalar for one-dimensional problems) and return a scalar.
      - num_params (int):
          Number of parameters (the dimensionality of the space). For a 1D problem, num_params = 1.
      - num_chains (int):
          Total number of independent MCMC chains to run.
      - tau (float):
          Variance (or scaling factor) used in the Gaussian proposal distribution. It controls the size of the 
          proposed moves.
      - num_iter (int):
          The number of iterations to perform for each chain. Note that the chain will actually produce 
          num_iter+1 samples since the initial state is included.
      - theta_init (jnp.array):
          An array containing the initial states for each chain with shape 
          $text{(num_chains, num_params)}$.
      - seeds (jnp.array):
          An array of random seeds, one for each chain, with shape $text{(num_chains,)}$.
      - warm_up (int, optional):
          Number of initial iterations/samples (burn-in period) to discard for each chain. Default is 0.
    
    **Output:**
      - thetas (jnp.array):
          A 3D array of the collected samples from each chain, after discarding warm-up samples.
          Final shape is $(\text{num_chains}, \text{num_iter}+1-\text{warm_up}, \text{num_params})$.
          Here, each chain produces num_iter+1 samples including the initial sample.
      - accept_rates (jnp.array):
          A 1D array of length num_chains, where each element is the acceptance rate (fraction of accepted
          proposals) for that chain.
    
    **Usage Example:**
      ```python
      # Number of chains and iterations
      num_chains = 4
      num_iter = 1000
      proposal_variance = 0.1  # Parameter tau for the Gaussian proposal
      num_params = 1
      warm_up = 0  # No burn-in samples to discard in this example
      seeds = jnp.arange(num_chains)  # One seed per chain

      # Generate initial states from a Normal distribution N(0, 5^2)
      key = random.PRNGKey(1)
      theta_init = 5 * random.normal(key, shape=(num_chains, num_params))  # Shape: (num_chains, num_params)

      # Run the sampler with the target log probability function `log_target`
      chains, accepts = metropolis_multiple_chains(log_target, num_params, num_chains,
                                                   proposal_variance, num_iter,
                                                   theta_init, seeds, warm_up)

      # Compute and print the overall estimated mean and variance from the samples
      estimated_mean = jnp.mean(chains.ravel())
      estimated_variance = jnp.var(chains.ravel())
      print(f'Estimated mean:\t\t{estimated_mean:+3.2f}')
      print(f'Estimated variance:\t{estimated_variance:+3.2f}')
      ```
    """

    # --- Verify Input Dimensions ---
    # Ensure that theta_init has the correct shape: (num_chains, num_params)
    assert theta_init.shape == (num_chains, num_params), (
        "theta_init seems to have the wrong dimensions. Please check your code: expected shape "
        "(num_chains, num_params)."
    )

    ###########################################
    # Step 1. Initialize Containers for Output
    ###########################################
    # Here, two lists are created to store:
    # 1. The chain samples from each independent chain.
    # 2. The corresponding acceptance indicators or rates.
    thetas = []          # Will hold the samples for each chain.
    accept_rates = []    # Will hold the acceptance rate for each chain.

    ###########################################
    # Step 2. Run Each Metropolis Chain
    ###########################################
    # Loop over the number of chains. For each chain, run the individual Metropolis sampler.
    for idx_chain in range(num_chains):
        # Inform the user which chain is currently running
        print(f"Running chain {idx_chain}. ", end='')

        # Run the individual Metropolis sampler (assumed to be pre-defined).
        #
        # The individual sampler 'metropolis' takes as inputs:
        #  - log_target: the log joint probability function.
        #  - num_params: the dimensionality of the state space.
        #  - tau: the variance of the Gaussian proposal distribution.
        #  - num_iter: the number of iterations to run.
        #  - theta_init: the initial state for this specific chain (1D vector of length num_params).
        #  - seed: the random seed for reproducibility in this chain.
        #
        # The function 'metropolis' returns:
        #  - thetas_temp: an array of samples, shape (num_iter + 1, num_params) because it includes the initial state.
        #  - accepts_temp: a list/array of acceptance indicators (1 for accepted move, 0 otherwise) for each iteration.
        thetas_temp, accepts_temp = metropolis_for_chains(     # shapes (num_iter + 1, num_params) and (num_iter + 1,)
            log_target=log_target,
            num_params=num_params,
            tau=tau,
            num_iter=num_iter,
            theta_init=theta_init[idx_chain],
            seed=seeds[idx_chain]
        )

        # Append the samples and acceptance indicators for the current chain to the respective lists.
        thetas.append(thetas_temp)   # shapes (num_iter + 1, num_params)
        accept_rates.append(jnp.array(accepts_temp)) # shapes (num_iter + 1,)

    ###########################################
    # Step 3. Combine the Results from All Chains
    ###########################################
    # Stack the lists into jnp.arrays along a new "chain" axis:
    #
    # thetas: becomes an array of shape (num_chains, num_iter+1, num_params)
    # accept_rates: becomes an array of shape (num_chains, ...)
    thetas = jnp.stack(thetas, axis=0)              # shapes (num_chains, num_iter + 1, num_params)
    accept_rates = jnp.stack(accept_rates, axis=0)  # shapes (num_chains, num_iter + 1)

    # Discard the initial 'warm_up' (burn-in) samples from each chain.
    # After discarding, the shape of 'thetas' is:
    #      (num_chains, num_iter+1-warm_up, num_params)
    thetas = thetas[:, warm_up:, :]                # shapes (num_chains, num_iter + 1 - warm_up, num_params)

    ###########################################
    # Step 4. Output Verification and Return
    ###########################################
    # Verify that the output dimensions match expectations.
    expected_shape = (num_chains, num_iter + 1 - warm_up, num_params)
    assert thetas.shape == expected_shape, (
        f"The expected shape of chains is {expected_shape}, but the actual shape is {thetas.shape}. "
        "Check your implementation."
    )
    # Verify the number of acceptance rate entries equals the number of chains.
    assert len(accept_rates) == num_chains, "Mismatch between the number of chains and acceptance rates."
        

    # Return the chains samples and acceptance rates.
    return thetas, accept_rates  # shapes (num_chains, num_iter+1-warm_up, num_params) and (num_chains, num_iter+1-warm_up)


def compute_Rhat(chains):
    """ Computes the Rhat convergence diagnostic for each parameter in a MCMC simulation. 
        The function expects the argument chain to be a numpy array of shape (num_chains x num_samples x num_params)
        and it return a numpy of shape (num_params) containing the Rhat estimates for each parameter
    """

    # get dimensions
    num_chains, num_samples, num_params = chains.shape

    # make subchains by splitting each chains in half
    sub_chains = []
    half_num_samples = int(0.5*num_samples)
    for idx_chain in range(num_chains):
        sub_chains.append(chains[idx_chain, :half_num_samples, :])
        sub_chains.append(chains[idx_chain, half_num_samples:, :])

    # count number of sub chains
    num_sub_chains = len(sub_chains)
        
    # compute mean and variance of each subchain
    chain_means = np.array([np.mean(s, axis=0) for s in sub_chains])                                             # dim: num_sub_chains x num_params
    chain_vars = np.array([1/(num_samples-1)*np.sum((s-m)**2, 0) for (s, m) in zip(sub_chains, chain_means)])    # dim: num_sub_chains x num_params

    # compute between chain variance
    global_mean = np.mean(chain_means, axis=0)                                                                   # dim: num_params
    B = num_samples/(num_sub_chains-1)*np.sum((chain_means - global_mean)**2, axis=0)                            # dim: num_params

    # compute within chain variance
    W = np.mean(chain_vars, 0)                                                                                   # dim: num_params                                                          

    # compute estimator and return
    var_estimator = (num_samples-1)/num_samples*W + (1/num_samples)*B                                            # dim: num_params 
    Rhat = np.sqrt(var_estimator/W)
    return Rhat

def compute_effective_sample_size(chains_):
    """ computes the effective sample size for each parameter in a MCMC simulation. 
        The function expects the argument chain to be a numpy array of shape (num_chains x num_samples x num_params)
        and it return a numpy of shape (num_params) containing the S_eff estimates for each parameter
    """

    # force numpy
    chains = np.array(chains_)

    # get dimensions
    num_chains, num_samples, num_params = chains.shape

    # estimate sample size for each parameter
    S_eff = np.array([compute_effective_sample_size_single_param(chains[:, :, idx_param]) for idx_param in range(num_params)])

    # return
    return S_eff

def compute_effective_sample_size_single_param(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation. """
    m_chains, n_iters = x.shape

    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    post_var = gelman_rubin(x)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))



def gelman_rubin(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    m_chains, n_iters = x.shape

    # Calculate between-chain variance
    B_over_n = ((jnp.mean(x, axis=1) - jnp.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    # (over) estimate of variance
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2