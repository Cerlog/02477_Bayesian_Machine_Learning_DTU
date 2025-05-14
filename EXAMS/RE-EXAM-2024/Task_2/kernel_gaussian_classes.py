import jax.numpy as jnp
import jax.numpy as jnp
import jax.random as random
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
