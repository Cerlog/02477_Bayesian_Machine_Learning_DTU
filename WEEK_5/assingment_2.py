import jax.numpy as jnp
from jax import value_and_grad
from jax import random
from scipy.optimize import minimize

from functions import * 
import matplotlib.pyplot as plt
import seaborn as snb


from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


# we want to use 64 bit floating precision
import jax
jax.config.update("jax_enable_x64", True)

snb.set_style('darkgrid')
snb.set_theme(font_scale=1.25)

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



def f_7(tau, prod=None, min_val=None, kappa_0=None, kappa_1=None, kappa_2=None, lengthscale=None):
    return  kappa_0**2 + kappa_1**2 * prod + kappa_2**2 * jnp.exp(- (tau**2) / (2 * lengthscale**2))


def f_1(tau, prod=None, min_val=None, kappa_0=None, kappa_1=None, kappa_2=None, lengthscale=None):
    return 2 * jnp.exp(- (tau**2 / (2 * 0.3**2)))


def f_2 (tau, prod=None, min_val=None, kappa_0=None, kappa_1=None, kappa_2=None, lengthscale=None):
    return jnp.exp(- (tau**2 / (2 * 0.3**2)))

def f_3(tau, prod=None, min_val=None, kappa_0=None, kappa_1=None, kappa_2=None, lengthscale=None):
    return 4 + 2* prod

def f_4(tau, prod=None, min_val=None, kappa_0=None, kappa_1=None, kappa_2=None, lengthscale=None):
    return jnp.exp(-2 * jnp.sin(3 * jnp.pi * tau)**2)


def f_5(tau, prod=None, min_val=None, kappa_0=None, kappa_1=None, kappa_2=None, lengthscale=None):
    return jnp.exp(-2 * jnp.sin(3 * jnp.pi * tau)**2) + 4* prod

def f_6(tau, prod=None, min_val=None, kappa_0=None, kappa_1=None, kappa_2=None, lengthscale=None):
    return 0.25 + min_val


class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun, kappa_0=1,  kappa_1=1., kappa_2=1, lengthscale=1.0):
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
        self.kappa_0 = kappa_0
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        self.lengthscale = lengthscale

    def contruct_kernel(self, X1, X2, kappa_0=None, kappa_1=None, kappa_2=None, lengthscale=None, jitter=1e-8):
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
        kappa_0 = self.kappa_0 if kappa_0 is None else kappa_0
        kappa_1 = self.kappa_1 if kappa_1 is None else kappa_1
        kappa_2 = self.kappa_2 if kappa_2 is None else kappa_2
        lengthscale = self.lengthscale if lengthscale is None else lengthscale

        # Compute pairwise distances (vectorized approach for efficiency)
        X1_expanded = X1[:, None, :]  # Shape: (N, 1, D)
        X2_expanded = X2[None, :, :]  # Shape: (1, M, D)
        pairwise_diff = X1_expanded - X2_expanded  # Shape: (N, M, D)
        pairwise_dist = jnp.linalg.norm(pairwise_diff, axis=2)  # Shape: (N, M)
        product = X2.T * X1 
       # print(product.shape)

        X1_flat = X1.flatten()  # Convert to 1D array
        X2_flat = X2.flatten()  # Convert to 1D array

        # Create broadcasting dimensions
        X1_expanded_ = X1_flat[:, None]  # Shape: (N, 1)
        X2_expanded_ = X2_flat[None, :]  # Shape: (1, M)

        # compute the min value 
        min = jnp.minimum(X1_expanded_, X2_expanded_)  # Shape: (N, M)

        # Apply kernel function
        K = self.kernel_fun(pairwise_dist, product,min ,kappa_0, kappa_1, kappa_2, lengthscale)

        # Add jitter to diagonal if X1 and X2 are the same set (stabilizes Cholesky decomposition)
        if jnp.array_equal(X1, X2):
            K = K + jnp.eye(N) * jitter

        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but got {K.shape}."
        return K


def plot_kernel_samples(X, kernel,  num_samples, key, filename, kappa_0=1., kappa_1=1., kappa_2=1., scale=0.5):
    # Instantiate kernel object and construct kernel matrix
    kernel = StationaryIsotropicKernel(kernel, kappa_0=kappa_0, kappa_1=kappa_1, kappa_2=kappa_2, lengthscale=scale)
    K = kernel.contruct_kernel(X, X)
    print("Covariance", K)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    
    # Kernel matrix visualization (shows covariance between points)
    m = jnp.zeros(len(X))
    im = ax[0].pcolormesh(X.flatten(), X.flatten(), K, shading='auto')
    ax[0].set(xlabel='Input x', ylabel="Input x'", 
              title=rf"Kernel function $k(x, x')$ for $\kappa_0$ = {kappa_0:2.1f}, $\kappa_1$ = {kappa_1:2.1f}, $\kappa_2$ = {kappa_2:2.1f} and $\ell$ = {scale:2.1f}")
    ax[0].grid(False)
    ax[0].set_aspect('equal')
    add_colorbar(im, fig, ax[0])
    
    # Generate and plot samples from the Gaussian process
    f_samples = generate_samples(key, m, K, num_samples=num_samples, jitter=1e-8)
    ax[1].plot(X, f_samples, alpha=0.75, linewidth=3)
    ax[1].grid(True)
    ax[1].set(xlabel='x', ylabel='f(x)', 
              title=rf'Samples from the Gaussian process with ${kernel.kernel_fun.__name__}$ kernel')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
