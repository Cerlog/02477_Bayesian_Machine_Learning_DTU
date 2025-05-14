import jax.numpy as jnp
import numpy as np # Often needed for defining the initial grid ranges
import matplotlib.pyplot as plt

class Grid2D(object):
    """
    Helper class for evaluating a function on a 2D grid defined by (alpha, beta).

    This class takes arrays of alpha and beta values, creates a 2D grid,
    evaluates a given function at each point (alpha, beta) on the grid,
    and provides methods for plotting contours and finding the grid point
    where the function value is maximum.
    """

    def __init__(self, alphas, betas, func, name="Grid2D"):
        """
        Initializes the Grid2D object.

        Parameters:
            alphas (array-like): 1D array or list of alpha values for the grid.
                                 Shape: $(N_{\alpha},)$
            betas (array-like):  1D array or list of beta values for the grid.
                                 Shape: $(N_{\beta},)$
            func (callable):     A function to evaluate on the grid. It should accept
                                 two arguments (alpha, beta) which can be scalars or
                                 broadcastable arrays (specifically, it will be called
                                 internally with shapes $(N_{\alpha}, N_{\beta}, 1)$ for alpha
                                 and $(N_{\alpha}, N_{\beta}, 1)$ for beta) and return a
                                 scalar or a broadcastable array result.
                                 Example: model.log_prior, model.log_likelihood.
            name (str):          Optional name for this grid, used as a default plot title.
        """
        # Store the grid values and the function to evaluate
        self.alphas = jnp.asarray(alphas) # Ensure JAX array, Shape: (N_alpha,)
        self.betas = jnp.asarray(betas)   # Ensure JAX array, Shape: (N_beta,)
        self.func = func
        self.name = name

        # Determine the grid dimensions (number of points along each axis)
        self.grid_size = (len(self.alphas), len(self.betas)) # Tuple: (N_alpha, N_beta)

        # Create a meshgrid for alpha and beta using 'ij' indexing so that the first dimension
        # corresponds to alpha and the second to beta.
        # self.alpha_grid shape: (N_alpha, N_beta)
        # self.beta_grid shape:  (N_alpha, N_beta)
        self.alpha_grid, self.beta_grid = jnp.meshgrid(self.alphas, self.betas, indexing='ij')

        # Compute the function values on the grid.
        # We call self.func on every point of the grid. For example, func might be
        # model.log_prior, so it gets evaluated on every (alpha, beta) pair in the grid.
        # By adding a new axis at the end (with [:, :, None]) we prepare the alpha and beta arrays
        # (shapes become (N_alpha, N_beta, 1)) for broadcasting with potential multi-dimensional
        # outputs from the function or functions expecting specific input dimensions.
        # The function `func` is expected to return values that are broadcast-compatible,
        # typically resulting in a shape like (N_alpha, N_beta, ...)
        # The squeeze() method removes any trailing singleton dimensions (like the one we added
        # or any others returned by func), aiming for a final 2D result.
        # self.values shape: (N_alpha, N_beta)
        self.values = self.func(self.alpha_grid[:, :, None], self.beta_grid[:, :, None]).squeeze()

        # Basic check for output shape consistency
        expected_shape = self.grid_size
        if self.values.shape != expected_shape:
             print(f"Warning: Unexpected shape for self.values. "
                   f"Expected {expected_shape}, got {self.values.shape}. "
                   f"This might cause issues in plotting or argmax.")


    def plot_contours(self, ax, color='b', num_contours=10, f=lambda x: x, alpha=1.0, title=None):
        """
        Plot contour lines for the function evaluated on the grid.

        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis object to plot on.
            color (str or tuple):      Color of the contour lines. Default: 'b'.
            num_contours (int):        Number of contour levels to draw. Default: 10.
            f (callable):              A transformation applied to self.values before plotting.
                                       Useful for visualizing log-probabilities as probabilities
                                       (e.g., f=jnp.exp). Takes input shape (N_alpha, N_beta),
                                       returns shape (N_alpha, N_beta). Default: identity function.
            alpha (float):             Transparency of contour lines (0=transparent, 1=opaque). Default: 1.0.
            title (str, optional):     Optional title for the plot. If None, uses self.name. Default: None.
                                       Note: The current implementation always uses self.name.
                                       To use this parameter, modify the ax.set_title line.
        """
        # Apply the transformation function f to the grid values.
        # Input shape to f: (N_alpha, N_beta)
        # Output shape from f: (N_alpha, N_beta)
        plot_values = f(self.values)

        # Plot contours using the transformed values.
        # ax.contour expects X, Y, Z where Z has shape (num_Y, num_X).
        # Our self.alphas correspond to X (shape N_alpha), self.betas correspond to Y (shape N_beta).
        # Our plot_values has shape (N_alpha, N_beta).
        # To match contour's expectation, we need Z with shape (N_beta, N_alpha).
        # Therefore, we transpose plot_values using .T.
        # Input shapes to contour:
        #   self.alphas: (N_alpha,)
        #   self.betas:  (N_beta,)
        #   plot_values.T: (N_beta, N_alpha)
        ax.contour(self.alphas, self.betas, plot_values.T, num_contours, colors=color, alpha=alpha)

        # Label the axes using LaTeX formatting
        ax.set(xlabel='$\\alpha$', ylabel='$\\beta$')

        # Set the title of the plot
        plot_title = title if title is not None else self.name
        ax.set_title(plot_title, fontweight='bold')

    @property
    def argmax(self):
        """
        Return the (alpha, beta) values corresponding to the maximum value in the grid.

        This property identifies the grid point (alpha, beta) with the highest
        evaluated function value stored in self.values.

        Returns:
            tuple: A tuple (alpha_max, beta_max) representing the coordinates
                   of the maximum value found on the grid. Shape: (2,)
        """
        # self.values shape: (N_alpha, N_beta)
        # Find the index of the maximum value in the flattened grid values array.
        # idx is a scalar integer index into the flattened array.
        idx = jnp.argmax(self.values)

        # Convert the flat index `idx` to 2D indices (row, column) corresponding
        # to the original grid shape `self.grid_size` (N_alpha, N_beta).
        # The 'ij' indexing used in meshgrid means the first dimension corresponds to alphas
        # and the second to betas. Therefore, `unravel_index` will return (alpha_idx, beta_idx).
        # alpha_idx ranges from 0 to N_alpha-1
        # beta_idx ranges from 0 to N_beta-1
        alpha_idx, beta_idx = jnp.unravel_index(idx, self.grid_size)

        # Return the corresponding alpha and beta values from the original input arrays
        # using the computed indices.
        return self.alphas[alpha_idx], self.betas[beta_idx]