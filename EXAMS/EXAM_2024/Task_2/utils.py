from rich import print as rprint
import numpy as np

def rlabel(label, color="light_salmon3"):
    """Print label in color, value in default style"""
    rprint(f"[{color}]{label}[/{color}]")


def stars(n=50):
    """Print n stars"""
    rprint(f"[light_salmon3]{'*' * n}[/light_salmon3]")
    
def s(n=50):
    print("*" * n)
    
    
def rbf(X1, X2, length_scale=1.0, kappa=1.0, debug=False):
    """
    Computes a “squared exponential style” kernel (here using an L₁ based form) plus a linear term.
    
    Parameters
    ----------
    X1 : np.ndarray, shape (n1, d)
        First batch of points (n1 points in d dimensions).
    X2 : np.ndarray, shape (n2, d)
        Second batch of points (n2 points in d dimensions).
    length_scale : float
        Lengthscale parameter ell > 0 (default 1.0).
    variance : float
        Variance (amplitude) parameter σ² (default 1.0).  Currently not used in your return statement,
        but you could multiply the whole kernel by this if desired.
    
    Returns
    -------
    K : np.ndarray, shape (n1, n2)
        Kernel matrix where
          K[i,j] = 1 * (1 + (‖X1[i] − X2[j]‖₁ / (2 ℓ²)))⁻¹  +  X1[i] · X2[j]
    """
    # ensure length_scale is positive and nonzero
    l = np.abs(length_scale) + 1e-12
    

    
    # 2) Compute pairwise differences via broadcasting:
    #    diff[i,j,k] = X1[i,k] - X2[j,k]
    #    shape of diff: (n1, n2, d)
    diff = X1[:, None, :] - X2[None, :, :]
    
    # 3) Sum absolute differences over the last axis to get L1 distance:
    #    sqdist[i,j] = sum_k |diff[i,j,k]|
    #    shape of sqdist: (n1, n2)*
    sqdist = np.sum((diff)**2, axis=2)
    
    # 4) Compute the kernel:
    #    A) “Squared‐exponential–style” term (but using L1 distance here):
    #         (1 + sqdist / (2 ℓ²))⁻¹
    #    B) Plus a linear term X1·X2ᵀ
    #    Final shape: (n1, n2)
    
    
    if debug:
        print("*" * 50)
        print("Debugging information:")
        
        print("Length scale (l):", l)
        # 1) Print shapes for debugging
        #   X1 shape: (n1, d)
        print("X1 shape before:", X1.shape)
        #   After adding a new axis: (n1, 1, d)
        print("X1[:, None, :] shape:", X1[:, None, :].shape)
        #   X2 shape: (n2, d)
        print("X2 shape before:", X2.shape)
        #   After adding a new axis: (1, n2, d)
        print("X2[None, :, :] shape:", X2[None, :, :].shape)
        #  diff shape: (n1, n2, d)
        print("diff shape:", diff.shape)
        #  sqdist shape: (n1, n2)
        print("sqdist shape:", sqdist.shape)
        #  K shape: (n1, n2)
        print("K shape:", (1 * (1 + (sqdist / (2 * l**2)))**(-1) + X1 @ X2.T).shape)
        print("*" * 50)
        print("*" * 50)
    
    return kappa**2 * np.exp(-sqdist / (2 * length_scale**2))

def abs_kernel(X1, X2, length_scale=1.0, c1=1.0, c2=1.0, debug=False):
    """
    Computes a kernel function combining a modified L1-distance based kernel 
    with a linear term.
    
    The kernel has two main components:
    1. A dampened inverse kernel based on L1 (Manhattan) distance
    2. A linear dot product term
    
    Parameters
    ----------
    X1 : np.ndarray, shape (n1, d)
        First batch of input points (n1 points in d dimensions).
    X2 : np.ndarray, shape (n2, d)
        Second batch of input points (n2 points in d dimensions).
    length_scale : float, optional (default=1.0)
        Lengthscale parameter that controls the kernel's smoothness.
        Smaller values make the kernel more sensitive to point differences.
    c1 : float, optional (default=1.0)
        Scaling factor for the L1-distance based kernel component.
    c2 : float, optional (default=1.0)
        Scaling factor for the linear kernel component.
    debug : bool, optional (default=False)
        If True, prints detailed debugging information about kernel computation.
    
    Returns
    -------
    K : np.ndarray, shape (n1, n2)
        Kernel matrix where each entry K[i,j] is computed as:
        c1 * (1 + L1_distance(X1[i], X2[j]) / (2 * length_scale²))⁻¹ 
        + c2 * (X1[i] · X2[j])
    """
    # Ensure length_scale is positive to prevent numerical instability
    # Add a small epsilon to prevent division by zero
    l = np.abs(length_scale) + 1e-12
    
    # Compute pairwise differences using broadcasting
    # This creates a 3D array of differences between all points in X1 and X2
    # diff[i,j,k] represents the difference between X1[i,k] and X2[j,k]
    # Shape transformation:
    #   X1: (n1, d) -> X1[:, None, :]: (n1, 1, d)
    #   X2: (n2, d) -> X2[None, :, :]: (1, n2, d)
    #   diff: (n1, n2, d)
    diff = X1[:, None, :] - X2[None, :, :]
    
    # Compute L1 (Manhattan) distance by summing absolute differences
    # This reduces the 3D difference array to a 2D distance matrix
    # absdist[i,j] is the L1 distance between X1[i] and X2[j]
    # Shape: (n1, n2)
    absdist = np.sum(np.abs(diff), axis=2)
    
    # Kernel computation combines two terms:
    # 1. L1-distance based kernel: 
    #    (1 + absdist / (2 * l²))⁻¹ 
    #    This acts as a dampened similarity measure
    # 2. Linear kernel: 
    #    X1 @ X2.T 
    #    This adds a linear relationship between points
    # Scaling factors c1 and c2 allow independent control of each term
    kernel = c1 * (1 + absdist / (2 * l**2))**(-1) + c2 * (X1 @ X2.T)
    
    # Optional debugging information
    if debug:
        print("*" * 50)
        print("Debugging information:")
        print("Length scale (l):", l)
        print("X1 shape before:", X1.shape)
        print("X1[:, None, :] shape:", X1[:, None, :].shape)
        print("X2 shape before:", X2.shape)
        print("X2[None, :, :] shape:", X2[None, :, :].shape)
        print("diff shape:", diff.shape)
        print("absdist shape:", absdist.shape)
        print("*" * 50)
    
    return kernel