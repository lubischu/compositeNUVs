"""
Specifies function to calculate messages passes through normal nodes.
"""

import numpy as np
import warnings

def normalNode(
        z: np.ndarray, ms_hat: np.ndarray, beta_l: float, inverse: bool=True
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates NUV messages through normal nodes with zero-mean. Thereby, 
    the generated outputs by the normal node are assumed to be perfectly known 
    (given in z). Note the difference in dimension between the normal 
    node outputs z (D-dimensional) and it inputs S (fD-dimensional). The 
    relationship between fD and D is 
        fD = D*(D+1)/2 ,
    following from the definition of S.
    
    Args:
        z (np.ndarray): Generated and observed outputs of normal node.
                .shape=(N,D)
        ms_hat (np.ndarray): Mean estimates of S from previous iteration.
                .shape=(N,fD)
        beta_l (float): Tuning parameter for positivity NUV prior. Higher 
            values correspond to a more aggressive prior, increasing 
            stability while decreasing speed of convergence.
        inverse (bool): If False, the representation of the outgoing 
            messages is in terms of their mean and covariance matrices. If 
            True, their inverses (i.e., the dual mean and precision matrices) 
            are given. Default is True.
                
    Returns:
        mxis_out (np.ndarray): Outgoing messages either by their mean or dual 
            mean representation.
                .shape=(N,fD)
        VWs_out (np.ndarray): Outgoing messages either by their covariance 
            or precision matrix representation.
                .shape=(N,fD,fD)
    """
    
    # Get dimensions
    N,D = z.shape
    _,fD = ms_hat.shape

    assert int(D*(D+1)/2) == fD, \
        f'D and fD do not match! They must be related as fD=D*(D+1)/2.'
    if not inverse:
        warnings.warn(
            f'Calculating the mean and covariance messages involves much ' + \
            f'more matrix inversions than their dual mean and precision ' + \
            f'representations. This is especially problematic in high ' + \
            f'dimensions!')
        
    # Construct Z_t matrix
    Z_t = constuctZTilde(z=z)
    
    # Calculate precision messages out of constraint nodes (per dimension!)
    Wc_b_im_woZeros = \
        (beta_l*np.abs(ms_hat[:,:D]) - 1) / ms_hat[:,:D]**2   # .shape=(N,D)
    Wc_b_im = np.concatenate(
        (Wc_b_im_woZeros, np.zeros((N,fD-D), dtype=float)), 
        axis=1)   # .shape=(N,fD)
    
    # Construct backward precision messages from constraint nodes
    Wc_b = np.array([np.diag(Wc_b_im_i) for Wc_b_im_i in Wc_b_im])
        # .shape=(N,fD,fD)
    
    # Calcualte dual mean and precission backward messages through S (dual 
    # representation)
    xis_out = np.concatenate(
        (np.full((N,D), beta_l), 
         np.zeros((N,fD-D), dtype=float)), 
         axis=1)   # .shape=(N,fD)
    Ws_out = np.transpose(Z_t, (0,2,1))@Z_t + Wc_b   # .shape=(N,fD,fD)
    
    if inverse:
        mxis_out = xis_out
        VWs_out = Ws_out
        
    # If the mean and covariance representations are needed
    else:
        VWs_out = np.linalg.inv(Ws_out)
        mxis_out = np.reshape(VWs_out @ np.reshape(xis_out, (N,fD,1)), (N,fD))
    
    return mxis_out, VWs_out
        
def constuctZTilde(z: np.ndarray) -> np.ndarray:
    """
    Constructs Z_tilde matrices.
        
    Args: 
        z (np.ndarray): Generated and observed outputs of normal node.
                .shape=(N,D)
                    
    Returns:
        Z_t (np.ndarray): Restulting Z_tilde matrices.
                .shape=(N,D,fD)
    """
    
    # Get dimensions
    N,D = z.shape
    
    # Construct Z_tilde per j
    Z_t_j = []
    for j in range(D):
        
        # Construct diagonal matrices for current j
        eye_D = np.identity(D-j, dtype=float)
        Z_diag_j = z[:,j:,np.newaxis] * eye_D[np.newaxis,:,:]
            # .shape=(N,D-j,D-j)
        
        # Construct Z_t_j and append
        if j == 0:
            Z_t_j.append(Z_diag_j)
        else:
            Z_t_j.append(
                np.concatenate(
                    (Z_diag_j, 
                        np.zeros((N,j,D-j), dtype=float)), 
                        axis=1)
                )   # .shape=(N,D,D-j)
        
    # Construct final Z_t
    Z_t = np.concatenate(Z_t_j, axis=2)   # .shape=(N,D,fD
    
    return Z_t