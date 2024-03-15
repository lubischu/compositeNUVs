"""
Specifies a model to estimate both, mean and covariance matrices of applied 
data in a hierarchical fashion.
"""

import time

import numpy as np
import pandas as pd
from tqdm import trange

from .constModel import ConstModel
from .covModel import CovModel
from .pwcModel import PWCModel


class HiModel():
    """
    Model to estimate both, the mean and covariance matrices of applied data. 
    Both can either assumed to be constant or PWC.
    """
    
    def __init__(
            self, N: int, D: int, evolType_mean: str='pwc', evolType_cov: 
            str='pwc'):
        """
        Args: 
            N (int): Number of observations.
            D (int): Number of dimensions.
            evolType_mean (str): Type of evolution model for mean. It can take 
                values in ['pwc', 'constant'], selecting either a PWC or 
                constant model. Default is 'pwc'.
            evolType_cov (str): Type of evolution model for covariance 
                matrices. It can take values in ['pwc', 'constant'], selecting 
                either a PWC or constant model. Default is 'pwc'.
        """
        
        # Check if selected evolution types are known
        valid_evolType = ['pwc', 'constant']
        assert evolType_mean in valid_evolType, \
            f'Unknown evolType_mean! Must be in {valid_evolType}.'
        assert evolType_cov in valid_evolType, \
            f'Unknown evolType_cov! Must be in {valid_evolType}.'
        
        # Calculate the dimension of J
        fD = int(D*(D+1)/2)
        
        # Initialize dimensions and evolution types
        self.N = N
        self.D = D
        self.fD = fD
        self.evolType_mean = evolType_mean
        self.evolType_cov = evolType_cov
        
        # Initialize evolution models
        if evolType_mean == 'pwc':
            self.evolModel_mean = PWCModel(N=N, D=D, mode='conventional')
        else:
            self.evolModel_mean = ConstModel(N=N, D=D, mode='conventional')
        self.evolModel_cov = CovModel(N=N, D=D, evolType=evolType_cov)

    def estimate_output(
            self, y: np.ndarray, n_it_outer: int=100, n_it_irls_mean: int=10, 
            n_it_irls_cov: int=10, beta_u_mean: float=None, 
            beta_u_cov: float=None, beta_l_cov: float=5.0, 
            met_convTh_mean: float=1e-4, met_convTh_cov: float=1e-4, 
            disable_progressBar: bool=False
            ) -> tuple[pd.DataFrame, int, float]:
        """
        Estimates outputs of both models.
        
        Args:
            y (np.ndarray): Observations.
                    .shape=(N,D)
            n_it_outer (int): Maximal number of outer loops. I.e., number of 
                iterations between mean and covariance estimation.
            n_it_irls_mean (int): Maximum number of iterations in the inner 
                loop (i.e., IRLS) for mean estimation.
            n_it_irls_cov (int): Maximum number of iterations in the inner 
                loop (i.e., IRLS) for covariance estimation.
            beta_u_mean (float): Tuning parameter for sparsifying NUV prior in 
                mean estimation. If None (default value), beta_u_mean will be 
                chosen equal to D, wich corresponds to a plain NUV. 
                If evolType_mean='constant', this is not used!
            beta_u_cov (float): Tuning parameter for sparsifying NUV prior in 
                covariance estimation. If None (default value), beta_u_cov 
                will be chosen equal to fD, wich corresponds to a plain NUV. 
                If evolType_cov='constant', this is not used!
            beta_l_cov (float): Tuning parameter for positivity constraint in 
                covariance estimation, must be non-negative. Should be chosen 
                'large enough', in the sense that the diagonal elements of 
                Vs_hat should always be positive. 
            met_convTh_mean (float): Threshold for convergence in mean 
                estimation.
            met_convTh_cov (float): Threshold for convergence in covariance 
                estimation.
            disable_progressBar (bool): If False, the progress bar is shown. 
                If True, no progress bar is shown. Default is False.
                
        Returns:
            performanceMetrics (pd.DataFrame): Contains performance metrics in 
                the format: ['change_mean', 'i_it_mean', 'change_cov', 
                'i_it_cov'].
            i_it_outer (int): Index of last outer iteration, starting at 0. 
                Therefore, the number of performed iterations is 
                i_it_outer + 1.
            conv_time (float): Time for convergence (in seconds).
        """
        
        # Check dimensions of inputs
        assert y.shape==(self.N,self.D), f'y must be of .shape=(N,D)!'
        
        # Start timer
        start_time = time.time()
            
        # Initialize X to mean of given observations. Must only been done if 
        # evolution model of mean is PWC!
        if self.evolType_mean=='pwc':
            mx_init = np.tile(np.mean(y, axis=0), (self.N,1))
            Vx_init = np.tile(
                np.identity(self.D, dtype=float)*1e3, (self.N,1,1))
            self.evolModel_mean.init_x(mx_init=mx_init, Vx_init=Vx_init)
    
        # Pandas DataFrame to save performance metrics
        performanceMetrics = pd.DataFrame(
            index=range(n_it_outer), 
            columns=['change_mean', 'i_it_mean', 'change_cov', 'i_it_cov'])
        
        # Iteratively estimate outputs of mean and covariance models
        for i_it_outer in trange(n_it_outer, disable=disable_progressBar):
            
            # Estimate outputs of mean model for fixed J
            sigmas_hat = self.get_sigmasHat()
            if self.evolType_mean=='pwc':
                changes_mean, i_it_mean, _ = \
                    self.evolModel_mean.estimate_output(
                        mxix_b=y, VWx_b=sigmas_hat, n_it_irls=n_it_irls_mean, 
                        beta_u=beta_u_mean, met_convTh=met_convTh_mean, 
                        disable_progressBar=True)
                change_mean = np.min(changes_mean[:i_it_mean+1])
            else:
                change_mean = self.evolModel_mean.estimate_output(
                        mxix_b=y, VWx_b=sigmas_hat)
                i_it_mean = 1
                
            # Estimate outputs of cov model for fixed X
            mx_hat, _ = self.get_xHat()
            z_hat = y - mx_hat
            changes_cov, i_it_cov, _ = self.evolModel_cov.estimate_VICF(
                z_hat=z_hat, n_it_irls=n_it_irls_cov, beta_u=beta_u_cov, 
                beta_l=beta_l_cov, met_convTh=met_convTh_cov, 
                disable_progressBar=True)
            change_cov = np.min(changes_cov[:i_it_cov+1])
        
            # Save performance metrics
            performanceMetrics.loc[i_it_outer] = [
                change_mean, i_it_mean, change_cov, i_it_cov]
        
            # Check convergence
            if change_mean < met_convTh_mean and change_cov < met_convTh_cov:
                f'Converged after {i_it_outer + 1} iterations.'
                break
            
        # Stop timer and calculate time needed for convergence
        stop_time = time.time()
        conv_time = stop_time - start_time
            
        return performanceMetrics, i_it_outer, conv_time

    def get_xHat(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimates of X (i.e., the means).
        
        Returns:
            mx_hat (np.ndarray): Current mean estimates of X
                    .shape=(N,D)
            Vx_hat (np.ndarray): Current covariance matrix estimates of X
                    .shape(N,D,D)
        """
        
        mx_hat = self.evolModel_mean.mx_hat
        Vx_hat = self.evolModel_mean.Vx_hat
        
        return mx_hat, Vx_hat

            
    def get_jHat(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimates of J (i.e., the VICF).
        
        Returns:
            mj_hat (np.ndarray): Current mean estimates of J.
                    .shape=(N,2)
            Vj_hat (np.ndarray): Current covariance matrix estimates of J.
                    .shape(N,2,2)
        """
        
        mj_hat, Vj_hat = self.evolModel_cov.get_jHat()
        
        return mj_hat, Vj_hat
    
    def get_sigmasHat(self) -> np.ndarray:
        """
        Returns current estimates of the noise covariance matrices..
        
        Returns:
            sigmas_hat (np.ndarray): Current estimates of noise covariance 
                matrices (based on mj_hat).
        """
        
        sigmas_hat = self.evolModel_cov.calculate_noiseCov()
        
        return sigmas_hat