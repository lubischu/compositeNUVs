"""
Specifies a model to estimate the (potentially evolving) covariance matrices 
of applied zero-mean Gaussian noise. 
"""

import numpy as np
import pandas as pd
import time
from tqdm import trange

from src.nuvPriors.normalNode import normalNode
from src.models.pwcModel import PWCModel


class COVModel():
    """
    Model to estimate the covariance matrices of applied zero-mean Gaussian 
    noise Z_i. Thereby, the covariances can either assumed to be constant or 
    PWC (estimator must be initialized accordingly). Furthermore, it should be 
    noted that the following methods heavily rely on a quantity called 
    Vectoriced Inverse Cholesky Factor (VICF), which is denoted by J. The 
    dimension of J is fD = D*(D+1)/2, where D is the dimension of the given 
    observations.
    """
    
    def __init__(self, N: int, D: int, evolType: str='pwc'):
        """
        Args: 
            N (int): Number of observations.
            D (int): Number of dimensions.
            evolType (str): Type of evolution model for J (which specifies the 
                covariances). It can take values in ['pwc', 'constant'], 
                selecting either a PWC or constant model. Default is 'pwc'.
        """
        
        # Check if selected evolution type is known
        valid_evolType = ['pwc', 'constant']
        assert evolType in valid_evolType, f'Unknown evolType! Must be in ' + \
            f'{valid_evolType}.'
        
        # Calculate the dimension of J
        fD = int(D*(D+1)/2)
        
        # Initialize dimensions and evolution type
        self.N = N
        self.D = D
        self.fD = fD
        self.evolType = evolType
        
        # Construct J such that resulting noise covariance matrices all are 
        # identity matrices
        mj_init = np.concatenate(
            (np.ones((N,D), dtype=float), np.zeros((N,fD-D), dtype=float)), 
            axis=1)
        Vj_init = np.tile(np.identity(fD, dtype=float)*1e3, (N,1,1))
        
        # Initialize evolution model (either PWC or constant, depending on 
        # values of 'evolType')
        if evolType == 'pwc':
            self.evolModel = PWCModel(
                N=N, D=fD, mode='dual', mx_init=mj_init, Vx_init=Vj_init)
        else:
            self.evolModel = ConstModel(
                N=N, D=fD, mode='dual', mx_init=mj_init, Vx_init=Vj_init)
            
    def estimate_VICF(
            self, z_hat: np.ndarray, n_it_irls: int=1000, beta_l: float=5.0, 
            met_convTh: float=1e-4, beta_u: float=None, 
            disable_progressBar: bool=False) -> tuple[np.ndarray, int]:
        """
        Estimates the Vectorised Inverse Cholesky Factor (VICF), denoted by J. 
        In other words, J_i specifies A_i, where A_i is a Cholesky factor from 
        the estimated noise covariance at index i. 
        
        Args:
            z_hat (np.ndarray): Estimated observation noise, assumed to be 
                normally distributed with zero mean.
                    .shape=(N,D)
            n_it_irls (int): Maximum number of iterations for IRLS.
            beta_l (float): Tuning parameter for positivity constraint, must 
                be non-negative. Should be chosen chosen 'large enough', in 
                the sense that the diagonal elements of Vs_hat should always 
                be positive. 
            beta_u (float): Tuning parameter for sparsifying NUV in PWC model 
                (only used when evolType == 'pwc'). Higher values correspond 
                to a more aggressive prior. If None (default value), beta_u 
                will be chosen equal to fD, wich corresponds to a plain NUV.
            met_convTh (float): Threshold for convergence.
            disable_progressBar (bool): If False, the progress bar is shown. 
                If True, no progress bar is shown. Default is False.
            
        Returns:
            changes (np.ndarray): Array containing relative changes per 
                iteration of IRLS.
            i_it (int): Index of last iteration in IRLS, starting at 0. 
                Therefore, the number of performed iterations is i_it + 1.
            conv_time (float): Time for convergence (in seconds).
        """
        
        # Check dimensions of inputs
        assert z_hat.shape==(self.N,self.D), f'z_hat must be of .shape=(N,D)!'
        
        # Start timer
        start_time = time.time()
        
        if beta_u is None:
            beta_u = self.fD
        assert beta_u > 0.0, \
            f'beta_u must be chosen greater than zero (or None)!'
        
        changes = np.empty(n_it_irls, dtype=float)
        
        # Get current estimate of J
        mj_hat, _ = self.get_jHat()
        
        # Perform IRLS
        for i_it in trange(n_it_irls, disable=disable_progressBar):
            
            # Calculate backward messages through normal node
            xij_b, Wj_b = normalNode(
                z=z_hat, ms_hat=mj_hat, beta_l=beta_l, inverse=True)

            # Estimate J with specified evolution model
            if self.evolType == 'pwc':
                Wu_f = self.evolModel.sparseInputs_f(
                    beta_u=beta_u, inverse=True)
                self.evolModel.BIFM(xix_b=xij_b, Wx_b=Wj_b, Wu_f=Wu_f)
            else:
                self.evolModel.estimate_output(mxix_b=xij_b, VWx_b=Wj_b)
                
            # Calculate average relative change in J and update mj_hat
            mj_hat_new, _ = self.get_jHat()
            changes[i_it] = np.mean(
                np.abs(mj_hat - mj_hat_new) / np.abs(mj_hat))
            mj_hat = mj_hat_new
            
            # Check if IRLS has converged (i.e., if change is below threshold)
            if changes[i_it] < met_convTh:
                break
            
        # Stop timer and calculate time needed for convergence
        stop_time = time.time()
        conv_time = stop_time - start_time

        return changes, i_it, conv_time

            
    def get_jHat(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimates of the VICF, i.e., J.
        
        Returns:
            mj_hat (np.ndarray): Current mean estimates of J.
                    .shape=(N,2)
            Vj_hat (np.ndarray): Current covariance matrix estimates of S.
                    .shape(N,2,2)
        """
        
        mj_hat = self.evolModel.mx_hat
        Vj_hat = self.evolModel.Vx_hat
        
        return mj_hat, Vj_hat
    
    def calculate_noiseCov(self) -> np.ndarray:
        """
        Calculates the corresponding estimated noise covariance matrices for 
        the current estimates of J.
        
        Returns:
            noiseCov_hat (np.ndarray): Corresponding estimated noise 
                covariaces matrices.
                    .shape=(N,D,D)
        """
        
        # Get current estimates of J
        mj_hat, _ = self.get_jHat()
    
        # Construct A^{-1} from S
        A_inv = np.zeros((self.N,self.D,self.D), dtype=float)
        ind_start = 0
        for d in range(self.D):
            A_inv += np.array([
                np.diag(mj_hat_i[ind_start:ind_start+self.D-d], k=d) 
                for mj_hat_i in mj_hat])
            ind_start += self.D-d
        
        # Calculate covariance / precision matrices as the products of AA^T
        A = np.linalg.inv(A_inv)
        noiseCov_hat = A @ np.transpose(A, (0,2,1))
        
        return noiseCov_hat
    
    def calculate_noisePrec(self, inverse: bool=False):
        """
        Calculates the corresponding estimated noise precision matrices for 
        the current estimates of J.
        
        Returns:
            noiseprec_hat (np.ndarray): Corresponding estimated noise 
                precision matrices.
                    .shape=(N,D,D)
        """
        
        # Get current estimates of J
        mj_hat, _ = self.get_jHat()
    
        # Construct A^{-1} from J
        A_inv = np.zeros((self.N,self.D,self.D), dtype=float)
        ind_start = 0
        for d in range(self.D):
            A_inv += np.array([
                np.diag(mj_hat_i[ind_start:ind_start+self.D-d], k=d) 
                for mj_hat_i in mj_hat])
            ind_start += self.D-d
        
        # Calculate precision matrices from A^-1
        noisePrec_hat = np.transpose(A_inv, (0,2,1)) @ A_inv
        
        return noisePrec_hat
            

class ConstModel():
    """
    Model to calculate the constant MAP estimates of a vector quantity given 
    the incomming Gaussian messages.
    """
    
    def __init__(
            self, N: int, D: int, mode: str, mx_init: np.ndarray=None, 
            Vx_init: np.ndarray=None, mx_prior: np.ndarray=None, 
            Vx_prior: np.ndarray=None):
        """
        Args:
            N (int): Number of observations.
            D (int): Number of dimensions.
            mode (str): Either 'conventional' or 'dual', specifying by which 
                representation message passing is handled. The ingoing 
                messages needed for the estimation must be given in the 
                respective representation (i.e., mean and covariance for 
                'conventional' or dual mean and precision for 'dual').
            mx_init (np.ndarray): Initial values of mx_hat. If None, mx_hat is 
                initialized randomly, close to zero.
                    .shape=(N,D)
            Vx_init (np.ndarray): Initial values of Vx_hat. If None, Vx_hat is 
                initialized to identity matrices, scaled by 1e3 (i.e., very 
                uncertain about initialization).
                    .shape=(N,D,D)
            mx_prior (np.ndarray): Prior on mean of X. If None, mx_prior is 
                initialized to zero.
                    .shape=D
            Vx_prior (np.ndarray): Prior on covariance of X. If None, 
                Vx_prior is initialized to identity matrix, scaled by 1e3.
                    .shape=(D,D)
        """
        
        # Check dimensions of inputs
        assert mx_init is None or mx_init.shape==(N,D), \
            f'mx_init must be None or of .shape=(N,D)!'
        assert Vx_init is None or Vx_init.shape==(N,D,D), \
            f'Vx_init must be None or of .shape=(N,D,D)!'
        assert mx_prior is None or mx_prior.shape==D, \
            f'mx_prior must be None or of .shape=D!'
        assert Vx_prior is None or Vx_prior.shape==(D,D), \
            f'Vx_prior must be None or of .shape=(D,D)!'
        
        # Check if selected mode is known
        valid_mode = ['conventional', 'dual']
        assert mode in valid_mode, \
            f'mode={mode} is unknown! Valid modes are {valid_mode}'
        
        # Initialize dimensions and mode
        self.N = N
        self.D = D
        self.mode = mode
        
        # Initialize K
        if mx_init is None:
            self.mx_hat = np.random.normal(0.0, 1e-3, (N,D))
        else:
            self.mx_hat = mx_init
        if Vx_init is None:
            self.Vx_hat = np.tile(np.identity(D, dtype=float), (N,1,1))
        else:
            self.Vx_hat = Vx_init
            
        # Calculate dual representation of prior and save it
        if mx_prior is None:
            mx_prior = np.zeros(self.D, dtype=float)
        if Vx_prior is None:
            Vx_prior = np.identity(D, dtype=float)*1e3
            
        self.Wx_prior = np.linalg.inv(Vx_prior)
        self.xix_prior = self.Wx_prior @ mx_prior
            
    def estimate_output(self, mxix_b: np.ndarray, VWx_b: np.ndarray) -> None:
        """
        Calculate MAP estimates of mean and covariance matrices of X.
        
        Args:
            mxix_b (np.ndarray): Either interpreted as ingoing mean or dual 
                mean messages, depending on mode.
                    .shape=(N,D)
            VWx_b (np.ndarray): Either interpreted as ingoing covariance or 
                precision messages, depending on mode.
                    .shape=(N,D,D)
        """
        
        # Check dimensions of inputs
        assert mxix_b.shape==(self.N,self.D), \
            f'mxix_b must be of .shape=(N,D)!'
        assert VWx_b.shape==(self.N,self.D,self.D), \
            f'VWx_b must be of .shape=(N,D,D)!'

        if self.mode == 'conventional':
            # Calculate dual representations of given messages
            Wx_b = np.linalg.inv(VWx_b)
            xix_b = np.reshape(
                Wx_b @ np.reshape(mxix_b, (self.N,self.D,1)), (self.N,self.D))
            
            # Sum everything up (including prior knowledge)
            Wx_hat = self.Wx_prior + np.sum(Wx_b, axis=0)
            xix_hat = self.xix_prior + np.sum(xix_b, axis=0)
            
        else:
            # Sum given messages directly up
            Wx_hat = self.Wx_prior + np.sum(VWx_b, axis=0)
            xix_hat = self.xix_prior + np.sum(mxix_b, axis=0)
            
        # Calculate MAP estimate of K
        Vx_hat_one = np.linalg.inv(Wx_hat)
        mx_hat_one = np.reshape(
            Vx_hat_one @ np.reshape(xix_hat, (self.D,1)), self.D)
        
        # Repeat estimate to every time index (done to match format of other 
        # estimators)
        self.Vx_hat = np.tile(Vx_hat_one, (self.N,1,1))
        self.mx_hat = np.tile(mx_hat_one, (self.N,1))