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
    TODO
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
        
        valid_evolType = ['pwc', 'constant']
        assert evolType in valid_evolType, f'Unknown evolType! Must be in ' + \
            f'{valid_evolType}.'
        
        self.N = N
        self.D = D
        
        # Calculate dimension of J
        fD = int(D*(D+1)/2)
        self.fD = fD
        
        # Construct J such that resulting noise covariance matrices would all 
        # be identity matrices
        mj_init = np.concatenate(
            (np.ones((N,D), dtype=float), np.zeros((N,fD-D), dtype=float)), 
            axis=1)
        Vj_init = np.tile(np.identity(fD, dtype=float)*1e3, (N,1,1))
        
        # Initialize evolution model (either PWC or constant, depending on 
        # values of 'evolType')

        if evolType == 'pwc':
            self.evolModel = PWCModel(
                N=N, D=fD, mode='dual', mk_init=mj_init, Vk_init=Vj_init)
        else:
            self.evolModel = ConstModel(
                N=N, D=D, mode='dual', mk_init=mj_init, Vk_init=Vj_init)
            
    def estimate_VICF(
            self, z_hat: np.ndarray, n_it_irls: int=1000, beta_l: float=5.0, 
            met_convTh: float=1e-4, beta_u: float=None
            ) -> tuple[np.ndarray, int]:
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
            
        Returns:
            changes (np.ndarray): Array containing relative changes per 
                iteration of IRLS.
            i_it (int): Index of last iteration in IRLS, starting at 0. 
                Therefore, the number of performed iterations is i_it + 1.
        """
        
        if beta_u is None:
            beta_u = self.D
        assert beta_u > 0.0, \
            f'beta_u must be chosen greater than zero (or None)!'
        
        changes = np.empty(n_it_irls, dtype=float)
        
        # Perform IRLS
        for i_it in range(n_it_irls):
            
            # Calculate backward messages through normal node
            mj_hat, _ = self.get_jHat()
            xij_b, Wj_b = normalNode(
                z=z_hat, ms_hat=mj_hat, beta_l=beta_l, inverse=True)

            # Estimate J with specified evolution model
            if self.evolModel == 'pwc':
                Wu_f = self.evolModel.sparseInputs_f(
                    beta_u=beta_u, inverse=True)
                self.evolModel.BIFM(xik_b=xij_b, Wk_b=Wj_b, Wu_f=Wu_f)
            else:
                self.evolModel.estimate_output(mxik_b=xij_b, VWk_b=Wj_b)
                
            # Calculate average relative change in J
            mj_hat_new = self.get_jHat()
            changes[i_it] = np.mean(
                np.abs(mj_hat - mj_hat_new) / np.abs(mj_hat))
            
            # Check if IRLS has converged (i.e., if change is below threshold)
            if changes[i_it] < met_convTh:
                break

        return changes, i_it

            
    def get_jHat(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimates of the VICF, i.e., J.
        
        Returns:
            mj_hat (np.ndarray): Current mean estimates of J.
                    .shape=(N,2)
            Vj_hat (np.ndarray): Current covariance matrix estimates of S.
                    .shape(N,2,2)
        """
        
        mj_hat = self.evolModel.mk_hat
        Vj_hat = self.evolModel.Vk_hat
        
        return mj_hat, Vj_hat
            

class ConstModel():
    """
    Model to calculate the constant MAP estimates of a vector quantity given 
    the incomming Gaussian messages.
    """
    
    def __init__(
            self, N: int, D: int, mode: str, mk_init: np.ndarray=None, 
            Vk_init: np.ndarray=None, mk_prior: np.ndarray=None, 
            Vk_prior: np.ndarray=None):
        """
        Args:
            N (int): Number of observations.
            D (int): Number of dimensions.
            mode (str): Either 'conventional' or 'dual', specifying by which 
                representation message passing is handled. The ingoing 
                messages needed for the estimation must be given in the 
                respective representation (i.e., mean and covariance for 
                'conventional' or dual mean and precision for 'dual').
            mk_init (np.ndarray): Initial values of mk_hat. If None, mk_hat is 
                initialized randomly, close to zero.
                    .shape=(N,D)
            Vk_init (np.ndarray): Initial values of Vk_hat. If None, Vk_hat is 
                initialized to identity matrices, scaled by 1e3 (i.e., very 
                uncertain about initialization).
                    .shape=(N,D,D)
            mk_prior (np.ndarray): Prior on mean of K_1. If None, mk_prior is 
                initialized to zero.
                    .shape=D
            Vk_prior (np.ndarray): Prior on covariance of K_1. If None, 
                Vk_prior is initialized to identity matrix, scaled by 1e3.
                    .shape=(D,D)
        """
        
        valid_mode = ['conventional', 'dual']
        assert mode in valid_mode, \
            f'mode={mode} is unknown! Valid modes are {valid_mode}'
        
        self.N = N
        self.D = D
        self.mode = mode
        
        # Initialize K
        if mk_init is None:
            self.mk_hat = np.random.normal(0.0, 1e-3, (N,D))
        else:
            self.mk_hat = mk_init
        if Vk_init is None:
            self.Vk_hat = np.tile(np.identity(D, dtype=float), (N,1,1))
        else:
            self.Vk_hat = Vk_init
            
        # Calculate dual representation of prior and save it
        if mk_prior is None:
            mk_prior = np.zeros(self.D, dtype=float)
        if Vk_prior is None:
            Vk_prior = np.identity(D, dtype=float)*1e3
            
        self.Wk_prior = np.linalg.inv(Vk_prior)
        self.xik_prior = self.Wk_prior @ mk_prior
            
    def estimate_output(self, mxik_b: np.ndarray, VWk_b: np.ndarray) -> None:
        """
        Calculate MAP estimates of mean and covariance matrices of K.
        
        Args:
            mxik_b (np.ndarray): Either interpreted as ingoing mean or dual 
                mean messages, depending on mode.
                    .shape=(N,D)
            VWk_b (np.ndarray): Either interpreted as ingoing covariance or 
                precision messages, depending on mode.
                    .shape=(N,D,D)
        """

        if self.mode == 'conventional':
            # Calculate dual representations of given messages
            Wk_b = np.linalg.inv(VWk_b)
            xik_b = np.reshape(
                Wk_b @ np.reshape(mxik_b, (self.N,self.D,1)), (self.N,self.D))
            
            # Sum everything up (including prior knowledge)
            Wk_hat = self.Wk_prior + np.sum(Wk_b, axis=0)
            xik_hat = self.xi_prior + np.sum(xik_b, axis=0)
            
        else:
            # Sum given messages directly up
            Wk_hat = self.Wk_prior + np.sum(VWk_b, axis=0)
            xik_hat = self.xi_prior + np.sum(mxik_b, axis=0)
            
        # Calculate MAP estimate of K
        self.Vk_hat = np.tile(np.linalg.inv(Wk_hat), (self.N,1,1))
        self.mk_hat = np.reshape(
            self.Vk_hat @ np.reshape(xik_hat, (self.N,self.D,1)), 
            (self.N,self.D))