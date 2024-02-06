"""
Specifies a model to fit constant data.
"""    

import numpy as np
import pandas as pd

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