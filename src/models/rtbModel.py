"""
Specifies a model to fit piecewise constant data with a preference to a 
specified base level. Therefore the name Return-To-Base (RTB) model.
"""

import numpy as np
import pandas as pd
import time
from tqdm import trange

from src.models.pwcModel import PWCModel
from src.models.modelSelector import ModelSelector

class RTBModel():
    """
    Model for fitting piecewise-constant levels to N observations in 
    D-dimensions, with a preference to a reoccuring known level (base). 
    Therefore the nameing to return-to-base (RTB) model. The general model 
    consists of a PWC model and a model selector, where the model selector 
    specifies whether the data at the current time index shall be explained 
    by the PWC model or the prefered constant level. 
    """
    
    def __init__(
            self, N: int, D: int, sigmas: np.ndarray, constLevel: np.ndarray, 
            mx_init: np.ndarray=None, Vx_init: np.ndarray=None):
        """
        Args:
            N (int): Number of observations.
            D (int): Number of dimensions.
            sigmas (np.ndarray): Observation noise covariance matrices for 
                both, the known constant and the PWC model. Note that a higher 
                variace makes the model more likely to be picked. Therefore, 
                the covarinace matrix at index 0 (corresponding to the 
                constant model) should be larger than the second one.
                    .shape=(2,D,D)
            constLevel (np.ndarray): Points to constant level.
                    .shape=D
        """
        
        self.N = N
        self.D = D
        self.constLevel = constLevel
        
        self.pwcModel = PWCModel(
            N=N, D=D, mode='dual', mk_init=mx_init, Vk_init=Vx_init)
        self.modelSelector = ModelSelector(N=N, M=2, sigmas=sigmas)
        
    def estimate_output(
            self, y: np.ndarray,n_it_outer: int=1000, n_it_irls_x: int=10, 
            n_it_irls_s: int=1, beta_u_x: float=None, beta_u_s: float=2.0, 
            beta_l_s: float=1.0, beta_h_s: float=1.0, met_convTh: float=1e-4, 
            diff_convTh: float=1e-3, disable_progressBar: bool=False
            ) -> tuple[pd.DataFrame, int, float]:
        """
        Estimates the output by iteratively improving the estimates of the PWC 
        model outputs and the model selection at each time step. 
        
        Args:
            y (np.ndarray): Observations.
                    .shape=(N,D)
            n_it_outer (int): Maximum number of outer iterations, i.e., 
                iterations between the two improvement steps.
            n_it_irls_x (int): Maximum number of iterations of IRLS for the 
                estimation of the outputs of the PWC model.
            n_it_irls_s (int): Maximum number of iterations of IRLS for the 
                model selection.
            beta_u_x (float): Tuning parameter for sparse input NUV for the 
                estimation of X (i.e., outputs of PWC model), must be positive.
            beta_u_s (float): Tuning parameter for sparse input NUV for the 
                estimation of S, must be positive.
            beta_l_s (float): Tuning parameter for positivity constraint for 
                the estimation of S, must be positive.
            beta_h_s (float): Tuning parameter for One-Hot constraint for the 
                estimation of S, must be non-negative.
            met_convTh (float): Threshold for convergence, checking change of 
                PWC model output.
            diff_convTh (float): Threshold for convergence, checking how far 
                away S is from all-{0,1} solution.
            disable_progressBar (bool): If False, the progress bar is shown. 
                If True, no progress bar is shown. Default is False.
                
        Returns:
            performanceMetrics (pd.DataFrame): Contains performance metrics in 
                the format: ['change_x_min', 'i_it_x', 'diffAZOSol_s', 
                'i_it_s'].
            i_it_outer (int): Index of last outer iteration, starting at 0. 
                Therefore, the number of performed iterations is 
                i_it_outer + 1.
            conv_time (float): Time for convergence (in seconds).
        """
        
        # Start timer
        start_time = time.time()
        
        # Pandas DataFrame to save performance metrics
        performanceMetrics = pd.DataFrame(
            index=range(n_it_outer), 
            columns=['change_x_min', 'i_it_x', 'diffAZOSol_s', 'i_it_s'])
        
        for i_it_outer in trange(n_it_outer, disable=disable_progressBar):
            
            # Get estimates of S
            ms_hat, _ = self.modelSelector.get_sHat()
    
            # Get 'ingoing' backward messages into PWC model
            Wx_b = \
                np.reshape(ms_hat[:,1]**2, (self.N,1,1)) * \
                np.linalg.inv(self.modelSelector.sigmas[:,1])
            xix_b = np.reshape(
                Wx_b @ np.reshape(y, (self.N,self.D,1)), (self.N,self.D))
    
            # Estimate outputs of piecewise-constant model
            change_x, i_it_x = self.pwcModel.estimate_output(
                n_it_irls=n_it_irls_x, mxik_b=xix_b, VWk_b=Wx_b, 
                beta_u=beta_u_x, met_convTh=met_convTh)
            change_x_min = np.min(change_x[:i_it_x+1])
    
            # Construct estimated outputs per model
            x_m0 = np.tile(self.constLevel, (self.N,1)) 
            x_m1 = self.pwcModel.mk_hat
            x_perModel = np.concatenate(
                (x_m0[:,np.newaxis,:], x_m1[:,np.newaxis,:]), axis=1)
            
            # Estimate model selection
            diffAZOSol_s, i_it_s = self.modelSelector.estimate_selectedModel(
                n_it_irls=n_it_irls_s, y=y, x_perModel=x_perModel, 
                beta_l=beta_l_s, beta_h=beta_h_s, beta_u=beta_u_s, 
                diff_convTh=diff_convTh)
            diffAZOSol_min = np.min(diffAZOSol_s[:i_it_s+1])
        
            # Save preformance metrics
            performanceMetrics.loc[i_it_outer] = \
                [change_x_min, i_it_x, diffAZOSol_min, i_it_s]
    
            # Check convergence
            if change_x_min < met_convTh and diffAZOSol_min < diff_convTh:
                break
            
        # Stop time and calculate convergence time
        stop_time = time.time()
        conv_time = stop_time - start_time
        
        return performanceMetrics, i_it_outer, conv_time
            
    def get_sHat(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimates of S.
        
        Returns:
            ms_hat (np.ndarray): Current mean estimates of S.
                    .shape=(N,2)
            Vs_hat (np.ndarray): Current covariance matrix estimates of S.
                    .shape(N,2,2)
        """
        
        ms_hat, Vs_hat = self.modelSelector.get_sHat()
        
        return ms_hat, Vs_hat
    
    def get_xHat(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimated outputs of the PWC model.
        
        Returns:
            mx_hat (np.ndarray): Current mean estimates of the output of PWC 
                model.
                    .shape=(N,D)
            Vx_hat (np.ndarray): Current covariance matrix estimates of the 
                output of PWC model.
                    .shape(N,D,D)
        """
        
        mx_hat = self.pwcModel.mk_hat
        Vx_hat = self.pwcModel.Vk_hat
        
        return mx_hat, Vx_hat
            
    def get_output(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimated outputs, where at each time index i the 
        output of the model for which ms_hat[i] is maximal is returned.
        
        Returns:
            output (np.ndarray): Current estimated outputs.
                    .shape=(N,D)
        """
        
        # Determine chosen model at each time index
        ms_hat, _ = self.get_sHat()
        selectedModel = np.argmax(ms_hat, axis=1)
        
        # Get outputs of PWC model
        mk_hat, _ = self.get_xHat()
        
        # Construct corresponding output
        output = np.where(
            selectedModel[:,np.newaxis]==0, 
            np.tile(self.constLevel, (self.N,1)), mk_hat)
        
        return output