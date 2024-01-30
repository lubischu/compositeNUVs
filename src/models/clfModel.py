"""
Specifies a model to fit data to M different levels, where the levels are 
either specified or estimated.
"""

import numpy as np
import pandas as pd
import time
from tqdm import trange
import warnings

from src.models.modelSelector import ModelSelector

class CLFModel():
    """
    Model for fitting M constant levels to N observations in D-dimensions, 
    therefore the name constant-level-fitting (CLF) model. The speciality is 
    that we restict ourselfs to a finite (i.e., M) number of levels, where the 
    (exctly) same level can potentially be revisitied multiple times over the 
    considered observation horizon. The over-all model consits of M constant 
    models and a model selector, where the model selector specifies which 
    observation is best explained by which constant level. Furthermore, we 
    assume sparse model (i.e., level) changes.
    """
    
    def __init__(
            self, N: int, M: int, D:int , knownLevels: np.ndarray=None, 
            knwonSigmas: np.ndarray=None, sh_squared: float=1e-6): 
        """
            N (int): Number of observations.
            M (int): Number of (assumed) models.
            D (int): Number of dimensions.
            knownLevels (np.ndarray): A priori known levels (for example the 
                all-zero base level). Note that the array must be exactly 
                2-dimensional, even if only one level is given!
                    .shape=(1toM,D) or None
            knwonSigmas (np.ndarray): Assumed observation noises per level. If 
                None or knwonSigmas.shape[0]<M, the remaining covariance 
                matrices are initialized to identity matrices.
                    .shape=(1toM,D,D)
            sh_squared (float): Uncertainty / slack associated with condition 
                that sum over S must be one. Should be chosen small (i.e., 
                close to zero). Equal to zero should work too, but tends to
                be quite unstable.
        """
        
        self.N = N
        self.M = M
        self.D = D
        
        # Calculate / initialize prior knowledge about levels (dual 
        # representation, as this is used later)
        ml_prior = np.zeros((M,D), dtype=float)
        Vl_prior = np.tile(
            np.identity(D, dtype=float)*1e3, (M,1,1))
        if knownLevels is not None:
            for i, level in enumerate(knownLevels):
                ml_prior[i] = level.copy()
                Vl_prior[i] = np.identity(D, dtype=float)*1e-3
        self.Wl_prior = np.linalg.inv(Vl_prior)
        self.xil_prior = np.reshape(
            self.Wl_prior@np.reshape(ml_prior, (M,D,1)), (M,D))
                
        # Initialize estimates of levels slightly randomized to break symmetry
        self.ml_hat = ml_prior + np.random.normal(0.0, 1.0, (M,D))
        self.Vl_hat = Vl_prior
                
        # Initialize sigmas. If None or sigmas.shape[0]<M, initialize unknowns 
        # to identity matrices
        sigmas = np.tile(np.identity(D, dtype=float), (M,1,1))
        if knwonSigmas is not None:
            for i, sigma in enumerate(knwonSigmas):
                sigmas[i] = sigma
                
        # Initialize model selector accordingly
        self.modelSelector = ModelSelector(
            N=N, M=M, sigmas=sigmas, sh_squared=sh_squared)
                
    def estimate_output(
            self, y: np.ndarray, n_it_outer: int=1000, n_it_irls_s: int=1, 
            beta_u_s: float=None, beta_h_s: float=5.0,  beta_l_s: float=10.0,
            levelEstType: str='superPos', met_convTh: float=1e-3, 
            diff_convTh: float=1e-3, disable_progressBar: bool=False
            ) -> tuple[pd.DataFrame, np.ndarray, int, float]:
        """
        Estimates M constant levels as well as the corresponding model 
        selection.
        
        Args:
            y (np.ndarray): Observations.
                    .shape=(N,D)
            n_it_outer (int): Maximal number of outer loops. I.e., number of 
                iterations between model selection and estimation of levels.
            n_it_irls_s (int): Maximum number of iterations in the inner loop 
                (i.e., IRLS) for model selection.
            beta_u_s (float): Tuning parameter for sparsifying NUV prior in 
                model selection. If None (default value), beta_u_s will be 
                chosen equal to M, wich corresponds to a plain NUV.
            beta_h_s (float): Tuning parameter for One-Hot constraint in 
                model selection, must be non-negative. Default is 5.0.
            beta_l_s (float): Tuning parameter for positivity constraint in 
                model selection, must be non-negative.
            levelEstType (str): Specifies the method by which the levels are 
                estimated for given model selection (i.e., fixed S). Possible 
                types are ['superPos', 'selective']. They correspond to
                    'superPos': Observations are weithed by S_{i,m}
                    'selective': Only observations for which this model is 
                        actually selected are considered.
            met_convTh (float): Threshold for convergence when checking 
                relative change of level estimates.
            diff_convTh (float): Threshold for convergence when checking 
                averaged difference of S from all-{0,1} solution.
            disable_progressBar (bool): If False, the progress bar is shown. 
                If True, no progress bar is shown. Default is False.
                
        Returns:
            performanceMetrics (pd.DataFrame): Contains performance metrics in 
                the format: ['diffAZOSol_s_min', 'i_it_s', 'change_ml'].
            ml_hat_evol (np.ndarray): Contains evolution of level estimates.
                    .shape=(N,M,D)
            i_it_outer (int): Index of last outer iteration, starting at 0. 
                Therefore, the number of performed iterations is 
                i_it_outer + 1.
            conv_time (float): Time for convergence (in seconds).
        """
        
        valid_levelEstType = ['superPos', 'selective']
        assert levelEstType in valid_levelEstType, \
            f'levelEstType={levelEstType} is unknown! Valid modes ' + \
            f'are {valid_levelEstType}'
        
        # Start timer
        start_time = time.time()
        
        # If no beta_u_s is given, set it to 1.0 (good choice according to 
        # simulations)
        if beta_u_s is None:
            beta_u_s = 1.0
    
        # Pandas DataFrame to save performance metrics
        performanceMetrics = pd.DataFrame(
            index=range(n_it_outer), 
            columns=['diffAZOSol_s_min', 'i_it_s', 'change_ml'])
        ml_hat_evol = np.empty((n_it_outer,self.M,self.D))
        
        # Outer loop
        for i_it_outer in trange(n_it_outer, disable=disable_progressBar):
            
            # Improve model selection for current estimates of levels
            x_perModel = np.tile(self.ml_hat, (self.N,1,1))
            diffAZOSol, i_it_s = self.modelSelector.estimate_selectedModel(
                y=y, x_perModel=x_perModel, n_it_irls=n_it_irls_s, 
                beta_l=beta_l_s, beta_h=beta_h_s, beta_u=beta_u_s)
            
            # Improve level estimation
            ml_hat_old = self.ml_hat.copy()   # .shape=(M,D)
            if levelEstType == 'superPos':
                self.levelEstimation_superPos(y=y)
            else:
                self.levelEstimation_selective(y=y)
                
            # Calculate average change of estimated levels
            change_ml = np.mean(np.abs(ml_hat_old  - self.ml_hat))
        
            # Save performance metrics
            diffAZOSol_min = np.min(diffAZOSol[:i_it_s+1])
            performanceMetrics.loc[i_it_outer] = \
                [diffAZOSol_min, i_it_s, change_ml]
            ml_hat_evol[i_it_outer] = self.ml_hat.copy()
        
            # Check convergence
            if change_ml < met_convTh and diffAZOSol_min < diff_convTh:
                f'Converged after {i_it_outer + 1} iterations.'
                break
            
        # Stop timer and calculate time needed for convergence
        stop_time = time.time()
        conv_time = stop_time - start_time
            
        return performanceMetrics, ml_hat_evol, i_it_outer, conv_time
            
    def levelEstimation_superPos(self, y: np.ndarray):
        """
        Estimates levels where observations are weighted by current estimates 
        of S.
        
        Args:
            y (np.ndarray): Observations. 
                    .shape=(N,D)
        """
        
        # Get current (fixed) estimate of S
        ms_hat, _ = self.modelSelector.get_sHat()   # .shape=(N,M)
        
        # Calculate scaled messages (sigmas.shape=(N,M,D,D))
        Wl_b = \
            np.reshape(ms_hat, (self.N,self.M,1,1))**2 * \
            np.linalg.inv(self.modelSelector.sigmas) # .shape=(N,M,D,D)
        xil_b = np.reshape(
            Wl_b @ np.tile(y[:,np.newaxis,:,np.newaxis], (1,self.M,1,1)), 
            (self.N,self.M,self.D))   # .shape=(N,M,D)
        
        # Calculate MAP estimates of the levels accordingly
        self.Vl_hat = np.linalg.inv(self.Wl_prior + np.sum(Wl_b, axis=0))
        self.ml_hat = np.reshape(
            self.Vl_hat @ np.reshape(
                self.xil_prior + np.sum(xil_b, axis=0), (self.M,self.D,1)), 
            (self.M,self.D))
            
    def levelEstimation_selective(self, y: np.ndarray):
        """
        Estimates levels where, for model m, only those observations are 
        considered for which S_i is maximal at index m.
        
        Args:
            y (np.ndarray): Observations.
                    .shape=(N,D)
        """
        
        # Get current (fixed) estimate of S
        ms_hat, _ = self.modelSelector.get_sHat()   # .shape=(N,M)
        
        # Find model index of maximum element of S_i at each time index
        selectedModel = np.argmax(ms_hat, axis=1)
        
        # Count number of observations per model and number of models that are 
        # stil detected
        countObs = np.array([
            np.count_nonzero(selectedModel==m) for m in range(self.M)])
        countModels = np.count_nonzero(countObs)

        # Calculate 'selective' estimates of levels
        for m in range(self.M):
            if countObs[m] > 0:
                # Calculate dual mean messages for the corresponding 
                # observations
                xil_b_m = np.reshape(
                    self.modelSelector.sigmas[0,m] @ 
                    np.reshape(np.sum(y[selectedModel==m], axis=0), 
                               (self.D,1)), 
                    self.D)

                self.Vl_hat[m] = np.linalg.inv(
                    self.Wl_prior[m] + 
                    countObs[m]*np.linalg.inv(self.modelSelector.sigmas[0,m]))
                self.ml_hat[m] = np.reshape(
                    self.Vl_hat[m] @ 
                    np.reshape(self.xil_prior[m] + xil_b_m, (self.D,1)), 
                    self.D)
            else:
                self.Vl_hat[m] = np.linalg.inv(self.Wl_prior[m])
                self.ml_hat[m] = np.reshape(
                    self.Vl_hat[m] @ np.reshape(self.xil_prior[m], (self.D,1)), 
                    self.D)
                
        # Print warning if less than M models are detected
        if countModels != self.M:
            warnings.warn(
                f'Less than M models have been detected! Detected models ' + \
                f'= {countModels} out of {self.M}.')
             
    def get_output(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimated outputs, where at each time index i the 
        output of the model for which ms_hat[i] is maximal is returned.
        
        Returns:
            output (np.ndarray): Current estimated outputs.
                    .shape=(N,D)
        """
        
        # Determine chosen model at each time index
        ms_hat, _ = self.modelSelector.get_sHat()
        selectedModel = np.argmax(ms_hat, axis=1)
        
        # Construct corresponding output
        output = np.array([
            self.ml_hat[selectedModel_i] for selectedModel_i in selectedModel])
        
        return output