"""
Specifies a model selector that can be used to fit observations to different 
models.
"""

import numpy as np
from src.nuvPriors.nuvPriors_basic import boxCostPositivity
from src.nuvPriors.oneHot import oneHot
from src.models.pwcModel import PWCModel

class ModelSelector():
    """
    Model to select between M different models given their estimated outputs, 
    the corresponding observations, and the assumed observation noise 
    covariance per model. The result is a vector S of dimension M and legnth N 
    (for N observations), whose elements indicate at each time index i which 
    model is most likely to have generated the corresponding observation. 
    Furthermore, we assume sparse model changes, where the sparsity can be 
    tuned.
    """
    
    def __init__(
            self, N: int, M: int, sigmas: np.ndarray, 
            sigmas_type: str='covariance', 
            ms_init: np.ndarray=None, Vs_init: np.ndarray=None, 
            mu_init: np.ndarray=None, Vu_init: np.ndarray=None, 
            sh_squared: float=1e-6):
        """
        Args:
            N (int): Number of observations.
            M (int): Number of models.
            sigmas_type (str): States whether the noise is specified by 
                covariance or precision matrices. Must be in 
                ['covariance', 'precision']. Default is 'covariance'.
            sigmas (np.ndarray): Specifies observation noise by either the 
                covariance or precision matrices per model (optionally per 
                index). If only M matrices are given, the noise is assumed to 
                be constant for all models.
                    .shape=(M,D,D) or .shape=(N,M,D,D)
            ms_init (np.ndarray): Initial values of ms_hat. If None, ms_hat is 
                initialized to 1/M.
                    .shape=(N,M)
            Vs_init (np.ndarray): Initial values of Vs_hat. If None, Vs_hat is 
                initialized to identity matrices, scaled by 1e3 (i.e., very 
                uncertain about initialization).
                    .shape=(N,M,M)
            mu_init (np.ndarray): Initial values of mu_hat. If None, mu_hat is 
                initialized to zero.
                    .shape=(N-1,M)
            Vu_init (np.ndarray): Initial values of Vu_hat. If None, Vu_hat is 
                initialized to identity matrices, scaled by 1e3 (i.e., very 
                uncertain about initialization).
                    .shape=(N-1,M,M)
            sh_squared (float): Uncertainty / slack associated with condition 
                that sum over S must be one. Should be chosen small (i.e., 
                close to zero). Equal to zero should work too, but tends to
                be quite unstable.
        """
        
        assert sigmas_type in ['covariance', 'precision'], \
            f'sigmas_type = {sigmas_type} is unknwon!'
        
        self.N = N
        self.M = M
        self.sigmas_type = sigmas_type
        self.sh_squared = sh_squared
        
        if len(sigmas.shape)==3:
            sigmas = np.tile(sigmas[np.newaxis,:,:,:], (N,1,1,1))
        self.sigmas = sigmas
        
        if ms_init is None:
            ms_init = np.full((N,M), 1.0/M, dtype=float)
        self.estimator = PWCModel(
            N=N, D=M, mode='dual', mk_init=ms_init, Vk_init=Vs_init, 
            mu_init=mu_init, Vu_init=Vu_init)

    def get_sHat(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimates of S.
        
        Returns:
            ms_hat (np.ndarray): Current mean estimates of S.
                    .shape=(N,M)
            Vs_hat (np.ndarray): Current covariance matrix estimates of S.
                    .shape(N,M,M)
        """
        
        ms_hat = self.estimator.mk_hat
        Vs_hat = self.estimator.Vk_hat
        
        return ms_hat, Vs_hat
        
    def update_sigmas(self, sigmas_new: np.ndarray):
        """
        Update values of used sigmas.
        
        Args:
            sigmas_new (np.ndarray): New sigmas.
                .shape=(N,M,D,D)
        """
        
        assert self.sigmas.shape == sigmas_new.shape, \
            f'Dimension of sigmas_new must be {self.sigmas.shape}, but ' + \
            f'are sigmas_new.shape={sigmas_new.shape}!'
        
        self.sigmas = sigmas_new
        
    def estimate_selectedModel(
            self, y: np.ndarray, x_perModel: np.ndarray, n_it_irls: int=100, 
            beta_l: float=5.0, beta_h: float=5.0, beta_u: float=None, 
            priorType_oneHot: str='repulsive_logCost', diff_convTh: float=1e-3
            ) -> tuple[np.ndarray, int]:
        """
        Estimates S and U by IRLS with maximum n_it_irls iterations (or until 
        converged). The results are saved in S and U hat. Convergence is 
        checked by the absolute change of ms_hat from the current to the 
        previous iteration.
        
        Args: 
            y (np.ndarray): Observations. 
                    .shape=(N,D)
            x_perModel (np.ndarray): Outputs per model.
                    .shape=(N,M,D)
            n_it_irls (int): Maximum number of iterations. 
            beta_l (float): Tuning parameter for positivity constraint, must 
                be non-negative.
            beta_h (float): Tuning parameter for One-Hot constraint, must be 
                non-negative.
            beta_u (float): Tuning parameter for sparse input NUV, must be 
                non-negative.
            priorType_oneHot (str): Specifies the prior type used to emphasize 
                an all-{0,1} solution in the One-Hot constraint. Possible 
                cases are ['sparse', 'repulsive_logCost', 'repulsive_laplace', 
                'discrete']. 
            diff_convTh (float): Threshold for convergence.
            
        Returns:
            diffAZOSol (np.ndarray): Array containing averaged difference from 
                all-{0,1} solution.
            i_it (int): Index of last iteration in IRLS, starting at 0. 
                Therefore, the number of performed iterations is i_it + 1.
        """
    
        valid_priorType = \
            ['sparse', 'repulsive_logCost', 'repulsive_laplace', 'discrete']
        assert priorType_oneHot in valid_priorType, \
            f'priorType={priorType_oneHot} is unknown! Valid prior types ' + \
            f'are {valid_priorType}'
        
        diffAZOSol = np.empty(n_it_irls, dtype=float)
        
        # If no beta_u is given, set it to M (i.e., use plain NUV)
        if beta_u is None:
            beta_u = self.M
        
        # Calculate backward precision matrix messages through multiplication
        # node. These stay the same for all iterations of IRLS.
        Ws_b = self.multiplicationNode_b(y=y, x_perModel=x_perModel)
        
        for i_it in range(n_it_irls):
            
            # Get messages out of constraint nodes
            xil_b, Wl_b = self.positivity_b(beta=beta_l)
            xih_b, Wh_b = self.oneHot_b(
                beta=beta_h, priorType=priorType_oneHot)
            Wu_f = self.estimator.sparseInputs_f(beta_u=beta_u, inverse=True)
            
            # Calculate resulting 'ingoing' mean and covariance messages to 
            # the piecewise constant model
            xisp_b = xil_b + xih_b
            Wsp_b = Ws_b + Wl_b + Wh_b
            
            # Perform BIFM to estimate S and U
            _ = self.estimator.BIFM(xik_b=xisp_b, Wk_b=Wsp_b, Wu_f=Wu_f)
            
            # Calculate averaged difference from all-{0,1} solution
            ms_hat,_ = self.get_sHat()
            diffAZOSol[i_it] = np.mean(np.minimum(ms_hat, 1-ms_hat))
            
            # Check if IRLS has converged (i.e., if S is close enough to 
            # all-{0,1} solution)
            if diffAZOSol[i_it] < diff_convTh:
                break
        
        return diffAZOSol, i_it

    def multiplicationNode_b(
            self, y: np.ndarray, x_perModel: np.ndarray) -> np.ndarray:
        """
        Calculates backward precision matrix messages through the 
        multiplication node for fixed model parameters. These messages (per 
        dimension) are then used to constuct backward messages through S''' 
        (vector).

        Args:
            y (np.ndarray): Observations. 
                    .shape=(N,D).
            x_perModel (np.ndarray): Outputs per model.
                    .shape=(N,M,D)
        
        Returns:
            Wsppp_b (np.ndarray): Backward precision matrix messages through 
                multiplication node. 
                    .shape=(N,M,M)
        """
        
        # Get dimension of observations
        _,D = y.shape
    
        # Calculate differences between observations and model outputs
        z_perModel = \
            np.reshape(x_perModel, (self.N,self.M,D,1)) - \
            np.repeat(np.reshape(y, (self.N,1,D,1)), self.M, axis=1) 
            # .shape=(N,M,D,1)
    
        # Calculate / construct the backward precision matrices through S'''
        if self.sigmas_type=='covariance':
            Wsppp_b_vec = \
                np.transpose(z_perModel, (0,1,3,2)) @ \
                np.linalg.inv(self.sigmas) @ \
                z_perModel 
                    # .shape=(N,M,1,1)
        elif self.sigmas_type=='precision':
            Wsppp_b_vec = \
                np.transpose(z_perModel, (0,1,3,2)) @ \
                self.sigmas @ \
                z_perModel 
                    # .shape=(N,M,1,1)
        Wsppp_b = np.array([
            np.diag(Wsppp_b_vec_i[:,0,0]) 
            for Wsppp_b_vec_i in Wsppp_b_vec])
    
        return Wsppp_b
    
    def positivity_b(self, beta: float):
        """
        Calculates backward dual mean and precision matrix messages out of 
        positivity constraint.
        
        Args:
            beta: Tuning parameter for NUV update. Higher values 
            correspond to a more aggressive prior.
            
        Returns:
            xil_b (np.ndarray): Backward mean message out of positivity 
                constraint.
                    .shape=(N,M)
            Wl_b (np.ndarray): Backward precision matrix message out of 
                positivity constraint.
                    .shape=(N,M,M)
        """
        
        # Get current estimates of S
        ms_hat, _ = self.get_sHat()
        
        xil_b, Wl_b = boxCostPositivity(
            mx_hat=ms_hat, beta=beta, inverse=True)
        
        return xil_b, Wl_b
    
    def oneHot_b(
            self, beta: float, priorType: str='repulsive_logCost'
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates backward dual mean and precision matrix messages out of 
        One-Hot constraint.
        
        Args:
            beta (float): Tuning parameter for repulsive NUV prior. Its value 
                must be positive! Note that the discreteBase NUV can not be 
                tuned.
            priorType (str): Specifies type of repulsive NUV prior used. 
                Possible cases are ['sparse', 'repulsive_laplace', 
                'repulsive_logCost', 'discrete'].
                
        Returns:
            xih_b (np.ndarray): Backward mean message out of One-Hot 
                constraint.
                    .shape=(N,M)
            Wh_b (np.ndarray): Backward precision matrix message out of 
                One-Hot constraint.
                    .shape=(N,M,M)
        """
        
        # Get current estimates of S
        ms_hat, Vs_hat = self.get_sHat()
        
        xih_b, Wh_b = oneHot(
            mx_hat=ms_hat, Vx_hat=Vs_hat, beta=beta, 
            sh_squared=self.sh_squared, priorType=priorType, inverse=True)
        
        return xih_b, Wh_b