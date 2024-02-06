"""
Specifies a model to fit Piece-Wise Constant (PWC) data.
"""

import numpy as np
import time
from tqdm import trange
from src.nuvPriors.nuvPriors_basic import logCost

num_zero = 1e-100   # To account for some numerical instabilities

class PWCModel():
    """
    Model for fitting piecewise-constant (PWC) levels to N observations in 
    D-dimensions. A piecewise-constant solution is emphasized by sparsifying 
    NUV priors at each input. Each model has to be initialized with N, D, mode 
    (specifying type of message passing algorithm), and optionally initial 
    values for X and U hat. The acutal model output estimation is performed by 
    IRLS, where the ingoing messages into the model must be given (in the 
    representation corresponding to the selected mode).
    """
    
    def __init__(
            self, N: int, D: int, mode: str, mx_init: np.ndarray=None, 
            Vx_init: np.ndarray=None, mx_prior: np.ndarray=None, 
            Vx_prior: np.ndarray=None, mu_init: np.ndarray=None, 
            Vu_init: np.ndarray=None):
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
                initialized to identity matrices.
                    .shape=(N,D,D)
            mx_prior (np.ndarray): Prior on mean of X'_1. If None, mx_prior is 
                initialized to zero.
                    .shape=D
            Vx_prior (np.ndarray): Prior on covariance of X'_1. If None, 
                Vx_prior is initialized to identity matrix, scaled by 1e3.
                    .shape=(D,D)
            mu_init (np.ndarray): Initial values of mu_hat. If None, mu_hat is 
                initialized randomly, close to zero.
                    .shape=(N-1,D)
            Vu_init (np.ndarray): Initial values of Vu_hat. If None, Vu_hat is 
                initialized to identity matrices.
                    .shape=(N-1,D,D)
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
        assert mu_init is None or mu_init.shape==(N-1,D), \
            f'mu_init must be None or of .shape=(N-1,D)!'
        assert Vu_init is None or Vu_init.shape==(N-1,D,D), \
            f'Vu_init must be None or of .shape=(N-1,D,D)!'
        
        # Check if selected mode is known
        valid_mode = ['conventional', 'dual']
        assert mode in valid_mode, \
            f'mode={mode} is unknown! Valid modes are {valid_mode}'
        
        # Initialize dimensions and mode
        self.N = N
        self.D = D
        self.mode = mode
        
        # Initialize X and U
        if mx_init is None:
            self.mx_hat = np.random.normal(0.0, 1e-3, (N,D))
        else:
            self.mx_hat = mx_init
        if Vx_init is None:
            self.Vx_hat = np.tile(np.identity(D, dtype=float), (N,1,1))
        else:
            self.Vx_hat = Vx_init
        if mx_prior is None:
            self.mx_prior = np.zeros(self.D, dtype=float)
        else:
            self.mx_prior = mx_prior
        if Vx_prior is None:
            self.Vx_prior = np.identity(D, dtype=float)*1e3
        else:
            self.Vx_prior = Vx_prior
        if mu_init is None:
            self.mu_hat = np.random.normal(0.0, 1e-3, (N-1,D))
        else:
            self.mu_hat = mu_init
        if Vu_init is None:
            self.Vu_hat = np.tile(np.identity(D, dtype=float), (N-1,1,1))
        else:
            self.Vu_hat = Vu_init

    def init_x(self, mx_init: np.ndarray, Vx_init: np.ndarray):
        """
        Re-initializes K.
        
        Args:
            mx_init (np.ndarray): Initial values of mx_hat.
                    .shape=(N,D)
            Vx_init (np.ndarray): Initial values of Vx_hat.
                    .shape=(N,D,D)
        """
        
        # Check dimensions of inputs
        assert mx_init.shape==(self.N,self.D), \
            f'mx_init must be of .shape=(N,D)!'
        assert Vx_init.shape==(self.N,self.D,self.D), \
            f'Vx_init must be of .shape=(N,D,D)!'
        
        # Update initialization of X and U
        self.mx_hat = mx_init
        self.Vx_hat = Vx_init
            
    def estimate_output(
            self, mxix_b: np.ndarray, VWx_b: np.ndarray, n_it_irls: int=1000, 
            beta_u: float=None, met_convTh: float=1e-4, 
            disable_progressBar: bool=False) -> tuple[np.ndarray, int, float]:
        """
        Estimates X and U by IRLS with maximum n_it_irls iterations (or until 
        converged). The results are saved in X and U hat. Convergence is 
        checked by the absolute change of mx_hat from the current to the 
        previous iteration. Note that forward- / backward- message passing is 
        either done by MBF or BIFM, depending on the selected mode. 
        Accordingly, the given ingoing messages in mxix_b and VWx_b are either 
        interpreted as mean and variance or as dual mean and precision.
        
        Args:
            mxix_b (np.ndarray): Either interpreted as ingoing mean or dual 
                mean messages, depending on mode.
                    .shape=(N,D)
            VWx_b (np.ndarray): Either interpreted as ingoing covariance or 
                precision messages, depending on mode.
                    .shape=(N,D,D)
            n_it_irls (int): Maximum number of iterations for IRLS. Default 
                value is 1000.
            beta_u (float): Tuning parameter for sparsifying NUV. Higher 
                values correspond to a more aggressive prior. Must be gereater 
                than zero. If None (default), it will be chosen equal to D 
                (i.e., plain NUV).
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
        assert mxix_b.shape==(self.N,self.D), \
            f'mxix_b must be of .shape=(N,D)!'
        assert VWx_b.shape==(self.N,self.D,self.D), \
            f'VWx_b must be of .shape=(N,D,D)!'
        
        # Start timer
        start_time = time.time()
        
        if beta_u is None:
            beta_u = self.D
        assert beta_u > 0.0, \
            f'beta_u must be chosen greater than zero (or None)!'
        
        changes = np.empty(n_it_irls, dtype=float)
        
        # Perform IRLS
        for i_it in trange(n_it_irls, disable=disable_progressBar):
            
            # Check for message passing mode, calculate posterior estimates 
            # accordingly
            if self.mode == 'conventional':
                # Calculate forward covariance matrix message out of NUV
                Vu_f = self.sparseInputs_f(beta_u=beta_u, inverse=False)
                changes[i_it] = self.MBF(mx_b=mxix_b, Vx_b=VWx_b, Vu_f=Vu_f)
                
            elif self.mode == 'dual':
                # Calculate forward precision matrix message out of NUV
                Wu_f = self.sparseInputs_f(beta_u=beta_u, inverse=True)
                changes[i_it] = self.BIFM(xix_b=mxix_b, Wx_b=VWx_b, Wu_f=Wu_f)
            else:
                assert True, f'mode = {self.mode} is not known!'
            
            # Check if IRLS has converged (i.e., if change is below threshold)
            if changes[i_it] < met_convTh:
                break
            
        # Stop timer and calculate time needed for convergence
        stop_time = time.time()
        conv_time = stop_time - start_time
        
        return changes, i_it, conv_time
                        
    def sparseInputs_f(self, beta_u: float, inverse: bool=False) -> np.ndarray:
        """
        Calculates outgoing messages out of sparsifying log-cost NUV. 
        
        Args: 
            beta_u (float): Tuning parameter for sparsifying NUV. Higher 
                values correspond to a more aggressive prior.
            inverse (bool): If False, the representation of the outgoing 
                messages is in terms of their covariance matrices. If True, 
                their inverses (i.e., the precision matrices) are given. 
                Default is False.
                
        Returns:
            VWu_f (np.ndarray): Outgoing messages either by their covariance 
                or precision matrix representation.
                    .shape=(N-1,D,D)
        """
        
        # Calculate (scalar) EM update for log-cost NUV
        VWu_f = logCost(
            mx_hat=self.mu_hat, Vx_hat=self.Vu_hat, beta=beta_u, 
            inverse=inverse)
        
        return VWu_f
        
    def MBF(
            self, mx_b: np.ndarray, Vx_b: np.ndarray, Vu_f: np.ndarray
            ) -> np.ndarray:
        """
        Performs MBF to estimate X and U. Used if mode is set to 
        'conventional'.
        
        Args:
            mx_b (np.ndarray): Ingoing mean messages.
                    .shape=(N,D)
            Vx_b (np.ndarray): Ingoing covariance matrix messages.
                    .shape=(N,D,D)
            Vu_f (np.ndarray): Covariance matrix messages out of sparsity 
                nodes.
                    .shape=(N-1,D,D)
                    
        Returns:
            change (float): Averaged difference between the new mean 
                estimation of K and the previous one, relative to the absolute 
                mean of the previous estimation.
        """
        
        # Check dimensions of inputs
        assert mx_b.shape==(self.N,self.D), \
            f'mx_b must be of .shape=(N,D)!'
        assert Vx_b.shape==(self.N,self.D,self.D), \
            f'Vx_b must be of .shape=(N,D,D)!'
        assert Vu_f.shape==(self.N-1,self.D,self.D), \
            f'Vu_f must be of .shape=(N-1,D,D)!'
    
        # Initialize forward messages
        mxp_f = np.empty((self.N,self.D), dtype=float)
        Vxp_f = np.empty((self.N,self.D,self.D), dtype=float)
        mxp_f[0] = self.mx_prior.copy()
        Vxp_f[0] = self.Vx_prior.copy()
        mxpp_f = np.empty((self.N-1,self.D), dtype=float)
        Vxpp_f = np.empty((self.N-1,self.D,self.D), dtype=float)
        G = np.empty((self.N-1,self.D,self.D), dtype=float)
        F = np.empty((self.N-1,self.D,self.D), dtype=float)

        # Do forwrad message passing
        for i in range(self.N-1):
            G[i] = np.linalg.inv(Vx_b[i] + Vxp_f[i])
            F[i] = np.identity(self.D, dtype=float) - Vxp_f[i]@G[i]

            mxpp_f[i] = mxp_f[i] + Vxp_f[i]@G[i]@(mx_b[i] - mxp_f[i])
            Vxpp_f[i] = Vxp_f[i] - Vxp_f[i]@G[i]@Vxp_f[i]

            mxp_f[i+1] = mxpp_f[i]
            Vxp_f[i+1] = Vxpp_f[i] + Vu_f[i]

        # Initialize tilde (backward) messages
        Wxp_t = np.empty((self.N,self.D,self.D), dtype=float)
        xixp_t = np.empty((self.N,self.D), dtype=float)
        Wxp_t[-1] = np.linalg.inv(Vxp_f[-1] + Vx_b[-1])
        xixp_t[-1] = Wxp_t[-1]@(mxp_f[-1] - mx_b[-1])

        # Do backward message passing
        for i in range(self.N-1,0,-1):
            xixp_t[i-1] = F[i-1].T@xixp_t[i] - G[i-1]@(mx_b[i-1] - mxp_f[i-1])
            Wxp_t[i-1] = F[i-1].T@Wxp_t[i]@F[i-1] + G[i-1]

        # Calculate posterior estimates
        mx_hat_new = \
            mxp_f - \
            np.reshape(Vxp_f@np.reshape(xixp_t, (self.N,-1,1)), (self.N,-1))
        self.Vx_hat = Vxp_f - Vxp_f@Wxp_t@Vxp_f

        self.mu_hat = np.reshape(
            -Vu_f@np.reshape(xixp_t[1:], (self.N-1,-1,1)), (self.N-1,-1))
        self.Vu_hat = Vu_f - Vu_f@Wxp_t[1:]@Vu_f
        
        # Calculate change
        change = np.mean(
            np.abs(mx_hat_new - self.mx_hat) / np.abs(self.mx_hat))
        self.mx_hat = mx_hat_new
        
        # Assert if any negative posterior variances have been estimated
        assert np.all(np.diagonal(self.Vx_hat, axis1=1, axis2=2) > -1e-10), \
            f'Detected negative variance in X, min = ' + \
            f'{np.min(np.diagonal(self.Vx_hat, axis1=1, axis2=2))}'
        assert np.all(np.diagonal(self.Vu_hat, axis1=1, axis2=2) > -1e-10), \
            f'Detected negative variance in U, min = ' + \
            f'{np.min(np.diagonal(self.Vu_hat, axis1=1, axis2=2))}'
    
        return change
    
    def BIFM(
            self, xix_b: np.ndarray, Wx_b: np.ndarray, Wu_f: np.ndarray
            ) -> np.ndarray:
        """
        Performs BIFM to estimate K and U. Used if mode is set to 'dual'.
        
        Args:
            xix_b (np.ndarray): Ingoing dual mean messages.
                    .shape=(N,D)
            Wx_b (np.ndarray): Ingoing precision matrix messages.
                    .shape=(N,D,D)
            Wu_f (np.ndarray): Precision matrix messages out of sparsity 
                nodes.
                    .shape=(N-1,D,D)
                    
        Returns:
            change (float): Averaged difference between the new mean 
                estimation of X and the previous one, relative to the absolute 
                mean of the previous estimation.
        """
        
        # Check dimensions of inputs
        assert xix_b.shape==(self.N,self.D), \
            f'xix_b must be of .shape=(N,D)!'
        assert Wx_b.shape==(self.N,self.D,self.D), \
            f'Wx_b must be of .shape=(N,D,D)!'
        assert Wu_f.shape==(self.N-1,self.D,self.D), \
            f'Wu_f must be of .shape=(N-1,D,D)!'
    
        # Initialize forward messages
        xixp_b = np.empty((self.N,self.D), dtype=float)
        Wxp_b = np.empty((self.N,self.D,self.D), dtype=float)
        xixpp_b = np.empty((self.N-1,self.D), dtype=float)
        Wxpp_b = np.empty((self.N-1,self.D,self.D), dtype=float)
        Hdd = np.empty((self.N,self.D,self.D), dtype=float)   
            # Hdd[0] should never be used!
        hdd = np.empty((self.N,self.D), dtype=float)   
            # hdd[0] should never be used!

        xixp_b[-1] = xix_b[-1]
        Wxp_b[-1] = Wx_b[-1]
    
        # Do backward message passing
        for i in range(self.N-1,0,-1):
            Hdd[i] = np.linalg.inv(Wu_f[i-1] + Wxp_b[i])
            hdd[i] = Hdd[i]@xixp_b[i]
        
            xixpp_b[i-1] = xixp_b[i] - Wxp_b[i]@hdd[i]
            Wxpp_b[i-1] = Wxp_b[i] - Wxp_b[i]@Hdd[i]@Wxp_b[i]
        
            xixp_b[i-1] = xixpp_b[i-1] + xix_b[i-1]
            Wxp_b[i-1] = Wxpp_b[i-1] + Wx_b[i-1]
        
        # Initialize tilde and hat messages for forward message passing
        mx_hat_new = np.empty((self.N,self.D), dtype=float)
        Vx_hat_new = np.empty((self.N,self.D,self.D), dtype=float)
        xiu_t = np.empty((self.N-1,self.D), dtype=float)
        Wu_t = np.empty((self.N-1,self.D,self.D), dtype=float)
        F_t = np.empty((self.N,self.D,self.D), dtype=float)
            # F_t[0] should never be used!
        
        Wxp_f_1 = np.linalg.inv(self.Vx_prior)
        xixp_f_1 = Wxp_f_1 @ self.mx_prior
        Vx_hat_new[0] = np.linalg.inv(Wxp_f_1 + Wxp_b[0])
        mx_hat_new[0] = Vx_hat_new[0]@(xixp_f_1 + xixp_b[0])
    
        for i in range(1,self.N):
            F_t[i] = np.identity(self.D, dtype=float) - Wxp_b[i]@Hdd[i]
        
            mx_hat_new[i] = F_t[i].T@mx_hat_new[i-1] + hdd[i]
            Vx_hat_new[i] = F_t[i].T@Vx_hat_new[i-1]@F_t[i] + Hdd[i]
        
            xiu_t[i-1] = Wxp_b[i]@mx_hat_new[i] - xixp_b[i]
            Wu_t[i-1] = Wxp_b[i] - Wxp_b[i]@Vx_hat_new[i]@Wxp_b[i]
        
        # Calculate change
        change = np.mean(
            np.abs(mx_hat_new - self.mx_hat) / np.abs(self.mx_hat))
        self.mx_hat = mx_hat_new
        self.Vx_hat = Vx_hat_new
        
        # Calculate posterior estimates of U (those of X have already been 
        # calculated)
        num_zeroMat = np.tile(
            np.identity(self.D, dtype=float)*num_zero, (self.N-1,1,1))
        Vu_f = np.linalg.inv(Wu_f + num_zeroMat)
        self.mu_hat = -np.reshape(
            Vu_f@np.reshape(xiu_t, (self.N-1,self.D,1)), (self.N-1,self.D))
        self.Vu_hat = Vu_f - Vu_f@Wu_t@Vu_f
    
        return change