"""
Specifies a model to fit Piece-Wise Constant (PWC) data.
"""

import numpy as np
from src.nuvPriors.nuvPriors_basic import logCost

num_zero = 1e-100   # To account for some numerical instabilities

class PWCModel():
    """
    Model for fitting piecewise-constant (PWC) levels to N observations in 
    D-dimensions. A piecewise-constant solution is emphasized by sparsifying 
    NUV priors at each input. Each model has to be initialized with N, D, mode 
    (specifying type of message passing algorithm), and optionally initial 
    values for K and U hat. The acutal model output estimation is performed by 
    IRLS, where the ingoing messages into the model must be given (in the 
    representation corresponding to the selected mode).
    """
    
    def __init__(
            self, N: int, D: int, mode: str, mk_init: np.ndarray=None, 
            Vk_init: np.ndarray=None, mk_prior: np.ndarray=None, 
            Vk_prior: np.ndarray=None, mu_init: np.ndarray=None, 
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
            mk_init (np.ndarray): Initial values of mk_hat. If None, mk_hat is 
                initialized randomly, to zero.
                    .shape=(N,D)
            Vk_init (np.ndarray): Initial values of Vk_hat. If None, Vk_hat is 
                initialized to identity matrices.
                    .shape=(N,D,D)
            mk_prior (np.ndarray): Prior on mean of K_1. If None, mk_prior is 
                initialized to zero.
                    .shape=D
            Vk_prior (np.ndarray): Prior on covariance of K_1. If None, 
                Vk_prior is initialized to identity matrix, scaled by 1e3.
                    .shape=(D,D)
            mu_init (np.ndarray): Initial values of mu_hat. If None, mu_hat is 
                initialized to zero.
                    .shape=(N-1,D)
            Vu_init (np.ndarray): Initial values of Vu_hat. If None, Vu_hat is 
                initialized to identity matrices.
                    .shape=(N-1,D,D)
        """
        
        # Check dimensions of inputs
        valid_mode = ['conventional', 'dual']
        assert mode in valid_mode, \
            f'mode={mode} is unknown! Valid modes are {valid_mode}'
        assert mk_init is None or mk_init.shape == (N,D), \
            f'mk_init must be None or of .shape=(N,D)!'
        assert Vk_init is None or Vk_init.shape == (N,D,D), \
            f'Vk_init must be None or of .shape=(N,D,D)!'
        assert mk_prior is None or mk_prior.shape == D, \
            f'mk_prior must be None or of .shape=D!'
        assert Vk_prior is None or Vk_prior.shape == (D,D), \
            f'Vk_prior must be None or of .shape=(D,D)!'
        assert mu_init is None or mu_init.shape == (N-1,D), \
            f'mu_init must be None or of .shape=(N-1,D)!'
        assert Vu_init is None or Vu_init.shape == (N-1,D,D), \
            f'Vu_init must be None or of .shape=(N-1,D,D)!'
        
        # Initialize dimensions and mode
        self.N = N
        self.D = D
        self.mode = mode
        
        # Initialize K and U
        if mk_init is None:
            self.mk_hat = np.random.normal(0.0, 1e-3, (N,D))
        else:
            self.mk_hat = mk_init
        if Vk_init is None:
            self.Vk_hat = np.tile(np.identity(D, dtype=float), (N,1,1))
        else:
            self.Vk_hat = Vk_init
        if mk_prior is None:
            self.mk_prior = np.zeros(self.D, dtype=float)
        else:
            self.mk_prior = mk_prior
        if Vk_prior is None:
            self.Vk_prior = np.identity(D, dtype=float)*1e3
        else:
            self.Vk_prior = Vk_prior
        if mu_init is None:
            self.mu_hat = np.random.normal(0.0, 1e-3, (N-1,D))
        else:
            self.mu_hat = mu_init
        if Vu_init is None:
            self.Vu_hat = np.tile(np.identity(D, dtype=float), (N-1,1,1))
        else:
            self.Vu_hat = Vu_init

    def init_k(self, mk_init: np.ndarray, Vk_init: np.ndarray):
        """
        Re-initializes K.
        
        Args:
            mk_init (np.ndarray): Initial values of mk_hat.
                    .shape=(N,D)
            Vk_init (np.ndarray): Initial values of Vk_hat.
                    .shape=(N,D,D)
        """
        
        # Check dimensions of inputs
        assert mk_init.shape==(self.N,self.D), \
            f'mk_init must be of .shape=(N,D)!'
        assert Vk_init.shape==(self.N,self.D,self.D), \
            f'Vk_init must be of .shape=(N,D,D)!'
        
        # Update initialization of K and U
        self.mk_hat = mk_init
        self.Vk_hat = Vk_init
            
    def estimate_output(
            self, mxik_b: np.ndarray, VWk_b: np.ndarray, n_it_irls: int=1000, 
            beta_u: float=None, met_convTh: float=1e-4
            ) -> tuple[np.ndarray, int]:
        """
        Estimates K and U by IRLS with maximum n_it_irls iterations (or until 
        converged). The results are saved in K and U hat. Convergence is 
        checked by the absolute change of mk_hat from the current to the 
        previous iteration. Note that forward- / backward- message passing is 
        either done by MBF or BIFM, depending on the selected mode. 
        Accordingly, the given ingoing messages in mxik_b and VWk_b are either 
        interpreted as mean and variance or as dual mean and precision.
        
        Args:
            mxik_b (np.ndarray): Either interpreted as ingoing mean or dual 
                mean messages, depending on mode.
                    .shape=(N,D)
            VWk_b (np.ndarray): Either interpreted as ingoing covariance or 
                precision messages, depending on mode.
                    .shape=(N,D,D)
            n_it_irls (int): Maximum number of iterations for IRLS. Default 
                value is 1000.
            beta_u (float): Tuning parameter for sparsifying NUV. Higher 
                values correspond to a more aggressive prior. Must be gereater 
                than zero. If None (default), it will be chosen equal to D 
                (i.e., plain NUV).
            met_convTh (float): Threshold for convergence.
            
        Returns:
            changes (np.ndarray): Array containing relative changes per 
                iteration of IRLS.
            i_it (int): Index of last iteration in IRLS, starting at 0. 
                Therefore, the number of performed iterations is i_it + 1.
        """
        
        # Check dimensions of inputs
        assert mxik_b.shape == (self.N,self.D), \
            f'mxik_b must be of .shape=(N,D)!'
        assert VWk_b.shape == (self.N,self.D,self.D), \
            f'VWk_b must be of .shape=(N,D,D)!'
        
        if beta_u is None:
            beta_u = self.D
        assert beta_u > 0.0, \
            f'beta_u must be chosen greater than zero (or None)!'
        
        changes = np.empty(n_it_irls, dtype=float)
        
        # Perform IRLS
        for i_it in range(n_it_irls):
            
            # Check for message passing mode, calculate posterior estimates 
            # accordingly
            if self.mode == 'conventional':
                # Calculate forward covariance matrix message out of NUV
                Vu_f = self.sparseInputs_f(beta_u=beta_u, inverse=False)
                changes[i_it] = self.MBF(mk_b=mxik_b, Vk_b=VWk_b, Vu_f=Vu_f)
                
            elif self.mode == 'dual':
                # Calculate forward precision matrix message out of NUV
                Wu_f = self.sparseInputs_f(beta_u=beta_u, inverse=True)
                changes[i_it] = self.BIFM(xik_b=mxik_b, Wk_b=VWk_b, Wu_f=Wu_f)
            else:
                assert True, f'mode = {self.mode} is not known!'
            
            # Check if IRLS has converged (i.e., if change is below threshold)
            if changes[i_it] < met_convTh:
                break
        
        return changes, i_it
                        
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
            self, mk_b: np.ndarray, Vk_b: np.ndarray, Vu_f: np.ndarray
            ) -> np.ndarray:
        """
        Performs MBF to estimate K and U. Used if mode is set to 
        'conventional'.
        
        Args:
            mk_b (np.ndarray): Ingoing mean messages.
                    .shape=(N,D)
            Vk_b (np.ndarray): Ingoing covariance matrix messages.
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
        assert mk_b.shape == (self.N,self.D), \
            f'mk_b must be of .shape=(N,D)!'
        assert Vk_b.shape == (self.N,self.D,self.D), \
            f'Vk_b must be of .shape=(N,D,D)!'
        assert Vu_f.shape == (self.N-1,self.D,self.D), \
            f'Vu_f must be of .shape=(N-1,D,D)!'
    
        # Initialize forward messages
        mkp_f = np.empty((self.N,self.D), dtype=float)
        Vkp_f = np.empty((self.N,self.D,self.D), dtype=float)
        mkp_f[0] = self.mk_prior.copy()
        Vkp_f[0] = self.Vk_prior.copy()
        mkpp_f = np.empty((self.N-1,self.D), dtype=float)
        Vkpp_f = np.empty((self.N-1,self.D,self.D), dtype=float)
        G = np.empty((self.N-1,self.D,self.D), dtype=float)
        F = np.empty((self.N-1,self.D,self.D), dtype=float)

        # Do forwrad message passing
        for i in range(self.N-1):
            G[i] = np.linalg.inv(Vk_b[i] + Vkp_f[i])
            F[i] = np.identity(self.D, dtype=float) - Vkp_f[i]@G[i]

            mkpp_f[i] = mkp_f[i] + Vkp_f[i]@G[i]@(mk_b[i] - mkp_f[i])
            Vkpp_f[i] = Vkp_f[i] - Vkp_f[i]@G[i]@Vkp_f[i]

            mkp_f[i+1] = mkpp_f[i]
            Vkp_f[i+1] = Vkpp_f[i] + Vu_f[i]

        # Initialize tilde (backward) messages
        Wkp_t = np.empty((self.N,self.D,self.D), dtype=float)
        xikp_t = np.empty((self.N,self.D), dtype=float)
        Wkp_t[-1] = np.linalg.inv(Vkp_f[-1] + Vk_b[-1])
        xikp_t[-1] = Wkp_t[-1]@(mkp_f[-1] - mk_b[-1])

        # Do backward message passing
        for i in range(self.N-1,0,-1):
            xikp_t[i-1] = F[i-1].T@xikp_t[i] - G[i-1]@(mk_b[i-1] - mkp_f[i-1])
            Wkp_t[i-1] = F[i-1].T@Wkp_t[i]@F[i-1] + G[i-1]

        # Calculate posterior estimates
        mk_hat_new = \
            mkp_f - \
            np.reshape(Vkp_f@np.reshape(xikp_t, (self.N,-1,1)), (self.N,-1))
        self.Vk_hat = Vkp_f - Vkp_f@Wkp_t@Vkp_f

        self.mu_hat = np.reshape(
            -Vu_f@np.reshape(xikp_t[1:], (self.N-1,-1,1)), (self.N-1,-1))
        self.Vu_hat = Vu_f - Vu_f@Wkp_t[1:]@Vu_f
        
        # Calculate change
        change = np.mean(
            np.abs(mk_hat_new - self.mk_hat) / np.abs(self.mk_hat))
        self.mk_hat = mk_hat_new
        
        # Assert if any negative posterior variances have been estimated
        assert np.all(np.diagonal(self.Vk_hat, axis1=1, axis2=2) > -1e-10), \
            f'Detected negative variance in K, min = ' + \
            f'{np.min(np.diagonal(self.Vk_hat, axis1=1, axis2=2))}'
        assert np.all(np.diagonal(self.Vu_hat, axis1=1, axis2=2) > -1e-10), \
            f'Detected negative variance in U, min = ' + \
            f'{np.min(np.diagonal(self.Vu_hat, axis1=1, axis2=2))}'
    
        return change
    
    def BIFM(
            self, xik_b: np.ndarray, Wk_b: np.ndarray, Wu_f: np.ndarray
            ) -> np.ndarray:
        """
        Performs BIFM to estimate K and U. Used if mode is set to 'dual'.
        
        Args:
            xik_b (np.ndarray): Ingoing dual mean messages.
                    .shape=(N,D)
            Wk_b (np.ndarray): Ingoing precision matrix messages.
                    .shape=(N,D,D)
            Wu_f (np.ndarray): Precision matrix messages out of sparsity 
                nodes.
                    .shape=(N-1,D,D)
                    
        Returns:
            change (float): Averaged difference between the new mean 
                estimation of K and the previous one, relative to the absolute 
                mean of the previous estimation.
        """
        
        # Check dimensions of inputs
        assert xik_b.shape == (self.N,self.D), \
            f'xik_b must be of .shape=(N,D)!'
        assert Wk_b.shape == (self.N,self.D,self.D), \
            f'Wk_b must be of .shape=(N,D,D)!'
        assert Wu_f.shape == (self.N-1,self.D,self.D), \
            f'Wu_f must be of .shape=(N-1,D,D)!'
    
        # Initialize forward messages
        xikp_b = np.empty((self.N,self.D), dtype=float)
        Wkp_b = np.empty((self.N,self.D,self.D), dtype=float)
        xikpp_b = np.empty((self.N-1,self.D), dtype=float)
        Wkpp_b = np.empty((self.N-1,self.D,self.D), dtype=float)
        Hdd = np.empty((self.N,self.D,self.D), dtype=float)   
            # Hdd[0] should never be used!
        hdd = np.empty((self.N,self.D), dtype=float)   
            # hdd[0] should never be used!

        xikp_b[-1] = xik_b[-1]
        Wkp_b[-1] = Wk_b[-1]
    
        # Do backward message passing
        for i in range(self.N-1,0,-1):
            Hdd[i] = np.linalg.inv(Wu_f[i-1] + Wkp_b[i])
            hdd[i] = Hdd[i]@xikp_b[i]
        
            xikpp_b[i-1] = xikp_b[i] - Wkp_b[i]@hdd[i]
            Wkpp_b[i-1] = Wkp_b[i] - Wkp_b[i]@Hdd[i]@Wkp_b[i]
        
            xikp_b[i-1] = xikpp_b[i-1] + xik_b[i-1]
            Wkp_b[i-1] = Wkpp_b[i-1] + Wk_b[i-1]
        
        # Initialize tilde and hat messages for forward message passing
        mk_hat_new = np.empty((self.N,self.D), dtype=float)
        Vk_hat_new = np.empty((self.N,self.D,self.D), dtype=float)
        xiu_t = np.empty((self.N-1,self.D), dtype=float)
        Wu_t = np.empty((self.N-1,self.D,self.D), dtype=float)
        F_t = np.empty((self.N,self.D,self.D), dtype=float)
            # F_t[0] should never be used!
        
        Wkp_f_1 = np.linalg.inv(self.Vk_prior)
        xikp_f_1 = Wkp_f_1 @ self.mk_prior
        Vk_hat_new[0] = np.linalg.inv(Wkp_f_1 + Wkp_b[0])
        mk_hat_new[0] = Vk_hat_new[0]@(xikp_f_1 + xikp_b[0])
    
        for i in range(1,self.N):
            F_t[i] = np.identity(self.D, dtype=float) - Wkp_b[i]@Hdd[i]
        
            mk_hat_new[i] = F_t[i].T@mk_hat_new[i-1] + hdd[i]
            Vk_hat_new[i] = F_t[i].T@Vk_hat_new[i-1]@F_t[i] + Hdd[i]
        
            xiu_t[i-1] = Wkp_b[i]@mk_hat_new[i] - xikp_b[i]
            Wu_t[i-1] = Wkp_b[i] - Wkp_b[i]@Vk_hat_new[i]@Wkp_b[i]
        
        # Calculate change
        change = np.mean(
            np.abs(mk_hat_new - self.mk_hat) / np.abs(self.mk_hat))
        self.mk_hat = mk_hat_new
        self.Vk_hat = Vk_hat_new
        
        # Calculate posterior estimates of U (those of K have already been 
        # calculated)
        num_zeroMat = np.tile(
            np.identity(self.D, dtype=float)*num_zero, (self.N-1,1,1))
        Vu_f = np.linalg.inv(Wu_f + num_zeroMat)
        self.mu_hat = -np.reshape(
            Vu_f@np.reshape(xiu_t, (self.N-1,self.D,1)), (self.N-1,self.D))
        self.Vu_hat = Vu_f - Vu_f@Wu_t@Vu_f
    
        return change