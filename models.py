# All different models defined to be used: 

import torch as tr
from entmax import sparsemax 
import numpy as np
import numpy.random as rn
from scipy import optimize

class model_pi_sure_no_grid_modeling(tr.nn.Module):
    """
    Optimize pi, theta grid fixed

    Typically used without training, just initialized at the NPMLE solution
    The default is for this model to run on CPU
    """

    def __init__(self, Z, B, init_val, device="cpu",
                 quantile_IQR = 0.95):

        super(model_pi_sure_no_grid_modeling, self).__init__()
        # real hyperparams 
        self.real_params = tr.nn.Parameter((tr.ones(B) * init_val).to(device), requires_grad=True)
        self.B = B

        self.min_Z = min(Z).to(device)
        self.max_Z = max(Z).to(device)

        # no grid modeling because only training pi, not theta
        self.device = device
        
    def forward(self):
        """Forward pass through the network. 
        Input:
            NONE
            
        Output: 
            a (B x 1) tensor of pi, the probability vector, 
                B: number of theta values in grid
        """
        return tr.nn.Softmax(dim=0)(self.real_params)
    
    def get_theta_grid_and_pi(self, n, B):
        pi_param = self.forward()
        pi_param = pi_param[None, :]
        pi_nB = pi_param.expand(n, B)

        theta_diff = (1/(B-1))*tr.ones(B-1).to(self.device)
        theta_cum = tr.cumsum( tr.concat((tr.zeros(1).to(self.device), theta_diff), dim=0), dim = 0)
        theta_grid = theta_cum*(self.max_Z - self.min_Z) + self.min_Z
        theta_grid = theta_grid[None, :]
        theta_nB = theta_grid.expand(n, B)

        return(theta_nB, pi_nB)

    def opt_func(self, Z, n, B, sigma): 
        """Takes a matrix of square of (z-theta) and features, returns a scalar to be optimized.
        """ # SURE of t(z) = z + sigma * s(z; w)

        assert len(sigma.shape) == 1
        
        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)
        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        # print(f"Z.device: {Z.device}")
        # print(f"Z_nb.device : {Z_nb.device}")
        # print(f"theta_nB.device : {theta_nB.device}")
        # print(f"pi_nB.device : {pi_nB.device}")
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2) 
        
        numerator = ((pi_nB*Z_theta_sq*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)) 
        denominator = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)
        ratio = 2*numerator/denominator

        score_squared = (self.compute_score(Z, n, B, sigma))**2 
        score_term = score_squared*(sigma**4) 

        sigma_squared = (sigma**2) 

        return (ratio.sum() - score_term.sum() - sigma_squared.sum())/n 
    
    def compute_score(self, Z, n, B, sigma, verbose=False):
        """
        sigma.shape = n,
        """
        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)
        
        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2) 
        
        numerator = ((pi_nB*(-1*Z_theta)*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1))/(sigma**2) 
        denominator = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)

        if verbose:
            print(f"Z_theta_by_sigma_sq.shape : {Z_theta_by_sigma_sq.shape}")
            print(f"numerator.shape : {numerator.shape}")
            print(f"denominator.shape : {denominator.shape}")


        return (numerator/denominator)
    
    def get_prior(self, Z):

        Z = Z.detach().numpy()
        pi_param = self.forward()

        theta_diff = np.concatenate([[0], 1/(self.B-1) * np.ones(self.B-1)])
        # standardize and scale
        theta_cum = np.cumsum(theta_diff, axis=0)
        theta_grid = theta_cum*(self.max_Z.cpu().item() - self.min_Z.cpu().item()) + self.min_Z.cpu().item()

        return theta_grid.reshape(self.B,), pi_param.to('cpu')

    def get_theta_hat(self, n, B, Z_grid, sigma):
        """
        Return n-length np.array of shrinkage rules for n values of Z

        sigma is either a float or (n,)
        """

        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = sigma*tr.ones(n,)
            variance = sigma**2
        else:
            variance = sigma**2
        
        score = self.compute_score(Z_grid, n, B, sigma)

        theta_hat = Z_grid + variance * score

        return(theta_hat.cpu().detach().numpy())

    def get_marginal(self, n, B, Z_grid, sigma):
        """
        * sigma: float
        """

        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)

        Z_nb = Z_grid[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma**2)

        normal_pdf = tr.exp(-Z_theta_by_sigma_sq/2) / (np.sqrt(2*np.pi) * sigma )

        conditional_marginal = (pi_nB * normal_pdf).sum(axis = 1) # conditional marginal

        return(conditional_marginal.detach().numpy())

class model_pi_sure(tr.nn.Module):
    """
    Optimize pi, theta grid fixed

    Typically used without training, just initialized at the NPMLE solution
    The default is for this model to run on CPU
    """

    def __init__(self, Z, B, init_val, device="cpu",
                 quantile_IQR = 0.95):

        super(model_pi_sure, self).__init__()
        # real hyperparams 
        self.real_params = tr.nn.Parameter((tr.ones(B) * init_val).to(device), requires_grad=True)
        self.B = B

        self.min_Z = min(Z).to(device)
        self.max_Z = max(Z).to(device)

        self.median_Z = tr.median(Z).to(device)
        self.lower_quantile_Z = tr.quantile(Z, 0.5-quantile_IQR/2).to(device)
        self.higher_quantile_Z = tr.quantile(Z, 0.5+quantile_IQR/2).to(device)
        self.log_IQR_Z = tr.log(self.higher_quantile_Z - self.lower_quantile_Z ).to(device)

        self.location=tr.nn.Parameter(tr.zeros(1).to(device)*self.median_Z, requires_grad=False)
        self.log_scale=tr.nn.Parameter(tr.ones(1).to(device)*self.log_IQR_Z, requires_grad=False)

        # no grid modeling because only training pi, not theta
        self.device = device
        
    def forward(self):
        """Forward pass through the network. 
        Input:
            NONE
            
        Output: 
            a (B x 1) tensor of pi, the probability vector, 
                B: number of theta values in grid
        """
        return tr.nn.Softmax(dim=0)(self.real_params)
    
    def get_theta_grid_and_pi(self, n, B):
        pi_param = self.forward()
        pi_param = pi_param[None, :]
        pi_nB = pi_param.expand(n, B)

        theta_diff = (1/(B-1))*tr.ones(B-1).to(self.device)
        theta_cum = (tr.cumsum( tr.concat((tr.zeros(1).to(self.device),
                      theta_diff), dim=0), dim = 0) 
                      - 0.5) * tr.exp(self.log_scale)
        theta_cum = theta_cum[None, :]
        theta_nB = theta_cum.expand(n, B) + self.location 

        return(theta_nB, pi_nB)

    def opt_func(self, Z, n, B, sigma): 
        """Takes a matrix of square of (z-theta) and features, returns a scalar to be optimized.
        """ # SURE of t(z) = z + sigma * s(z; w)

        assert len(sigma.shape) == 1
        
        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)
        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        # print(f"Z.device: {Z.device}")
        # print(f"Z_nb.device : {Z_nb.device}")
        # print(f"theta_nB.device : {theta_nB.device}")
        # print(f"pi_nB.device : {pi_nB.device}")
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2) 
        
        numerator = ((pi_nB*Z_theta_sq*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)) 
        denominator = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)
        ratio = 2*numerator/denominator

        score_squared = (self.compute_score(Z, n, B, sigma))**2 
        score_term = score_squared*(sigma**4) 

        sigma_squared = (sigma**2) 

        return (ratio.sum() - score_term.sum() - sigma_squared.sum())/n 
    
    def compute_score(self, Z, n, B, sigma, verbose=False):
        """
        sigma.shape = n,
        """
        assert len(sigma.shape) == 1

        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)
        
        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2) 
        
        numerator = ((pi_nB*(-1*Z_theta)*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1))/(sigma**2) 
        denominator = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)

        if verbose:
            print(f"Z_theta_by_sigma_sq.shape : {Z_theta_by_sigma_sq.shape}")
            print(f"numerator.shape : {numerator.shape}")
            print(f"denominator.shape : {denominator.shape}")


        return (numerator/denominator)
    
    def get_prior(self, Z):

        Z = Z.detach().numpy()
        pi_param = self.forward()

        theta_diff = np.concatenate([[0], 1/(self.B-1) * np.ones(self.B-1)])
        # standardize and scale
        theta_grid = (np.cumsum(theta_diff) - 0.5)*(tr.exp(self.log_scale).cpu().item()) + self.location.cpu().item()

        return theta_grid.reshape(self.B,), pi_param.to('cpu')

    def get_theta_hat(self, n, B, Z_grid, sigma):
        """
        Return n-length np.array of shrinkage rules for n values of Z

        sigma is either a float or (n,)
        """

        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = sigma*tr.ones(n,)
            variance = sigma**2
        else:
            variance = sigma**2
        
        score = self.compute_score(Z_grid, n, B, sigma)

        theta_hat = Z_grid + variance * score

        return(theta_hat.cpu().detach().numpy())

    def get_marginal(self, n, B, Z_grid, sigma):
        """
        * sigma: float
        """

        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)

        Z_nb = Z_grid[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma**2)

        normal_pdf = tr.exp(-Z_theta_by_sigma_sq/2) / (np.sqrt(2*np.pi) * sigma )

        conditional_marginal = (pi_nB * normal_pdf).sum(axis = 1) # conditional marginal

        return(conditional_marginal.detach().numpy())



class model_theta_sure(tr.nn.Module):
    """
    Optimize theta grid, pi fixed = 1/B
    """

    def __init__(self, Z, B, init_val=tr.log(tr.Tensor([1.5])), device="cpu", use_location=False, use_scale=True,
                 quantile_IQR = 0.95):

        super(model_theta_sure, self).__init__()

        # real hyperparams 
        self.theta_real_vals = tr.nn.Parameter((tr.ones(B-1) * init_val).to(device), requires_grad=True)

        self.B = B
        self.min_Z = min(Z).to(device)
        self.max_Z = max(Z).to(device)

        self.median_Z = tr.median(Z).to(device)
        self.lower_quantile_Z = tr.quantile(Z, 0.5-quantile_IQR/2).to(device)
        self.higher_quantile_Z = tr.quantile(Z, 0.5+quantile_IQR/2).to(device)
        self.log_IQR_Z = tr.log(self.higher_quantile_Z - self.lower_quantile_Z ).to(device)

        self.use_location=use_location
        self.use_scale=use_scale

        self.location=tr.nn.Parameter(tr.zeros(1).to(device)*self.median_Z, requires_grad=use_location)
        self.log_scale=tr.nn.Parameter(tr.ones(1).to(device)*self.log_IQR_Z, requires_grad=use_scale)

        self.device = device
        
    def forward(self):
        """Forward pass through the network. 
        Input:
            NONE
            
        Output: 
            a (B x 1) tensor of pi, the probability vector, 
                B: number of theta values in grid
        """
        return tr.nn.Softmax(dim=0)(self.theta_real_vals) 
    
    def get_theta_grid_and_pi(self, n, B):

        theta_diff = self.forward() 
        theta_cum = (tr.cumsum( tr.concat((tr.zeros(1).to(self.device),
                            theta_diff), dim=0), dim = 0) 
                            - 0.5) * tr.exp(self.log_scale)
        theta_cum = theta_cum[None, :]
        theta_nB = theta_cum.expand(n, B) + self.location

        pi_param = (1/B)*tr.ones(B).to(self.device)
        pi_param = pi_param[None, :]
        pi_nB = pi_param.expand(n, B)

        return(theta_nB, pi_nB)
    
    def opt_func(self, Z, n, B, sigma): 
        """Takes a matrix of square of (z-theta) and features, returns a scalar to be optimized.
        """ # SURE of t(z) = z + sigma * s(z; w)

        assert len(sigma.shape) == 1
        
        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)

        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2)
    
        numerator = ((pi_nB*Z_theta_sq*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)) 
        denominator = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)
        ratio = 2*numerator/denominator

        score_squared = (self.compute_score(Z, n, B, sigma))**2 
        score_term = score_squared*(sigma**4) 

        sigma_squared = (sigma**2) 

        return (ratio.sum() - score_term.sum() - sigma_squared.sum())/n 
    
    def compute_score(self, Z, n, B, sigma):

        assert len(sigma.shape) == 1

        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)

        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2)

        numerator = ((pi_nB*(-1*Z_theta)*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1))/(sigma**2)
        denominator = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)

        return (numerator/denominator)
    
    def get_marginal(self, n, B, Z_grid, sigma):
        
        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)

        # print(f"theta_cum: {theta_cum}")
        Z_nb = Z_grid[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma**2)

        conditional_marginal = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1) # conditional marginal

        return(conditional_marginal.detach().numpy())


class model_theta_pi_sure(tr.nn.Module):
    """
    Optimize both theta and pi (SURE-PM). 
    """

    def __init__(self, Z, B, init_val_theta, init_val_pi, use_location=False, use_scale=True, device="cpu",
                 quantile_IQR = 0.95):

        super(model_theta_pi_sure, self).__init__()

        # real hyperparams 
        self.theta_real_vals = tr.nn.Parameter((tr.ones(B-1) * init_val_theta).to(device), requires_grad=True)
        self.pi_real_vals = tr.nn.Parameter((tr.ones(B) * init_val_pi).to(device), requires_grad=True)

        self.B = B
        self.min_Z = min(Z).to(device)
        self.max_Z = max(Z).to(device)

        self.median_Z = tr.median(Z).to(device)
        self.lower_quantile_Z = tr.quantile(Z, 0.5-quantile_IQR/2).to(device)
        self.higher_quantile_Z = tr.quantile(Z, 0.5+quantile_IQR/2).to(device)
        self.log_IQR_Z = tr.log(self.higher_quantile_Z - self.lower_quantile_Z ).to(device)

        self.use_location=use_location
        self.use_scale=use_scale

        self.location=tr.nn.Parameter(tr.zeros(1).to(device)*self.median_Z, requires_grad=use_location)
        self.log_scale=tr.nn.Parameter(tr.ones(1).to(device)*self.log_IQR_Z, requires_grad=use_scale)

        self.device = device
        
    def forward(self):
        """Forward pass through the network. 
        Input:
            NONE
            
        Output: 
            a (B x 1) tensor of pi, the probability vector, 
                B: number of theta values in grid
        """
        return (tr.nn.Softmax(dim=0)(self.theta_real_vals), tr.nn.Softmax(dim=0)(self.pi_real_vals))
    
    def get_theta_grid_and_pi(self, n, B):
        theta_diff, pi_param = self.forward()
        theta_cum = tr.concat((tr.zeros(1).to(self.device), theta_diff), dim=0)
        theta_cum = (tr.cumsum(theta_cum, dim = 0) - 0.5) * tr.exp(self.log_scale)

        theta_cum = theta_cum[None, :]
        theta_grid = theta_cum.expand(n, B) + self.location

        pi_param = pi_param[None, :]
        pi_param = pi_param.expand(n, B)

        return(theta_grid, pi_param)


    def opt_func(self, Z, n, B, sigma): 
        """Takes a matrix of square of (z-theta) and features, returns a scalar to be optimized.
        sigma: n x 1 tensor
        """ # SURE of t(z) = z + sigma * s(z; w)

        assert len(sigma.shape) == 1

        theta_grid, pi_param = self.get_theta_grid_and_pi(n, B)
        
        # print(f"theta_cum: {theta_cum}")
        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_grid
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2)
        
        numerator = ((pi_param*Z_theta_sq*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)) 
        denominator = (pi_param*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)
        ratio = 2*numerator/denominator

        score_squared = (self.compute_score(Z, n, B, sigma))**2 
        score_term = score_squared*(sigma**4) 

        sigma_squared = (sigma**2) 

        return (ratio.sum() - score_term.sum() - sigma_squared.sum())/n 
    
    def compute_score(self, Z, n, B, sigma, verbose=False): 
        """
        sigma.shape = (n,) tensor
        """

        assert len(sigma.shape) == 1

        theta_grid, pi_param = self.get_theta_grid_and_pi(n, B)

        Z_nb = Z[:, None]
        # print(f"type(Z_nb): {type(Z_nb)}")
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_grid
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2)
        
        numerator = ((pi_param*(-1*Z_theta)*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1))/(sigma**2)
        denominator = (pi_param*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)

        if verbose:
            print(f"theta pi sure sigma[:, None]: {sigma[:, None]}")
            print(f"Z_theta_by_sigma_sq.shape : {Z_theta_by_sigma_sq.shape}")
            print(f"numerator.shape : {numerator.shape}")
            print(f"denominator.shape : {denominator.shape}")
        return (numerator/denominator)
    
    def get_prior(self, n, B, Z):
        Z = Z.detach().numpy()

        theta_grid, pi_param = self.get_theta_grid_and_pi(n, B)
        theta_grid = theta_grid.detach().numpy()[0,:]
        pi_param = pi_param.detach().numpy()[0,:]

        return theta_grid.reshape(self.B,), pi_param.reshape(self.B,)
    
    def get_theta_hat(self, n, B, Z_grid, sigma):
        """
        Return n-length np.array of shrinkage rules for n values of Z
        """

        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = sigma*tr.ones(n,)
            variance = sigma**2
        else:
            variance = sigma**2
        
        score = self.compute_score(Z_grid, n, B, sigma)

        theta_hat = Z_grid + variance * score

        return(theta_hat.cpu().detach().numpy())
    
    def get_marginal(self, n, B, Z_grid, sigma):
        """
        * sigma is a float
        """

        theta_grid, pi_param = self.get_theta_grid_and_pi(n, B)

        # print(f"theta_cum: {theta_cum}")
        Z_nb = Z_grid[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_grid
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma**2)

        normal_pdf = tr.exp(-Z_theta_by_sigma_sq/2) / (np.sqrt(2*np.pi) * sigma )

        conditional_marginal = (pi_param * normal_pdf).sum(axis = 1) # conditional marginal

        return(conditional_marginal.detach().numpy())



class model_pi_sure_sparse(tr.nn.Module):

    """
    Optimize both theta and pi with sparsemax instead of softmax. 
    """

    def __init__(self, Z, B, init_val, device="cpu",
                 quantile_IQR = 0.95):

        super(model_pi_sure_sparse, self).__init__()
        # real hyperparams 
        self.real_params = tr.nn.Parameter((tr.ones(B) * init_val).to(device), requires_grad=True)

        self.min_Z = min(Z).to(device)
        self.max_Z = max(Z).to(device)

        self.median_Z = tr.median(Z).to(device)
        self.lower_quantile_Z = tr.quantile(Z, 0.5-quantile_IQR/2).to(device)
        self.higher_quantile_Z = tr.quantile(Z, 0.5+quantile_IQR/2).to(device)
        self.log_IQR_Z = tr.log(self.higher_quantile_Z - self.lower_quantile_Z ).to(device)

        self.location=tr.nn.Parameter(tr.zeros(1).to(device)*self.median_Z, requires_grad=False)
        self.log_scale=tr.nn.Parameter(tr.ones(1).to(device)*self.log_IQR_Z, requires_grad=False)

        self.device = device
        
    def forward(self):
        """Forward pass through the network. 
        Input:
            NONE
            
        Output: 
            a (B x 1) tensor of pi, the probability vector, 
                B: number of theta values in grid
        """
        return sparsemax(self.real_params, dim=0)

    def get_theta_grid_and_pi(self, n, B):
        pi_param = self.forward()
        pi_param = pi_param[None, :]
        pi_nB = pi_param.expand(n, B)

        theta_diff = (1/(B-1))*tr.ones(B-1).to(self.device)
        theta_cum = (tr.cumsum( tr.concat((tr.zeros(1).to(self.device),
                      theta_diff), dim=0), dim = 0) 
                      - 0.5) * tr.exp(self.log_scale)
        theta_cum = theta_cum[None, :]
        theta_nB = theta_cum.expand(n, B) + self.location 

        return(theta_nB, pi_nB)
    
    def opt_func(self, Z, n, B, sigma): 
        """Takes a matrix of square of (z-theta) and features, returns a scalar to be optimized.
        """ # SURE of t(z) = z + sigma * s(z; w)

        assert len(sigma.shape) == 1
        
        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)
        
        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2) 
        
        numerator = ((pi_nB*Z_theta_sq*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)) 
        denominator = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)
        ratio = 2*numerator/denominator

        score_squared = (self.compute_score(Z, n, B, sigma))**2 
        score_term = score_squared*(sigma**4) 

        sigma_squared = (sigma**2) 

        return (ratio.sum() - score_term.sum() - sigma_squared.sum())/n 
    
    def compute_score(self, Z, n, B, sigma):

        assert len(sigma.shape) == 1

        theta_nB, pi_nB = self.get_theta_grid_and_pi(n, B)
        
        Z_nb = Z[:, None]
        Z_nb = Z_nb.expand(n, B) 
        Z_theta = Z_nb - theta_nB
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2) 
        
        numerator = ((pi_nB*(-1*Z_theta)*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1))/(sigma**2)
        denominator = (pi_nB*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)

        return (numerator/denominator) 
    
 ## MARK: - Covariates


def create_initialized_network(input_size, hidden_sizes, output_size, 
                               activation_fn=tr.nn.ReLU(), 
                               device="cpu"):
    """ 
    Create a fully connected neural network with specified sizes, and initialized weights.

    Args:
        input_size: int, size of the input layer
        hidden_sizes: list of ints, each element is the size of the corresponding hidden layer
        output_size: int, size of the output layer
        activation_fn: torch.nn.Module, the activation function to use
            Smooth activations
                * ELU
                * SiLU (swish)
                * LogSigmoid
                * CELU
        init_std: float, standard deviation of the normal distribution to draw initial weights from
    
    Returns:
        net: torch.nn.Sequential, the initialized network
    """
    
    if hidden_sizes != None: 
        all_sizes = [input_size] + list(hidden_sizes) + [output_size]
    else: 
        all_sizes = [input_size] + [output_size]
    
    modules = []
    for l in range(len(all_sizes)-1):
        lu = tr.nn.Linear(all_sizes[l], all_sizes[l+1], bias=True) # Check if the values are changing
        # add to the list of modules
        modules.append(lu)
        # add activation function (if not in the last layer)
        if l < (len(all_sizes)-2):
            modules.append(activation_fn)

    sequential = tr.nn.Sequential(*modules)
    sequential.to(device)

    return sequential



class model_covariates(tr.nn.Module):
    """
    Optimize over theta grid and pi with covariates (SURE-THING). 
    """

    def __init__(self, X, Z, d=2, B=100, hidden_sizes=(8,8), use_location=False, use_scale=False,
                 drop_sigma=False, device="cpu", activation_fn=tr.nn.ReLU(), quantile_IQR=0.95, skip_connect=False):
        
        super(model_covariates, self).__init__()

        #  hyperparams 
        self.device = device
        self.B = B
        self.d = d

        self.min_Z = min(Z).to(device)
        self.max_Z = max(Z).to(device)
        self.median_Z = tr.median(Z)
        self.lower_quantile_Z = tr.quantile(Z, 0.5-quantile_IQR/2)
        self.higher_quantile_Z = tr.quantile(Z, 0.5+quantile_IQR/2)
        self.IQR_Z = self.higher_quantile_Z - self.lower_quantile_Z 

        self.use_location=use_location
        self.use_scale=use_scale
        self.skip_connect=skip_connect

        if use_location and use_scale:
            output_size=2*B+1
        elif (not (use_location or use_scale)):
            output_size=2*B-1
        else:
            output_size=2*B

        if skip_connect == False or hidden_sizes == None: 
            if drop_sigma: 
                self.sequential = create_initialized_network(input_size=self.get_input_size(X)-1, # no sigma covariate 
                                                            hidden_sizes=hidden_sizes, 
                                                            output_size= output_size,
                                                            activation_fn=activation_fn,
                                                            device=device)
            else:  
                self.sequential = create_initialized_network(input_size=self.get_input_size(X), 
                                                            hidden_sizes=hidden_sizes, 
                                                            output_size= output_size,
                                                            activation_fn=activation_fn,
                                                            device=device)
            self.hidden_layers = None 
        else: 
            if drop_sigma: 
                input_size = self.get_input_size(X)-1 
                self.hidden_layers = tr.nn.ModuleList() 
                prev_size = input_size 
                for hidden_size in hidden_sizes:
                    self.hidden_layers.append(
                        tr.nn.Linear(prev_size, hidden_size)  # +input_size for concatenation
                    )
                    prev_size = hidden_size
                self.hidden_layers.to(device) 
                self.output_layer = tr.nn.Linear(prev_size + input_size, output_size)
                self.output_layer.to(device) 
            else:  
                input_size = self.get_input_size(X) 
                self.hidden_layers = tr.nn.ModuleList() 
                prev_size = input_size 
                for hidden_size in hidden_sizes:
                    self.hidden_layers.append(
                        tr.nn.Linear(prev_size, hidden_size)  # +input_size for concatenation
                    )
                    prev_size = hidden_size
                self.hidden_layers.to(device) 
                self.output_layer = tr.nn.Linear(prev_size + input_size, output_size)
                self.output_layer.to(device) 
        # print(f"self.sequential: {self.sequential}")
        # print(f"next(self.sequential.parameters()).is_cuda: {next(self.sequential.parameters()).is_cuda}")
        
    def forward(self, feature_representation):
        """Forward pass through the network. 
        Input:
            X: an (n x (m+1)) tensor
                n: number of observations
                m+1: number of covariates (m is the dim of X tilde, and the last column is sigma)

            B: size of the grid

            hidden_sizes: a tuple of integers, where each integer is the number of hidden units in a layer.

            
        Output: 
            a touple of two items:
            (1) an n x (B-1) tensor: relative differences of theta_b values (probability vec) 
            (2) an n x B tensor:pi(X_i), probability vector

            where
                n: number of observations
                B: number of theta values in grid
        """
        # print(f"forward pass feature_representation.shape: {feature_representation.shape}")
        if self.skip_connect == False or self.hidden_layers == None: 
            params_NN = self.sequential(feature_representation) # size: n x (2*(B-1))
        else: 
            original_input = feature_representation
            x = feature_representation
            for layer in self.hidden_layers:
                # x = tr.cat([x, original_input], dim=1)  # concatenate input
                x = tr.nn.functional.relu(layer(x)) 
            x = tr.cat([x, original_input], dim=1) 
            params_NN = self.output_layer(x) # concatenate input
        theta_nBminus1 = tr.nn.Softmax(dim=1)(params_NN[:, :(self.B-1)]) # first B-1 columns
        pi_nB = tr.nn.Softmax(dim=1)(params_NN[:, (self.B-1):(2*self.B-1)]) # middle B columns

        if self.use_location and self.use_scale:
            location = params_NN[:, 2*self.B-1] 
            # location = params_NN[:, 2*self.B-1] + median_Z
            scale = tr.exp(params_NN[:, 2*self.B])
            # scale = tr.exp(params_NN[:, 2*self.B]) * IQR_Z
            return (theta_nBminus1, pi_nB, location, scale)
        elif self.use_location:
            location = params_NN[:, 2*self.B-1]
            # location = params_NN[:, 2*self.B-1] + median_Z
            return (theta_nBminus1, pi_nB, location)
        elif self.use_scale:
            scale = tr.exp(params_NN[:, 2*self.B-1]) # overflow from initialization
            # scale = tr.exp(params_NN[:, 2*self.B]) * IQR_Z
            return (theta_nBminus1, pi_nB, scale)
        else:
            return (theta_nBminus1, pi_nB)
    
    def get_input_size(self, X):
        mplus1 = X.shape[1]
        # print(f"get_input_size: {self.feature_representation(tr.ones([1,mplus1])).shape[1]}")
        return self.feature_representation(tr.ones([1,mplus1])).shape[1]
    
    def get_theta_grid_and_pi_nB(self, feature_representation):
        """
        Grid modeling
        """

        if self.use_location and self.use_scale:
            theta_nBminus1, pi_nB, location, scale = self.forward(feature_representation)
        elif self.use_location:
            theta_nBminus1, pi_nB, location = self.forward(feature_representation)
        elif self.use_scale:
            theta_nBminus1, pi_nB, scale  = self.forward(feature_representation)
        else:
            theta_nBminus1, pi_nB  = self.forward(feature_representation)
        
        n, B = pi_nB.shape

        theta_cumulative_diff = tr.concat((tr.zeros(n,1).to(self.device), theta_nBminus1), dim=1)
        theta_cumulative_diff = tr.cumsum(theta_cumulative_diff, dim = 1)

        if self.use_location and self.use_scale:
            theta_grid = (theta_cumulative_diff - 0.5) * tr.transpose(scale.repeat(self.B,1), 0, 1) + tr.transpose(location.repeat(self.B,1), 0, 1)
        elif self.use_location:
            theta_grid = (theta_cumulative_diff - 0.5) * self.IQR_Z + tr.transpose(location.repeat(self.B,1), 0, 1)
        elif self.use_scale:
            theta_grid = (theta_cumulative_diff - 0.5) * tr.transpose(scale.repeat(self.B,1), 0, 1) + self.median_Z
        else: 
            theta_grid = (theta_cumulative_diff - 0.5) * self.IQR_Z  + self.median_Z
        
        return(theta_grid, pi_nB)

        

    def feature_representation(self, X):
        """Takes a matrix of covariates, returns a fixed-length matrix (n x 2*m).
        2*m because there is a polynomial term
        """
        n = X.shape[0]
        m = X.shape[1] - 1
        actual_degree = self.d-1
        # print(f"n: {n}")
        # print(f"m: {m}")
        # print(f"actual_degree: {actual_degree}")
        feature_representation = tr.zeros(n, actual_degree*m + 1) 
        feature_representation[:, -1] = X[:,-1] # last column is sigma


        for j in range(m): # columns, dimension of X_tilde
            feature_representation[:, (actual_degree*j):(actual_degree*j+actual_degree)] = tr.pow(X[:, j].unsqueeze(1), tr.arange(1, actual_degree + 1, device=X.device)).reshape(n, actual_degree)

        return feature_representation.to(self.device)

    def feature_misrepresentation(self, X): # misspecified case
        """
        Drop sigma
        Takes a matrix of covariates, returns a fixed-length matrix (n x 2*m).
        2*m because there is a polynomial term
        """
        n = X.shape[0]
        m = X.shape[1] - 1
        actual_degree = self.d - 1

        feature_representation = tr.zeros(n, actual_degree*m)  # drop sigma, which would be actual_degree*m + 1
        for i in range(n): # rows
                for j in range(m): # columns, dimension of X_tilde
                    feature_representation[i, (actual_degree*j):(actual_degree*j+actual_degree)] = tr.tensor([np.power(X[i, j].detach().numpy(), k+1) for k in range(actual_degree)])

        return feature_representation.to(self.device)
    
    def opt_func_SURE(self, Z, feature_representation, X): 
        """Takes Z and feature representations,
        
        """ # SURE of t(z) = z + sigma * s(z; w)
        
        theta_grid, pi_nB = self.get_theta_grid_and_pi_nB(feature_representation)

        n, B = pi_nB.shape
        sigma = X[:,-1]

        Z_expanded = (Z[:, None].expand(n,B) ).clone().requires_grad_(False)
        Z_theta = Z_expanded - theta_grid 
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = (Z_theta_sq / sigma[:, None]**2)
        
        normal_kernel = tr.exp(-Z_theta_by_sigma_sq/2) 
        
        conditional_marginal = (normal_kernel*pi_nB).sum(axis=1) 
        
        term1 = 2 * ( Z_theta_sq * normal_kernel * pi_nB ).sum(axis=1) / (conditional_marginal ) 
        
        score = ( -Z_theta * normal_kernel * pi_nB).sum(axis=1) / (conditional_marginal * sigma**2)
        score_sq = score**2
        
        term2 = (sigma**4) * (score_sq )
        

        return (term1 - term2 - sigma**2).sum()/n
    
    def opt_func_NLL(self, Z, feature_representation, X): 
        # normalized by n

        theta_grid, pi_nB = self.get_theta_grid_and_pi_nB(feature_representation)

        n, B = pi_nB.shape
        sigma = X[:,-1]

        Z_expanded = (Z[:, None].expand(n,B)).clone().requires_grad_(False)
        Z_theta = Z_expanded - theta_grid 
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = (Z_theta_sq / sigma[:, None]**2 )

        normal_kernel = tr.exp(-Z_theta_by_sigma_sq/2) # n x B

        conditional_marginal_kernel = (normal_kernel*pi_nB).sum(axis=1) 

        return (- tr.log(conditional_marginal_kernel.sum()))/n

    def compute_score(self, Z, feature_representation, X, verbose=False):

        theta_grid, pi_nB = self.get_theta_grid_and_pi_nB(feature_representation)

        n, B = pi_nB.shape
        sigma = X[:,-1]

        # construct theta_b
        # expand (duplicate each row B times) and center with -self.theta1
        Z_expanded = Z[:, None].expand(n,B)
        Z_theta = Z_expanded
        Z_theta = Z_expanded - theta_grid # keep first column fixed
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = (Z_theta_sq / sigma[:, None]**2 )

        normal_kernel = tr.exp(-Z_theta_by_sigma_sq/2) # n x B

        # remove the 1/sqrt(2pi sigma^2) from numerator, denominator

        # n-length vector for f(z_i | x_i)
        conditional_marginal = (normal_kernel*pi_nB).sum(axis=1) 

        # n-length vector for score, s(z_i, x_i; w)
        score = ( -Z_theta * normal_kernel * pi_nB).sum(axis=1) / (conditional_marginal * sigma**2) 

        if verbose:
            print(f"sigma[:, None]: {sigma[:, None]}")
            print(f"Z_theta_by_sigma_sq.shape : {Z_theta_by_sigma_sq.shape}")
            print(f"numerator.shape : {( -Z_theta * normal_kernel * pi_nB).sum(axis=1).shape}")
            print(f"denominator.shape : {(conditional_marginal * sigma**2).shape}")

        return(score)
    
    def get_prior(self, feature_representation):

        theta_grid, pi_nB = self.get_theta_grid_and_pi_nB(feature_representation)
        n, B = pi_nB.shape
        
        # standardize so the grid begins at min(Z)
        theta_grid = theta_grid[None, :] # ?
        theta_grid = theta_grid.detach().numpy()

        return theta_grid.reshape(n,B), pi_nB.detach().numpy()
    
    def get_theta_hat(self, n, Z_grid, X):
        """
        * Z_grid is tensor
        Return n-length np.array of shrinkage rules for n values of Z_grid
        """

        if isinstance(X, float) or isinstance(X, int):
            X = X * tr.ones(n,1)
            variance = X**2

        else:
            variance = X[:,-1]**2
            

        feature_representation = self.feature_representation(X)
        
        score = self.compute_score(Z_grid, feature_representation, X)

        theta_hat = Z_grid + variance * score

        return(theta_hat.cpu().detach().numpy())
    
    def get_marginal(self, Z_grid, X):

        # n is not defined...
        # TODO: add self.n to the model class?
        if isinstance(X, float) or isinstance(X, int):
            X = X * tr.ones(n, 1)


        feature_representation = self.feature_representation(X)
        
        theta_grid, pi_nB = self.get_theta_grid_and_pi_nB(feature_representation)


        n, B = pi_nB.shape
        sigma = X[:, -1]

        Z_expanded = (Z_grid[:, None].expand(n,B)).clone().requires_grad_(False)
        Z_theta = Z_expanded - theta_grid # keep first column fixed
        Z_theta_sq = Z_theta**2
        Z_theta_by_sigma_sq = Z_theta_sq / (sigma[:, None]**2)


        normal_pdf = tr.exp(-Z_theta_by_sigma_sq/2) / (tr.tensor(np.sqrt(2*np.pi)) * sigma[:, None]) # n x B 

        conditional_marginal = (normal_pdf*pi_nB).sum(axis=1) 

        return(conditional_marginal.detach().numpy())

## MARK: - Covariates (SURE LS)

class model_sure_ls(tr.nn.Module): 
    """
    Optimize both mean and variance as a function of covariates in regression setting (SURE-LS). 
    """

    def __init__(self, X, Z, d=2, hidden_sizes=(8,8), device="cpu", activation_fn=tr.nn.ReLU()): 

        super(model_sure_ls, self).__init__()

        #  hyperparams 
        self.device = device
        self.d = d

        self.sequential = create_initialized_network(input_size=self.get_input_size(X),
                                                     hidden_sizes=hidden_sizes, 
                                                     output_size= 2,
                                                     activation_fn=activation_fn,
                                                     device=device) 
    
    def forward(self, feature_representation):
        params_NN = self.sequential(feature_representation)
        sigma = feature_representation[:, -1] 

        A_n = tr.exp(params_NN[:, 0]) 
        m_n = params_NN[:, 1] 

        lambda_n = (sigma**2) / (sigma**2 + A_n) 
        b_n = lambda_n*m_n 

        # lambda_n = 1 / (1 + tr.exp(-params_NN[:, 0])) 
        # b_n = params_NN[:, 1] 

        return (lambda_n, b_n) 
    
    def get_input_size(self, X):
        mplus1 = X.shape[1]
        # print(f"get_input_size: {self.feature_representation(tr.ones([1,mplus1])).shape[1]}")
        return self.feature_representation(tr.ones([1,mplus1])).shape[1] 

    def feature_representation(self, X):
        """Takes a matrix of covariates, returns a fixed-length matrix (n x 2*m).
        2*m because there is a polynomial term
        """
        n = X.shape[0]
        m = X.shape[1] - 1
        actual_degree = self.d-1
        # print(f"n: {n}")
        # print(f"m: {m}")
        # print(f"actual_degree: {actual_degree}")
        feature_representation = tr.zeros(n, actual_degree*m + 1) 
        feature_representation[:, -1] = X[:,-1] # last column is sigma


        for j in range(m): # columns, dimension of X_tilde
            feature_representation[:, (actual_degree*j):(actual_degree*j+actual_degree)] = tr.pow(X[:, j].unsqueeze(1), tr.arange(1, actual_degree + 1, device=X.device)).reshape(n, actual_degree)

        return feature_representation.to(self.device) 
    
    def opt_func_SURE(self, Z, feature_representation, X): 

        lambda_n, b_n = self.forward(feature_representation) 
        n = len(lambda_n) 
        sigma = X[:,-1] 

        term1 = (lambda_n * Z - b_n)**2 
        term2 = 2 * (sigma**2) * lambda_n 

        return (sigma**2 + term1 - term2).sum()/n 
    
    def opt_func_NLL(self, Z, feature_representation, X): 

        lambda_n, b_n = self.forward(feature_representation) 
        n = len(lambda_n) 
        sigma = X[:,-1] 

        term1 = lambda_n*((Z - (b_n / lambda_n))**2) / (sigma**2) 
        term2 = 2*tr.log(sigma) - tr.log(lambda_n) 

        return ((term1 + term2).sum()) / n
    
    def compute_score(self, Z, feature_representation, X): 

        lambda_n, b_n = self.forward(feature_representation) 
        sigma = X[:,-1] 

        s_G_n = (b_n - (Z * lambda_n)) / (sigma**2) 

        return s_G_n 
    
    def get_theta_hat(self, Z, X): 

        feature_representation = self.feature_representation(X)

        lambda_n, b_n = self.forward(feature_representation) 

        return (b_n + (1 - lambda_n) * Z) 

## MARK: - EBCF

class model_EBCF_NPreg(tr.nn.Module): 
    """
    Fitting the regression model for EBCF (Ignatiadis and Wager, 2019), using a neural network with covariates. 
    """

    def __init__(self, X, Z, d=2, hidden_sizes=(8,8), device="cpu", activation_fn=tr.nn.ReLU()): 

        super(model_EBCF_NPreg, self).__init__()

        #  hyperparams 
        self.device = device
        self.d = d

        self.sequential = create_initialized_network(input_size=self.get_input_size(X),
                                                     hidden_sizes=hidden_sizes, 
                                                     output_size=1,
                                                     activation_fn=activation_fn,
                                                     device=device) 
    
    def forward(self, feature_representation):

        params_NN = self.sequential(feature_representation)
        Z_hat_n = params_NN[:, 0]

        return Z_hat_n
    
    def get_input_size(self, X):

        mplus1 = X.shape[1]
        # print(f"get_input_size: {self.feature_representation(tr.ones([1,mplus1])).shape[1]}")

        return self.feature_representation(tr.ones([1,mplus1])).shape[1] 

    def feature_representation(self, X):

        """
        Takes a matrix of covariates, returns a fixed-length matrix (n x 2*m).
        2*m because there is a polynomial term
        """
        n = X.shape[0]
        m = X.shape[1] - 1
        actual_degree = self.d-1
        # print(f"n: {n}")
        # print(f"m: {m}")
        # print(f"actual_degree: {actual_degree}")
        feature_representation = tr.zeros(n, actual_degree*m + 1) 
        feature_representation[:, -1] = X[:,-1] # last column is sigma


        for j in range(m): # columns, dimension of X_tilde
            feature_representation[:, (actual_degree*j):(actual_degree*j+actual_degree)] = tr.pow(X[:, j].unsqueeze(1), tr.arange(1, actual_degree + 1, device=X.device)).reshape(n, actual_degree)

        return feature_representation.to(self.device) 
    
    def opt_func_MSE(self, Z, feature_representation, X): 

        Z_hat_n = self.forward(feature_representation) 
        n = len(Z) 

        return ((Z - Z_hat_n)**2).sum()/n 
    
    def get_Z_hat(self, Z, X): 

        feature_representation = self.feature_representation(X) 
        Z_hat_n = self.forward(feature_representation) 

        return Z_hat_n 

def SURE_EBCF(A_param, *args): 
    '''
    SURE function for EBCF model (Ignatiadis and Wager, 2019). 
    '''

    Z, variance, Z_hat = args 

    term = variance + (variance**2)*((Z-Z_hat)**2)/((variance+A_param)**2) - 2*(variance**2)/(A_param+variance) 
    sure_term = np.mean(term)

    return sure_term

def theta_hat_EBCF(theta, Z, X, Z_hat): 
    '''
    Estimated theta function for EBCF model (Ignatiadis and Wager, 2019). 
    '''

    if Z.is_cuda:
        Z = Z.cpu().detach().numpy()
        n = Z.shape[0]
    else:
        Z = Z.detach().numpy()
        n = Z.shape[0]
    
    if Z_hat.is_cuda:
        Z_hat = Z_hat.cpu().detach().numpy()
    else:
        Z_hat = Z_hat.detach().numpy() 

    if isinstance(X, float) or isinstance(X, int):
        variance = X[:,-1]**2 * np.ones(n,)
    elif X.is_cuda:
        variance = X[:,-1].cpu().detach().numpy().reshape(n,)**2
    else:
        variance = X[:,-1].detach().numpy().reshape(n,)**2
    
    A_hat = optimize.fminbound(SURE_EBCF, 0, 1000, args=(Z, variance, Z_hat)) 
    theta_hat = A_hat*Z/(A_hat+variance) + variance*Z_hat/(A_hat+variance) 
    MSE = np.linalg.norm(theta_hat - theta)**2/n 
    SURE = SURE_EBCF(A_hat, Z, variance, Z_hat) 

    return theta_hat, MSE, SURE, A_hat 

 ## MARK: - Parametric

def SURE_G(lambda_param, *args):
    """
    Unbiased risk estimate of shrinkage towards the grand mean.
    lambda_param: float, shrinkage factor. Larger lambda_param means less shrinkage.
    *args = (Z, X)
    n: number of observations
    Z: np array, the data observed. Notated X_i in the paper
    variance: np array, the vector of variances. Notated A_i in the paper
    """
    n, Z, variance, grand_mean = args

    term1 = variance**2 / (variance + lambda_param)**2
    term1 *= (Z - grand_mean)**2
    # print(f"term1: {term1}")
    term1 = np.mean(term1)

    term2 = variance / (variance + lambda_param)
    term2 *= (lambda_param - variance + 2*variance/n)
    term2 = np.mean(term2)

    return term1+term2

def theta_hat_G(theta, Z, X):
    """
    Shrinkage towards the grand mean of Z
    lambda_parm: float, shrinkage factor. Larger lambda_param means less shrinkage.

    theta: np array, true mean. To get MSE
    Z: tensor, the data observed. Notated X_i in the paper
    X: tensor, the vector of variances. Notated A_i in the paper
    """

    if Z.is_cuda:
        Z = Z.cpu().detach().numpy()
        n = Z.shape[0]
    else:
        Z = Z.detach().numpy()
        n = Z.shape[0]

    if isinstance(X, float) or isinstance(X, int):
        variance = X[:,-1]**2 * np.ones(n,)
    elif X.is_cuda:
        variance = X[:,-1].cpu().detach().numpy().reshape(n,)**2
    else:
        variance = X[:,-1].detach().numpy().reshape(n,)**2


    grand_mean = np.mean(Z)
    lambda_hat = optimize.fminbound(SURE_G, 0, 1000, args=(n, Z, variance, grand_mean))
    theta_hat_G = (lambda_hat / (variance + lambda_hat)) * Z + (variance / (variance + lambda_hat)) * grand_mean

    MSE = np.linalg.norm(theta_hat_G - theta)**2/n
    SURE = SURE_G(lambda_hat, n, Z, variance, grand_mean)

    return theta_hat_G, MSE, SURE, grand_mean, lambda_hat

