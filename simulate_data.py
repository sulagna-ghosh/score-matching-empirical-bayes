import numpy as np
import numpy.random as rn
import torch as tr
from scipy.stats import norm

device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")
print(f"simulate_data.py: device = {device}")

def simulate_data_normal_nocovariates(n=1000, mu_theta=10, sigma_theta=1, sigma=1): 
    '''
    Simulates data with: 
    theta ~ N(mu_theta, sigma_theta) 
    Z | theta ~ N(theta, sigma) 
    These are the experiments done in table 1 in section 6.1. 
    '''

    # theta ~ Normal(mu_theta, sigma_theta)
    theta = mu_theta + sigma_theta*rn.normal(size = (n,))

    # heteroskedastic
    if isinstance(sigma, int) or isinstance(sigma, float):
        sigma = sigma*np.ones((n,))

    Z = theta + sigma*rn.normal(size = (n,))

    # Tensors
    Z = tr.tensor(Z, requires_grad = False) 
    X = tr.tensor(sigma, requires_grad = False)
    X = tr.reshape(X, (n, 1))

    Z = Z.to(device)
    X = X.to(device)

    return (theta, Z, X) 

def simulate_data_discrete_nocovariates(n=1000, k=5, val_theta = 3, sigma=1): 
    '''
    Simulates data with: 
    theta = val_theta for k of them and = 0 otehrwise 
    Z | theta ~ N(theta, sigma) 
    These are the experiments done in figure 2 in section 6.1. 
    '''

    theta = np.zeros(n)
    theta[0:k] = val_theta # the last n - k of theta is 0

    assert(theta.sum() == val_theta * k)

    # Z ~ Normal(theta, sigma^2)
    # heteroskedastic
    if isinstance(sigma, int) or isinstance(sigma, float):
        sigma = sigma*np.ones((n,))

    Z = theta + sigma*rn.normal(size = (n,))

    # Tensors
    Z = tr.tensor(Z, requires_grad = False)
    X = tr.tensor(sigma, requires_grad = False)
    X = tr.reshape(X, (n, 1)) 

    Z = Z.to(device)
    X = X.to(device)
    
    return (theta, Z, X) 

def xie(experiment, n=10000, device=device):
    '''
    Eight experiments: First four of them are the same experiments as c, d, e, f from Xie 2012 
    and rest are new additional experiments. These are the experiments done in figure 1 in section 6.2. 
    '''

    if experiment == "c":
        variance = rn.uniform(0.1, 1, size=(n,))
        sigma = np.sqrt(variance) 
        theta = variance
        Z = variance + sigma*rn.normal(size=(n,))

    elif experiment == "d":
        precision = rn.chisquare(df=10, size=(n,))
        variance = 1/precision
        sigma = np.sqrt(variance)
        theta = variance
        Z = variance + sigma*rn.normal(size=(n,))

    elif experiment == "d5":
        precision = rn.chisquare(df=5, size=(n,))
        variance = 1/precision
        sigma = np.sqrt(variance)
        theta = variance
        Z = variance + sigma*rn.normal(size=(n,))

    elif experiment == "e":
        bernoulli = rn.binomial(n=1, p=0.5, size=(n,))
        variance = 0.1*bernoulli + (0.5)*(1-bernoulli)
        # print(variance)

        theta = np.zeros(shape=(n,))

        for var, idx in zip(variance, range(n)):
            if var == 0.1:
                theta[idx] = rn.normal(loc=2, scale=np.sqrt(0.1))
            elif var == 0.5:
                theta[idx] = rn.normal(loc=0, scale=np.sqrt(0.5))
            else:  
                print("experiment (e) variance error")

        sigma = np.sqrt(variance)
        Z = theta + sigma*rn.normal(size=(n,))

    elif experiment == "f":
        variance = rn.uniform(0.1, 1, size=(n,))
        sigma = np.sqrt(variance) 
        theta = variance
        Z = rn.uniform(variance - np.sqrt(3)*sigma, variance + np.sqrt(3)*sigma,
                       size=(n,))
        
    elif experiment == "g": 
        # bernoulli1 = rn.binomial(n=1, p=0.5, size=(n,))
        # variance = 0.1*bernoulli1 + (0.5)*(1-bernoulli1)
        variance = rn.uniform(0.1, 0.5, size=(n,))

        bernoulli = rn.binomial(n=1, p=0.5, size=(n,))
        theta = bernoulli*(variance) + (1-bernoulli)*(10*variance)
        sigma = np.sqrt(variance)

        Z = theta + sigma*rn.normal(size=(n,))
    
    elif experiment == "h": 
        variance = rn.uniform(0.1, 1, size=(n,))

        theta = rn.poisson(lam = 2*variance) 
        sigma = np.sqrt(variance)

        Z = theta + sigma*rn.normal(size=(n,)) 

    elif experiment == "i": 
        variance = rn.uniform(1.5, 2.5, size=(n,))
        sigma = np.sqrt(variance)

        X_prev = rn.uniform(0, 1, size=(n, 5))
        theta =  (np.pi*X_prev[:,0]*X_prev[:,1]) + 20*((X_prev[:,2]-0.5)**2) + 5*X_prev[:,3] + 2*rn.normal(size=(n,)) 

        Z = theta + sigma*rn.normal(size=(n,)) 

    elif experiment == "j": 
        X_prev = rn.uniform(0, 1, size=(n, 1)) 
        variance = 2*(X_prev[:,0]**2) + 5*X_prev[:,0] + 1 
        sigma = np.sqrt(variance) 

        m_X = 2*variance + 0.5
        A_X = 0.25*variance
        theta = m_X + np.sqrt(A_X)*rn.normal(size=(n,)) 

        Z = theta + sigma*rn.normal(size=(n,))

    Z = tr.tensor(Z, requires_grad = False)

    if experiment != "i" and experiment != "j": 
        X = tr.tensor(sigma, requires_grad = False)
        X = tr.reshape(X, (n, 1)) 
    else: 
        X_wo_sigma = tr.tensor(X_prev, requires_grad = False) 
        X_sigma = tr.tensor(sigma, requires_grad = False)
        X_sigma = tr.reshape(X_sigma, (n, 1)) 
        X = tr.cat((X_wo_sigma, X_sigma), dim=1) 

    Z = Z.to(device)
    X = X.to(device)

    return (theta, Z, X) 

def xie_Z_grid(n, experiment, sigma_i, expanded=False,
               endpoints=None):
    """
    sigma_i is a float

    Create a grid of Z given variance for the experiments c, d, e, and f.
    1st and 99th quantiles of the distribution Z | theta ~ N(theta, A)
    """

    if endpoints is None:

        if expanded:
            sigma_i = sigma_i*30

        if experiment != "e":
            theta = sigma_i**2 # theta = variance in experiments other than e
        else:
            if (sigma_i == np.sqrt(0.1)) & (not expanded):
                theta = 2
            elif (sigma_i == 2*np.sqrt(0.1)) & expanded:
                theta = 2
            else:
                theta = 0
        
        if experiment != "f":

            if experiment != "d5": 
                Z_grid = np.linspace(norm.ppf(0.01, loc=theta, scale=sigma_i), # scale is SD
                                    norm.ppf(0.99, loc=theta, scale=sigma_i), n)
                
            else: 

                Z_grid = np.linspace(norm.ppf(0.0001, loc=theta, scale=sigma_i), # scale is SD
                                    norm.ppf(0.9999, loc=theta, scale=sigma_i), n)
        else:
            Z_grid = np.linspace(theta - np.sqrt(3)*sigma_i, # scale is SD
                                theta + np.sqrt(3)*sigma_i, n)
    
    else:
        Z_grid = np.linspace(endpoints[0], endpoints[1], n)


    # Z_grid = Z_grid.reshape((n,1))
    
    # print(f"Z_grid.shape: {Z_grid.shape}")
    
    return tr.tensor(Z_grid, requires_grad=False)
