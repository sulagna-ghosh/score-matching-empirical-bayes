# Contains functions to train different models used (section 6) 

import models 

import numpy as np
import numpy.random as rn
import torch as tr
import cvxpy as cp 
import math

import simulate_data # simulate_data.device


def train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', 
                        init_val_theta = tr.log(tr.Tensor([1.5])), init_val_pi = tr.log(tr.Tensor([1.5])),
                        use_location=False, use_scale=False,
                        device=simulate_data.device, optimizer_str="adam",
                        lr=1e-2, n_iter=4000): 
    """
    Training function for SURE-PM. 
    is_cuda: if True, then generate model on simulate_data.device (which is GPU if GPU is available).
             So if is_cuda=True but only CPU is available, it will be on CPU.
    """
    
    # define model
    if opt_objective == 'pi-only': 
        model = models.model_pi_sure(Z, B, init_val=init_val_pi, device=device)
    elif opt_objective == 'theta-only': 
        model = models.model_theta_sure(Z, B, init_val=init_val_theta, 
                                        use_location=use_location, use_scale=use_scale, device=device)
    elif opt_objective == 'both': 
        model = models.model_theta_pi_sure(Z, B, init_val_theta=init_val_theta, init_val_pi=init_val_pi, 
                                           use_location=use_location, use_scale=use_scale, device=device)
    elif opt_objective == 'pi-sparse': 
        model = models.model_pi_sure_sparse(Z, B, init_val=init_val_pi, device=device) 

    model.to(device)
    
    # Tolerance for stopping criteria
    tol = 1e-6
    
    # sigma of Z
    sigma = X[:, -1] 

    # Optimizer
    if optimizer_str == "adam":
        optimizer = tr.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_str == "bfgs":
        optimizer = tr.optim.LBFGS(model.parameters(), lr=lr, history_size=75, line_search_fn = "strong_wolfe")
    else:
        print("optimizer_str typo")

    # Loss history and other variables to be saved
    losses = []
    theta_hats = [] # estimated theta_hat for each iterate
    two_norm_differences = []
    scores = []

    for i in range(n_iter):

        if optimizer_str == "bfgs":
            
            def closure():
                # Zero the gradients
                optimizer.zero_grad()

                # Compute the loss
                loss = model.opt_func(Z, n, B, sigma) 
                losses.append(loss.item())
            
                # Backprop on the loss
                loss.backward()
                # Return the loss

                return(loss)
            
            optimizer.step(closure)
        
        else: 

            optimizer.zero_grad()
            loss = model.opt_func(Z, n, B, sigma) 
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        score = model.compute_score(Z, n, B, sigma).cpu().detach().numpy()
        scores.append(score)
        theta_hat_iterate = Z.cpu().detach().numpy() + sigma.cpu().detach().numpy()**2 *score
        theta_hats.append(theta_hat_iterate)

        two_norm_differences.append(np.linalg.norm(theta_hat_iterate - theta)**2)
        
        # if i > 0:
            # loss_chng = abs((losses[i]-losses[i-1])/(losses[i-1])) # Using moving average
            # if loss_chng<tol:
                # break

        if np.isnan(losses[-1]): 
                break
    
    return (model, losses, np.array(scores), np.array(theta_hats), two_norm_differences[-1]) 


def train_npmle(n, B, Z, theta, X, quantile_IQR = 0.95): 
    '''
    Training function for NPMLE. 
    '''

    if X.is_cuda | Z.is_cuda: 
        X = X.cpu()
        Z = Z.cpu()

    # sigma of Z
    sigma = X[:, -1] 

    theta_diff = (1/(B-1))*tr.ones(B-1) 
    theta_cum = tr.cumsum(theta_diff, dim = 0)*(max(Z)-min(Z))
    theta_cum = theta_cum[None, :]
    theta_cum = theta_cum.expand(n, B-1) 
    
    Z_nb = Z[:, None]
    Z_nb = Z_nb.expand(n, B) 
    Z_theta = Z_nb - min(Z)
    Z_theta[:, 1:] = Z_theta[:, 1:] - theta_cum
    Z_theta_sq = Z_theta**2
    Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2) 
    
    # (Re)Defining numpy variables
    Z_theta_by_sigma_sq_np = Z_theta_by_sigma_sq.detach().numpy() 
    sigma_np = sigma.detach().numpy() 
    norm_dens_mat = np.exp(-Z_theta_by_sigma_sq_np/2)/(((2*math.pi)**0.5)*sigma_np[:, None]) 

    # Variable to be optimized
    pi_np = cp.Variable(B)

    # Objective and constraints
    objective = cp.Maximize(cp.sum(cp.log(norm_dens_mat @ pi_np)))
    constraints = [cp.sum(pi_np) == 1, pi_np >= 0]

    # Optimization
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver = cp.MOSEK)

    # Optimized objective and varaible values
    loss = prob.value
    pi_hat = tr.tensor(pi_np.value)
    pi_hat[pi_hat < 0] = 0

    # Optmized theta_hat
    pi_param_est = pi_hat[None, :]
    pi_param_est = pi_param_est.expand(n, B)
    numerator = ((pi_param_est*(-1*Z_theta)*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1))/(sigma**2) 
    denominator = (pi_param_est*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)
    score = numerator/denominator 
    theta_hat = Z + sigma**2 *score

    # L2 norm
    two_norm_difference = (np.linalg.norm(theta_hat.detach().numpy() - theta)**2)

    return (prob, loss, score.detach().numpy(), theta_hat.detach().numpy(), two_norm_difference, pi_hat) 


def train_covariates(X, Z, theta, objective="SURE", set_seed=None,
                     d=2, B=100, drop_sigma=False, device=simulate_data.device,
                     optimizer_str="adam", activation_fn=tr.nn.ReLU(), hidden_sizes=(8,8),
                     use_location=False, use_scale=False, skip_connect=False,
                     lr=1e-2):
    '''
    Training function for SURE-THING. 
    '''

    if set_seed is not None:
        tr.manual_seed(set_seed)

    model = models.model_covariates(X, Z, d=d, B=B, hidden_sizes=hidden_sizes, use_location=use_location, use_scale=use_scale,
                                    drop_sigma=drop_sigma, activation_fn=activation_fn, device=device, skip_connect=skip_connect)
    
    # print("In train covariates, before to device")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    
    model.to(simulate_data.device)

    if drop_sigma:
        # print("hello")
        feature_representation = model.feature_misrepresentation(X)
        # print(f"train_covariates X.shape: {X.shape}")
        # print(f"train_covariates feature_representation.shape: {feature_representation.shape}")
    else:
        feature_representation = model.feature_representation(X)
        # print(f"train_covariates X.shape: {X.shape}")
        # print(f"train_covariates feature_representation.shape: {feature_representation.shape}")

    if set_seed is not None: 
        rn.seed(set_seed)

    # Tolerance for stopping criteria
    tol = 1e-6
    # Number of iterations
    n_iter = 2000

    # Optimizer
    if optimizer_str == "adam":
        optimizer = tr.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_str == "bfgs":
        optimizer = tr.optim.LBFGS(model.parameters(), lr=lr, history_size=50)
    else:
        print("optimizer_str typo")

    # Loss history
    losses_SURE = []
    losses_NLL = []
    scores = []
    theta_hats = [] # estimated theta_hat for each iterate
    two_norm_differences = []

    if objective == "SURE": 

        for i in range(n_iter):

            if optimizer_str == "bfgs":

                def closure():
                    
                    optimizer.zero_grad()
                    loss = model.opt_func_SURE(Z, feature_representation, X)
                    # print(f"loss: {loss.item()}")
                    losses_SURE.append(loss.item())
                    losses_NLL.append(model.opt_func_NLL(Z, feature_representation, X).cpu().detach().numpy()) # just for fun
                    loss.backward()

                    return(loss)
                
                optimizer.step(closure)

            else:

                optimizer.zero_grad()
                loss = model.opt_func_SURE(Z, feature_representation, X)
                # print(f"loss: {loss.item()}")
                losses_SURE.append(loss.item())
                losses_NLL.append(model.opt_func_NLL(Z, feature_representation, X).cpu().detach().numpy()) # just for fun
                loss.backward()
                optimizer.step()
                
                
            score = model.compute_score(Z, feature_representation, X).cpu().detach().numpy()
            scores.append(score)
            # print(f"score: {score}")
            sigma = X[:,-1] 
            theta_hat_iterate = Z.cpu().detach().numpy() + sigma.cpu().detach().numpy()**2 * score
            # print(f"theta_hat_iterate: {theta_hat_iterate.shape}")
            theta_hats.append(theta_hat_iterate)
            # print(f"difference shape {(theta_hat_iterate - theta).shape}")

            two_norm_differences.append(np.linalg.norm(theta_hat_iterate - theta)**2)

            if np.isnan(losses_SURE[-1]): 
                break
            
            # if i > 0:
                # loss_chng = abs((losses_SURE[i]-losses_SURE[i-1])/(losses_SURE[i-1])) # Using moving average
                # if loss_chng<tol:
                    # print(i)
                    # break
    
    elif objective == "NLL":

        for i in range(n_iter):
            if optimizer_str == "bfgs":

                def closure():
                    optimizer.zero_grad()
                    loss = model.opt_func_NLL(Z, feature_representation, X)
                    # print(f"loss: {loss.item()}")
                    losses_NLL.append(loss.item())
                    losses_SURE.append(model.opt_func_SURE(Z, feature_representation, X).cpu().detach().numpy()) # just for fun
                    loss.backward()

                    return(loss)
                
                optimizer.step(closure)


            else:

                optimizer.zero_grad()
                loss = model.opt_func_NLL(Z, feature_representation, X)
                # print(f"loss: {loss.item()}")
                losses_NLL.append(loss.item())
                losses_SURE.append(model.opt_func_SURE(Z, feature_representation, X).cpu().detach().numpy()) # just for fun
                loss.backward()
                optimizer.step()
                

            score = model.compute_score(Z, feature_representation, X).cpu().detach().numpy()
            scores.append(score)
            # print(f"score: {score}")
            sigma = X[:,-1] 
            theta_hat_iterate = Z.cpu().detach().numpy() + sigma.cpu().detach().numpy()**2 * score
            # print(f"theta_hat_iterate: {theta_hat_iterate.shape}")
            theta_hats.append(theta_hat_iterate)
            # print(f"difference shape {(theta_hat_iterate - theta).shape}")

            two_norm_differences.append(np.linalg.norm(theta_hat_iterate - theta)**2)
    
            if np.isnan(losses_NLL[-1]): 
                break
            
            # if i > 0:
                # loss_chng = abs((losses_NLL[i]-losses_NLL[i-1])/(losses_NLL[i-1])) # Using moving average
                # if loss_chng<tol:
                    # print(i)
                    # break
    
    else: 

        print("error!!!")


    return(model, feature_representation, losses_SURE, losses_NLL, 
           np.array(scores), np.array(theta_hats), two_norm_differences[-1])

def train_sure_ls(X, Z, theta, objective="SURE", set_seed = None, d=2, device=simulate_data.device, 
                  optimizer_str="adam", activation_fn=tr.nn.ReLU(), 
                  hidden_sizes=(8,8), lr=1e-2): 
    '''
    Training function for SURE-LS. 
    '''
    
    if set_seed is not None:
        tr.manual_seed(set_seed)

    model = models.model_sure_ls(X, Z, d=d, hidden_sizes=hidden_sizes, activation_fn=activation_fn, device=device)
    model.to(simulate_data.device)

    feature_representation = model.feature_representation(X) 

    if set_seed is not None: 
        rn.seed(set_seed)

    # Tolerance for stopping criteria
    tol = 1e-6
    # Number of iterations
    n_iter = 2000

    # Optimizer
    if optimizer_str == "adam":
        optimizer = tr.optim.Adam(model.parameters(), lr=lr)
    else:
        print("optimizer_str typo") 
    
    losses = []
    scores = []
    theta_hats = [] # estimated theta_hat for each iterate
    two_norm_differences = []

    if objective == "SURE": 

        for i in range(n_iter):

            optimizer.zero_grad()
            loss = model.opt_func_SURE(Z, feature_representation, X)
            # print(f"loss: {loss.item()}")
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        score = model.compute_score(Z, feature_representation, X).cpu().detach().numpy()
        scores.append(score)
        # print(f"score: {score}")
        sigma = X[:,-1] 
        theta_hat_iterate = Z.cpu().detach().numpy() + sigma.cpu().detach().numpy()**2 * score
        # print(f"theta_hat_iterate: {theta_hat_iterate.shape}")
        theta_hats.append(theta_hat_iterate)
        # print(f"difference shape {(theta_hat_iterate - theta).shape}")

        two_norm_differences.append(np.linalg.norm(theta_hat_iterate - theta)**2)
    
    elif objective == "NLL": 

        for i in range(n_iter):
            
            optimizer.zero_grad()
            loss = model.opt_func_NLL(Z, feature_representation, X)
            # print(f"loss: {loss.item()}")
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        score = model.compute_score(Z, feature_representation, X).cpu().detach().numpy()
        scores.append(score)
        # print(f"score: {score}")
        sigma = X[:,-1] 
        theta_hat_iterate = Z.cpu().detach().numpy() + sigma.cpu().detach().numpy()**2 * score
        # print(f"theta_hat_iterate: {theta_hat_iterate.shape}")
        theta_hats.append(theta_hat_iterate)
        # print(f"difference shape {(theta_hat_iterate - theta).shape}")

        two_norm_differences.append(np.linalg.norm(theta_hat_iterate - theta)**2)

    return(model, feature_representation, losses, np.array(scores), np.array(theta_hats), two_norm_differences[-1]) 

def train_EBCF(X, Z, theta, K = 5, set_seed = None, d=2, device=simulate_data.device, 
               optimizer_str="adam", activation_fn=tr.nn.ReLU(), 
               hidden_sizes=(8,8), lr=1e-2): 
    '''
    Training function for EBCF from Ignatiadis and Wager, 2019. 
    '''
    
    if set_seed is not None:
        tr.manual_seed(set_seed)
        rn.seed(set_seed)
    
    # Tolerance for stopping criteria
    tol = 1e-6
    # Number of iterations
    n_iter = 2000
    # Number of observations 
    n = len(Z) 
    # Batch size 
    m = int(n/K) 

    theta_hats = np.zeros(n)  # estimated theta_hat for each iterate
    A_hats = []
    MSE = 0

    for k in range(K): 

        Z_test = Z[(k*m):((k+1)*m)] 
        X_test = X[(k*m):((k+1)*m), :] 
        theta_test = theta[(k*m):((k+1)*m)] 

        Z_train = tr.cat((Z[:(k*m)], Z[((k+1)*m):])) 
        X_train = tr.cat((X[:(k*m),:], X[((k+1)*m):,:])) 

        model = models.model_EBCF_NPreg(X_train, Z_train, d=d, hidden_sizes=hidden_sizes, activation_fn=activation_fn, device=device)
        model.to(simulate_data.device)

        feature_representation_train = model.feature_representation(X_train) 

        # Optimizer
        if optimizer_str == "adam":
            optimizer = tr.optim.Adam(model.parameters(), lr=lr)
        else:
            print("optimizer_str typo") 
        
        for i in range(n_iter):

            optimizer.zero_grad()
            loss = model.opt_func_MSE(Z_train, feature_representation_train, X_train) 
            # print(f"loss: {loss.item()}")
            # losses.append(loss.item())
            loss.backward()
            optimizer.step() 
        
        Z_test_hat = model.get_Z_hat(Z_test, X_test) 
        theta_test_hat, MSE_test, SURE_test, A_test_hat = models.theta_hat_EBCF(theta_test, Z_test, X_test, Z_test_hat) 
        theta_hats[(k*m):((k+1)*m)] = theta_test_hat 
        A_hats.append(A_test_hat) 
        MSE += MSE_test 

        print(k) 
    
    MSE = MSE/K 

    return (theta_hats, A_hats, MSE, model) 
