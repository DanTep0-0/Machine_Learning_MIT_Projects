"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    def Gaussian(x_i, mu_j, var_j):
        C_x = x_i != 0
        #print("Gaussian: " + str((1 / np.sqrt(2 * np.pi * var_j)) * np.exp((-1/2) * np.sum((x_i - mu_j) ** 2) / var_j)))
        return (1 / (np.sqrt(2 * np.pi * var_j) ** x_i.shape[0])) * np.exp((-1/2) * np.sum((x_i[C_x]- mu_j[C_x]) ** 2) / var_j)
    
    K = mixture.p.shape[0]
    
    probabilities = np.zeros((X.shape[0], K))
    All_points_generation_Likelihood = 0

    for i, x in enumerate(probabilities):
        #Calculating everything in log()
        Gaussians_weightet_log = np.array([np.log(mixture.p[k] + 1e-16) + np.log(Gaussian(X[i], mixture.mu[k], mixture.var[k])) for k in range(K)])
        Gaussians_weightet_log_sum = logsumexp(Gaussians_weightet_log) # = log(sum(exp(Gaussian_weightet[k]))) over all of k
        All_points_generation_Likelihood += Gaussians_weightet_log_sum
        # print()
        # print("Point_Gaussians_Weightet: " + str(Gaussians_weightet))
        # print("Point_Cluster_Likelihoods: " + str(mixture.p))
        # print("Point_Generaition_Likelihood: " + str(Gaussians_weightet_sum))
        
        probabilities[i] = np.exp(Gaussians_weightet_log - Gaussians_weightet_log_sum) # Correct
        
    # print("Likelihood = " + str(All_points_generation_Likelihood))
    # print("Log_ikelihood = " + str(np.log(All_points_generation_Likelihood)))
    # print()
    #print("Post in Estep: " + str(probabilities))
    return probabilities, All_points_generation_Likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    C_X = X != 0
    K = post.shape[1]
    d = X.shape[1]
    num_points = X.shape[0]
    #print("Post in Mstep: " + str(post))

    mu_new = np.zeros((K, d))
    n_new = np.zeros((K, d))
    for k in range(K):
        n_k_new = np.sum(post[:, k][:, np.newaxis] * C_X, axis=0) # vector length d: n for each dimension    
        mu_new[k][n_k_new >= 1] = (np.sum(post[:, k][:, np.newaxis] * X, axis=0) / n_k_new) # vector length d: mu for each dimension
        mu_new[k][n_k_new < 1] = mixture.mu[k][n_k_new < 1]
        n_new[k] = n_k_new


    #n_new = np.sum(post, axis=0).reshape((K, 1)) # K * 1: sum_likelihood of every cluster
    #print("N_new: " + str(n_new))
    number_point_of_every_cluster = np.sum(post, axis=0).reshape((K, 1)) # K * 1
    p_new = number_point_of_every_cluster / num_points
    #print("P_new: " + str(p_new)) # Correct <-- Debugging from here weiter
    #mu_new = np.matmul(np.transpose(post), X) / n_new # K * d: Mean Vector for all clusters
    var_new = np.ones(K)
    for j in range(K):
        # print("X: " + str(X))
        # print("Mu_new_j: " + str(mu_new[j]))
        # print("X - mu = " + str((X - mu_new[j])))
        # print("(X - mu)^2 = " + str((X - mu_new[j]) ** 2))
        # print("modules[j]^2: " + str(np.sum((X - mu_new[j]) ** 2, axis=1)))
        # print("post[j]: " + str(post[:, j]))
        # print("element-wise multiplication p * modules: " + str(post[:, j] * np.sum((X - mu_new[j]) ** 2, axis=1)))
        # print("N_new_j: " + str(n_new[j]))
        # print("n[j] * d: " + str((n_new[j] * d).item()))
        # For var we use n_new as a normalisation because we count only parameters, that != 0 and with respect to p(j|u) => norm = sum of d_valid for every point, = np.sum(n_new[j]) = np.sum(C_X * post[j])
        print(((X - mu_new[j]) ** 2).shape)
        print(C_X.shape)
        print(np.dot((X - mu_new[j]) ** 2, C_X).shape)
        var_new_j = np.sum(post[:, j] * np.sum(((X - mu_new[j]) ** 2)[C_X], axis=1)) / np.sum(n_new[j]).item() # scalar: Variance  #post[:, j] * np.sum((X - mu_new[j]) ** 2, axis=1)[:, np.newaxis] worked before
        if var_new_j < min_variance: var_new_j = min_variance
        var_new[j] = var_new_j 
        # print("New_Var = " + str(np.sum(np.sum(post[:, j] * np.sum((X - mu_new[j]) ** 2, axis=1)) / (n_new[j] * d).item())))

    #print("C_X = " + str(np.sum(C_X * post[j])))
    #print("np.sum(n_new[j]) = " + str(np.sum(n_new[j])))

    #n_new = n_new.reshape((n_new.shape[0]))
    p_new = p_new.reshape((p_new.shape[0]))
    new_Mixture = GaussianMixture(mu=mu_new, var=var_new, p=p_new)

    return new_Mixture


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    print("STARTING EM: ")
    post, likelihood_previous = estep(X, mixture)
    #print("post returned: " + str(post))
    print("likelihood on start = " + str(likelihood_previous))
    likelihood_difference = None

    while(likelihood_difference is None or likelihood_difference > 1e-6):
        #print("Likelihood Difference = " + str(likelihood_difference))
        mixture = mstep(X, post, mixture)
        post, likelihood_new = estep(X, mixture)
        #print("post returned: " + str(post))
        likelihood_difference = likelihood_new - likelihood_previous
        likelihood_previous = likelihood_new
        #print("Likelihood_change after this Mstep: " + str(likelihood_difference))
        if(likelihood_difference < 0): 
            print("Mistake! Likelohood decreases")
            break;
        print("Likelihood after this Mstep: " + str(likelihood_new))
        # print()

    # print("END EM. FINAL RESULT:")

    post, log_likelihood_new = estep(X, mixture)

    return mixture, post, log_likelihood_new


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
