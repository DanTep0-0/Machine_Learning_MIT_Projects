"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    def Gaussian(x_i, mu_j, var_j):
        #print("Gaussian: " + str((1 / np.sqrt(2 * np.pi * var_j)) * np.exp((-1/2) * np.sum((x_i - mu_j) ** 2) / var_j)))
        return (1 / (np.sqrt(2 * np.pi * var_j) ** x_i.shape[0])) * np.exp((-1/2) * np.sum((x_i - mu_j) ** 2) / var_j)
    
    K = mixture.p.shape[0]
    
    probabilities = np.zeros((X.shape[0], K))
    All_points_generation_Likelihood = 0

    for i, x in enumerate(probabilities):
        Gaussians_weightet = np.array([mixture.p[k] * Gaussian(X[i], mixture.mu[k], mixture.var[k]) for k in range(K)])
        Gaussians_weightet_sum = np.sum(Gaussians_weightet)
        All_points_generation_Likelihood += np.log(Gaussians_weightet_sum)
        # print()
        # print("Point_Gaussians_Weightet: " + str(Gaussians_weightet))
        # print("Point_Cluster_Likelihoods: " + str(mixture.p))
        # print("Point_Generaition_Likelihood: " + str(Gaussians_weightet_sum))
        
        probabilities[i] = Gaussians_weightet / Gaussians_weightet_sum # Correct
        
    # print("Likelihood = " + str(All_points_generation_Likelihood))
    # print("Log_ikelihood = " + str(np.log(All_points_generation_Likelihood)))
    # print()
    #print("Post in Estep: " + str(probabilities))
    return probabilities, All_points_generation_Likelihood



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    K = post.shape[1]
    d = X.shape[1]
    #print("Post in Mstep: " + str(post))
    n_new = np.sum(post, axis=0).reshape((K, 1)) # K * 1: sum_likelihood of every cluster
    #print("N_new: " + str(n_new))
    p_new = (n_new / X.shape[0])
    #print("P_new: " + str(p_new)) # Correct <-- Debugging from here weiter
    mu_new = np.matmul(np.transpose(post), X) / n_new # K * d: Mean Vector for all clusters
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
        var_new[j] = np.sum(post[:, j] * np.sum((X - mu_new[j]) ** 2, axis=1)) / (n_new[j] * d).item() # scalar: Variance  #post[:, j] * np.sum((X - mu_new[j]) ** 2, axis=1)[:, np.newaxis] worked before
        # print("New_Var = " + str(np.sum(np.sum(post[:, j] * np.sum((X - mu_new[j]) ** 2, axis=1)) / (n_new[j] * d).item())))

    n_new = n_new.reshape((n_new.shape[0]))
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

    while(likelihood_difference is None or likelihood_difference > 1e-6 * np.abs(likelihood_previous)):
        #print("Likelihood Difference = " + str(likelihood_difference))
        mixture = mstep(X, post)
        post, likelihood_new = estep(X, mixture)
        #print("post returned: " + str(post))
        likelihood_difference = likelihood_new - likelihood_previous
        likelihood_previous = likelihood_new
        #print("Likelihood_change after this Mstep: " + str(likelihood_difference))
        if(likelihood_difference < 0): 
            print("Mistake!")
            break;
        print("Likelihood after this Mstep: " + str(likelihood_new))
        # print()

    # print("END EM. FINAL RESULT:")

    post, log_likelihood_new = estep(X, mixture)

    return mixture, post, log_likelihood_new
