"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import math


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
    def Gaussian_log(x_i, mu_j, var_j):
        C_x = x_i != 0
        #print("Gaussian: " + str((1 / (np.sqrt(2 * np.pi * var_j) ** x_i.shape[0])) * np.exp((-1/2) * np.sum((x_i[C_x]- mu_j[C_x]) ** 2) / var_j)))
        # if (np.sqrt(2 * np.pi * var_j)) == 0:
        #     print(var_j)
        #     breakpoint()

        # var_j is nan!!!
        Gaussian_log = x_i[C_x].shape[0] * np.log(1 / (np.sqrt(2 * np.pi * var_j))) + ((-1/2) * np.sum((x_i[C_x]- mu_j[C_x]) ** 2) / var_j)
        #print("Gaussian = " + str(Gaussian_log) + ", var = " + str(var_j))
        #print("(1) = " + str(x_i.shape[0] * np.log(1 / (np.sqrt(2 * np.pi * var_j)))) + ", (2) = " + str(((-1/2) * np.sum((x_i[C_x]- mu_j[C_x]) ** 2) / var_j)))
        # if Gaussian_log is not float:
        #     print("Gaussian_log = " + str(Gaussian_log))
        #     print(np.log(1 / (np.sqrt(2 * np.pi * var_j))))
        #     print(((-1/2) * np.sum((x_i[C_x]- mu_j[C_x]) ** 2) / var_j))
        #     breakpoint()
        return Gaussian_log
    
    K = mixture.p.shape[0]
    
    probabilities = np.zeros((X.shape[0], K))
    All_points_generation_Likelihood = 0

    for i, x in enumerate(probabilities):
        #Calculating everything in log()
        #print("mixture.var = " + str(mixture.var))
        Gaussians_weightet_log = np.array([np.log(mixture.p[k] + 1e-16) + Gaussian_log(X[i], mixture.mu[k], mixture.var[k]) for k in range(K)])
        Gaussians_weightet_log_sum = logsumexp(Gaussians_weightet_log) # = log(sum(exp(Gaussian_weightet[k]))) over all of k
        All_points_generation_Likelihood += Gaussians_weightet_log_sum
        if Gaussians_weightet_log_sum.shape is None:
            print("Gaussians_weightet_log_sum is Nan, Gaussians_weightet_log:")
            print(Gaussians_weightet_log)
            for k in range(K):
                print("Gaussial_log for k = " + str(k) +": " + str(Gaussian_log(X[i], mixture.mu[k], mixture.var[k])))

            breakpoint()
        #print("All_points_generation_Likelihood = " + str(All_points_generation_Likelihood))
        # print()
        # print("Point_Gaussians_Weightet: " + str(Gaussians_weightet))
        # print("Point_Cluster_Likelihoods: " + str(mixture.p))
        # print("Point_Generaition_Likelihood: " + str(Gaussians_weightet_sum))
        
        probabilities[i] = np.exp(Gaussians_weightet_log - Gaussians_weightet_log_sum) # Correct
        # if np.sum(probabilities[i] == 0) > 0:
        #     print("Gaussians_weightet_log = " + str(Gaussians_weightet_log))
        #     print("Gaussians_weightet_log_sum = " + str(Gaussians_weightet_log_sum))
        #     breakpoint()
        
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
    probabilities_each_class_for_each_point = post / np.sum(post, axis=1).reshape(num_points, 1) # n * K
    #print("Zeros in post % = " + str(np.sum(post == 0) / (post.shape[0] * post.shape[1])))

    mu_new = np.zeros((K, d))
    n_new = np.zeros((K, d))
    number_points_for_each_dimension = np.zeros((K, d))
    for k in range(K):
        n_new[k] = np.sum(post[:, k][:, np.newaxis] * C_X, axis=0) # vector length d: n for each dimension   
        number_points_for_each_dimension[k] = np.sum(probabilities_each_class_for_each_point[:, k].reshape((num_points, 1)) * C_X, axis=0) # d values vector for class k
        # if np.sum(n_k_new == 0) > 0:
        #     print(np.sum(n_k_new == 0))
        #     print("P[k]: ")
        #     print(post[:, k])
        #     print(np.mean(post[:, k]))
        #     # print("P[k-1]: ")
        #     # print(post[:, k-1])
        #     breakpoint()
    
    for k in range(K):
        n_k_new = n_new[k]
        # mu_new[k] = (np.sum(post[:, k][:, np.newaxis] * X, axis=0) / n_k_new)
        # print("sum of X in mu assignment: " + str(np.sum(post[:, k][:, np.newaxis] * X, axis=0)))
        # print("n_k_new in mu assignment: " + str(n_k_new))

        mu_new[k][number_points_for_each_dimension[k] >= 1] = (np.sum(post[:, k][:, np.newaxis] * X, axis=0)[number_points_for_each_dimension[k] >= 1] / n_k_new[number_points_for_each_dimension[k] >= 1]) # vector length d: mu for each dimension
        mu_new[k][number_points_for_each_dimension[k] < 1] = mixture.mu[k][number_points_for_each_dimension[k] < 1]
        n_new[k][number_points_for_each_dimension[k] >= 1] = n_k_new[number_points_for_each_dimension[k] >= 1]

       # if np.sum(number_points_for_each_dimension[k] < 1) > 0:
        #     print(number_points_for_each_dimension[k-1])
        #     # print(np.sum(number_points_for_each_dimension[k] < 1))
        #     # print(number_points_for_each_dimension[k] < 1)
        #     # print(probabilities_each_class_for_each_point[:, k])
        #     # print(post[:, k])
        #     # print(C_X)
        #     # print(post[:, k][:, np.newaxis] * C_X)
        #     # print(np.sum(post[:, k][:, np.newaxis] * X, axis=0))
        #     breakpoint()


    #n_new = np.sum(post, axis=0).reshape((K, 1)) # K * 1: sum_likelihood of every cluster
    #print("N_new: " + str(n_new))
    number_point_of_every_cluster = np.sum(post / np.sum(post, axis=1).reshape((num_points, 1)), axis=0) # K values
    p_new = number_point_of_every_cluster.reshape(K, 1) / num_points
    #print("P_new: " + str(p_new)) # Correct <-- Debugging from here weiter
    #mu_new = np.matmul(np.transpose(post), X) / n_new # K * d: Mean Vector for all clusters
    
    
    # TODO:
    # The problem is that after first mstep var is assigned to nan
    
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
        # print(((X - mu_new[j]) ** 2))
        # print(C_X)
        # print(((X - mu_new[j]) ** 2) * C_X)
        # print("mu_j.shape = " + str(mu_new[j].shape))
        # print("X.shape = " + str(X.shape))
        # print("Norm = " + str(post[:, j] * np.sum(C_X, axis=1)))
        var_new_j = np.sum(post[:, j] * np.sum(np.multiply(((X - mu_new[j]) ** 2), (C_X + 0)), axis=1)) / np.sum(post[:, j] * np.sum(C_X, axis=1)) # scalar: Variance  #post[:, j] * np.sum((X - mu_new[j]) ** 2, axis=1)[:, np.newaxis] worked before
        if var_new_j < min_variance: var_new_j = min_variance
        if math.isnan(var_new_j):
            print("Var is NaN")
            #print(np.sum(post[:, j] * np.sum(np.multiply(((X - mu_new[j]) ** 2), C_X), axis=1)))
            #print(np.sum(post[:, j] * np.sum(C_X, axis=1)))
            print("post: " + str(post[:, j]))
            #print("varianses for all points: " + str(np.sum(np.multiply(((X - mu_new[j]) ** 2), C_X), axis=1)))
            print("varianses matrix: " + str(np.multiply(((X - mu_new[j]) ** 2), C_X)))
            print("C_X: " + str(C_X + 0))
            print("varianses of single features" + str((X - mu_new[j]) ** 2))
            print("mu: " + str(mu_new[j]))
            breakpoint()

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

    step = 1
    while(likelihood_difference is None or likelihood_difference > 1e-6 * np.abs(likelihood_previous)):
        #print("Likelihood Difference = " + str(likelihood_difference))
        mixture = mstep(X, post, mixture)
        post, likelihood_new = estep(X, mixture)
        #print("post returned: " + str(post))
        likelihood_difference = likelihood_new - likelihood_previous
        likelihood_previous = likelihood_new
        #print("Likelihood_change after this Mstep: " + str(likelihood_difference))
        print("Likelihood after " + str(step) + "th Mstep: " + str(likelihood_new))
        step += 1
        if(likelihood_difference < 0): 
            print("Mistake! Likelohood decreases")
            break;
        # print()

    # print("END EM. FINAL RESULT:")

    post, log_likelihood_new = estep(X, mixture)

    return mixture, post, log_likelihood_new


def fill_matrix(X: np.ndarray, mixture: GaussianMixture, way='soft') -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    post, log_likelihood = estep(X, mixture)
    
    filled_X = np.copy(X)
    # Pick the most likely cluster and take the 
    if way == 'hard':
        clusters = np.argmax(post, axis=1)
        for i, x in enumerate(filled_X):
            x[x == 0] = mixture.mu[clusters[i]][x == 0]

    #Pick the most likely cluster, take the mu and round it to ints
    if way == 'int':
        # Hard
        clusters = np.argmax(post, axis=1)
        for i, x in enumerate(filled_X):
            x[x == 0] = np.rint(mixture.mu[clusters[i]][x == 0]).astype(int)

        #Soft 
        # for i, x in enumerate(filled_X):
        #     x[x == 0] = np.rint(np.dot(post[i], mixture.mu)[x == 0]).astype(int)

        filled_X[filled_X == 0] = 1
        filled_X[filled_X == 6] = 5

    # Softly calculate the centroid
    if way == 'soft':
        for i, x in enumerate(filled_X):
            x[x == 0] = np.dot(post[i], mixture.mu)[x == 0]

    return filled_X
