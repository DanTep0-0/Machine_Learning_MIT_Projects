import numpy as np
import time


### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE

    print('in polynomial kernel')
    print("X.shape = " + str(X.shape))
    print("Y.shape = " + str(Y.shape))

    time_start = time.time()
    matrix = (np.matmul(X, np.transpose(Y)) + c) ** p
    print("time for calculating matrix = " + str(time.time() - time_start))

    return matrix


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE

    print('in rbf kernel')
    kernel_matrix = np.zeros([X.shape[0], Y.shape[0]])
    print(kernel_matrix.shape)

    for i in range(X.shape[0]):
        kernel_matrix[i] = np.exp((-1) * gamma * np.sum((Y-X[i]) ** 2, axis=1))

    return kernel_matrix


def kernelised_compute_powers(a_indices, idx_x, kernel_matrix):
    return np.matmul(a_indices, kernel_matrix[:, idx_x])


def kernelised_compute_probabilities(powers, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE 
    powers = powers / temp_parameter
    C = np.max(powers, axis=0)
    exp_results = np.exp(powers - C)

    return exp_results / np.sum(exp_results, axis=0)


def kernelised_compute_cost_function(kernel_matrix, Y, a_indices, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE

    powers = np.matmul(a_indices, kernel_matrix)
    probabilities = kernelised_compute_probabilities(powers, temp_parameter)
    #predictions = np.where(probabilities == np.max(powers, axis=0).reshape(powers.shape[1], 1))
    #correct_tetas = (probabilities == np.max(probabilities, axis=0))

    correct_tetas = np.full([a_indices.shape[0], a_indices.shape[1]], False)

    for i in range(a_indices.shape[0]):
        try:
            correct_tetas[i, :][(Y == i)] = True
        except: continue

    #print(correct_tetas[:, :20])
    #print(probabilities[correct_tetas])
        
    # Calculating just loss. without Regularisation. because it's Kernels => algorytm without derivatives
    softmax_loss = np.mean((-1) * np.log(probabilities[correct_tetas]))
    #print("softmax_loss = " + str(softmax_loss))
    return softmax_loss


def kernelised_softmax_iteration(X, Y, kernel_matrix, idx_x, a_indices, type='poly3', gamma=1):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE

    powers_for_new_x = kernelised_compute_powers(a_indices, idx_x, kernel_matrix)
    #print(np.where(powers_for_new_x == np.max(powers_for_new_x)))
    most_likely_number_power = np.max(powers_for_new_x)
    if most_likely_number_power > 0 and  np.where(powers_for_new_x == most_likely_number_power)[0] == Y[idx_x]:
        #print("a is correct")
        return a_indices
    else:
        #print("updating a")
        a_indices[Y[idx_x]][idx_x] +=1
        return a_indices


def kernelised_softmax_regression(X, Y, kernel_matrix, temp_parameter, k, num_iterations, type='poly3', gamma=1):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    print('in kernelised_softmax_regression()')
    #X = coo_matrix(augment_feature_vector(X))
    #Y = coo_matrix(Y)
    a_indices = np.zeros([k, X.shape[0]])
    #print("a_indices.shape = " + str(a_indices.shape))

    cost_function_progression = []
    #print('Number of iterations = ' + str(num_iterations))

    time_start_main = time.time()
    for i in range(num_iterations):
        #cost_function_progression.append(kernelised_compute_cost_function(kernel_matrix, Y, a_indices, temp_parameter))
        time_start = time.time()

        for n in range(X.shape[0]):
            a_indices = kernelised_softmax_iteration(X, Y, kernel_matrix, n, a_indices, type=type, gamma = 1)
        
        print("Time for " + str(i) + " iteration = " + str(time.time() - time_start))

    print("Time for all iterations = " + str(time.time() - time_start_main))
    return a_indices, cost_function_progression


def kernelised_compute_test_error(test_kernel_matrix, a_indices, test_Y, temp_parameter):
    print('in kernelised compute test error()')

    test_powers = np.matmul(a_indices, test_kernel_matrix)
    probabilities = kernelised_compute_probabilities(test_powers, temp_parameter)
    #predictions = np.where(probabilities == np.max(powers, axis=0).reshape(powers.shape[1], 1))
    predictions = probabilities == np.max(probabilities, axis=0)

    answers = np.zeros(predictions.shape[1])
    for i in range(predictions.shape[0]):
        answers[predictions[i] == 1] = i

    #.reshape(test_powers.shape[1], 1)
    #print(np.max(probabilities, axis=0))
    #print(np.max(probabilities, axis=0).shape)
    #print(probabilities == np.max(probabilities, axis=0))
    

    #print(answers)
    #print(predictions[0].shape)
    #print(test_Y)
        
    # Calculating just loss. without Regularisation. because it's Kernels => algorytm without derivatives
    return 1 - np.mean(answers == test_Y)