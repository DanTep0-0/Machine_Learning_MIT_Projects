import sys
sys.path.append("../mnist_project_2")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import  coo_matrix
import math
import time


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
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

    # print('in compute_probabilities()')

    # H = np.zeros([theta.shape[0], X.shape[0]])
    
    # for i in range(H.shape[1]):
    #     sum_of_all_e = 0
    #     probability_e_vector = np.zeros(theta.shape[0])
            
    #     c = np.max(np.matmul(theta, X[i]/temp_parameter))

    #     for j, current_theta in enumerate(theta):
    #         e_for_current_theta = math.exp(np.matmul(current_theta, X[i]/temp_parameter) - c)
    #         probability_e_vector[j] = e_for_current_theta
    #         sum_of_all_e += e_for_current_theta
        
    #     H[:, i] = (1/ sum_of_all_e) * probability_e_vector

    # return H
    # print('X: ')
    # print(X)
    # print("Theta: ")
    # print(theta)
    # print("np.inner(X, theta): ")
    # print(np.inner(X, theta))
    # print("Exp_results: ")
    powers = np.inner(X, theta) / temp_parameter
    C = np.max(powers, axis=1).reshape(powers.shape[0], 1)
    exp_results = np.exp(powers - C)
    # print(exp_results)
    # print("Sum of exp: ")
    # print(np.sum(exp_results, axis=1).reshape((exp_results.shape[0], 1)))
    # print()
    return np.transpose(exp_results / np.sum(exp_results, axis=1).reshape((exp_results.shape[0], 1)))


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
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

    def softmax_loss_for_correct_thetas(theta, Y, X, temp):

        correct_tetas = np.full([X.shape[0], theta.shape[0]], False)

        for i in range(theta.shape[0]):
            try:
                correct_tetas[:, i][(Y == i)] = True
            except: continue

        exp_results = np.exp(np.inner(X, theta) / temp)
        return (-1) * np.log(exp_results[correct_tetas] / np.sum(exp_results, axis=1).reshape((1, exp_results.shape[0])))


    sum_of_all_thetas = np.sum(theta ** 2)
    sum_of_losses = np.sum(softmax_loss_for_correct_thetas(theta, Y, X, temp_parameter))
    
    return sum_of_losses/X.shape[0] + (lambda_factor/2 * sum_of_all_thetas)





def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
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

    # def Gradient_for_theta_m(temp, X, theta, m, Y, lambda_param):
    #     print('in Gradient_for_theta_m()')
 
    #     P = compute_probabilities(X, theta, temp)
    #     loss_derivative_parameter = (1 / (temp * X.shape[0])) * np.sum([np.sum(X * P[:, m].reshape(X.shape[0], 1), axis=0), (-1) * np.sum(X[Y == m], axis=0)], axis=0)
    #     #sum_parameter_m = np.sum([ X[i] * (1 - p_j(m, X[i], theta, temp)) if (m == Y[i]) else X[i] * ((-1) * p_j(m, X[i], theta, temp)) for i in range(X.shape[0])], axis=0)
    #     return loss_derivative_parameter + lambda_param * theta[m]
 
    P = compute_probabilities(X, theta, temp_parameter)
    correct_tetas = np.full([theta.shape[0], X.shape[0]], 0)

    for i in range(theta.shape[0]):
            correct_tetas[i, :][(Y == i)] = 1

    loss_derivative_parameter = (1 / (temp_parameter * X.shape[0])) * np.matmul(P - correct_tetas, X)
    return theta - (alpha * (loss_derivative_parameter + lambda_factor * theta))


def run_gradient_descent_iteration_coo_matrixes(X, Y, theta, alpha, lambda_factor, temp_parameter):
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

    itemp=1./temp_parameter
    num_examples = X.shape[0]
    num_labels = theta.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    # M[i][j] = 1 if y^(j) = i and 0 otherwise.
    M = coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
    non_regularized_gradient = np.dot(M-probabilities, X)
    non_regularized_gradient *= -itemp/num_examples
    return theta - alpha * (non_regularized_gradient + lambda_factor * theta)


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE

    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3

    return (train_y_mod3, test_y_mod3)

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    # For the assignment in Curse, X must be inputed of shape d-1. It's without bias.
    print('in compute test error mod 3()')
    assigned_labels = get_classification(X, theta, temp_parameter, bias=True) % 3
    assigned_labels, Y = update_y(assigned_labels, Y)

    return 1 - np.mean(assigned_labels == Y)


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
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
    print('in softmax_regression()')
    #X = coo_matrix(augment_feature_vector(X))
    #Y = coo_matrix(Y)
    theta = np.zeros([k, X.shape[1]])
    print("theta.shape = " + str(theta.shape))
    cost_function_progression = []
    print('Number of iterations = ' + str(num_iterations))

    time_start_main = time.time()
    for i in range(num_iterations):
        #print('in loop, i = ' + str(i))
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        time_start = time.time()
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
        print("Time for " + str(i) + " iteration = " + str(time.time() - time_start))

    print("Time for all iterations = " + str(time.time() - time_start_main))
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter, bias=True):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    if bias:
        X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    print('in plot_cost_function_over_time()')
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter, bias=True):
    print('in compute test error()')
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter, bias)
    return 1 - np.mean(assigned_labels == Y)

