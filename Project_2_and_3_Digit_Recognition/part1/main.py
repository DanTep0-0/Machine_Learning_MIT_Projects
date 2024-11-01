import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../mnist_project_2")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *
import sklearn.svm as svm


#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
#plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# TODO: first fill out functions in linear_regression.py, otherwise the functions below will not work


def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    #print(train_x_bias.shape)
    #print(train_y)
    #print(train_y.shape)
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
#print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=100))


#######################################################################
# 3. Support Vector Machine
#######################################################################

# TODO: first fill out functions in svm.py, or the functions below will not work

def run_svm_one_vs_rest_on_MNIST(C=0.1):
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x, C)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    train_error = compute_test_error_svm(train_y, one_vs_rest_svm(train_x, train_y, train_x))
    return test_error, 


def test_different_C(range = 6, divider = 1000):
    test_train_errors = np.zeros([range, 2])

    for i, values in enumerate(test_train_errors):
        errors = run_svm_one_vs_rest_on_MNIST((i + 1)/divider)
        values[0] = errors[0]
        values[1] = errors[1]


    for i, values in enumerate(test_train_errors):
        print('SVM one vs. rest, C = ' + str((i+1)/divider) + ' test_error:', values[0])
        print('SVM one vs. rest, C = ' + str((i+1)/divider) + ' train_error:', values[1])


#print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST()[0])
#print('SVM one vs. rest train_error:', run_svm_one_vs_rest_on_MNIST()[1])

# Test different values of C:
#test_different_C(range=6, divider=1000)



def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


#print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

# TODO: first fill out functions in softmax.py, or run_softmax_on_MNIST will not work


def run_softmax_on_MNIST(temp_parameter=1, bias=False):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    print('in run_softmax_on_MNIST()')
    train_x, train_y, test_x, test_y = get_MNIST_data()
    if bias:
        train_x = augment_feature_vector(train_x)
    print("X.shape = " + str(train_x.shape))
    print("Y.shape = " + str(train_y.shape))
    print("X_Test.shape = " + str(test_x.shape))
    print("Y_Test.shape = " + str(test_y.shape))
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    
    #Normal Test Error:
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter, bias=bias)

    #Module 3 Test Error:
    #test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    # TODO: add your code here for the "Using the Current Model" question in tab 4.
    #      and print the test_error_mod3
    return test_error


#print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1, bias = True))

# TODO: Find the error rate for temp_parameter = [.5, 1.0, 2.0]
#      Remember to return the tempParameter to 1, and re-run run_softmax_on_MNIST

#######################################################################
# 6. Changing Labels
#######################################################################



def run_softmax_on_MNIST_mod3(temp_parameter=1, bias=True):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    # YOUR CODE HERE

    train_x, train_y, test_x, test_y = get_MNIST_data()

    if bias:
        train_x = augment_feature_vector(train_x)

    train_y, test_y = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=3, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)

    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    write_pickle_data(theta, "./theta.pkl.gz")
    return test_error


#print('softmax mod3 test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1, bias = True))

# TODO: Run run_softmax_on_MNIST_mod3(), report the error rate


#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.


# n_components = 400
# bias = True
# feature_means = np.mean(train_x, axis=0)

# pcs = principal_components(train_x - feature_means)
# train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
# test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# if bias:
#     train_pca = augment_feature_vector(train_pca)
#     test_pca = augment_feature_vector(test_pca)

# print(train_pca.shape)
# print(test_pca.shape)
# # train_pca (and test_pca) is a representation of our training (and test) data
# # after projecting each example onto the first 18 principal components.


# # TODO: Train your softmax regression model using (train_pca, train_y)
# #       and evaluate its accuracy on (test_pca, test_y).

# Temperature 0.5 is optimal
# theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter=0.5, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=2000)
# test_error = compute_test_error(test_pca, test_y, theta, temp_parameter=1, bias=False)
# print("Test Error of PCA Softmax = " + str(test_error))


# TODO: Use the plot_PC function in features.py to produce scatterplot
#       of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
#plot_PC(train_x[range(100)], pcs, train_y[range(100)], feature_means)


# TODO: Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.

# firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x)
# plot_images(firstimage_reconstructed)
# plot_images(train_x[0, ])

# secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x)
# plot_images(secondimage_reconstructed)
# plot_images(train_x[1, ])


## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set

# n_components = 10
# bias = True
# feature_means = np.mean(train_x, axis=0)

# pcs = principal_components(train_x - feature_means)
# train_pca_n = project_onto_PC(train_x, pcs, n_components, feature_means)
# test_pca_n = project_onto_PC(test_x, pcs, n_components, feature_means)

# # if bias:
# #     train_pca = augment_feature_vector(train_pca)
# #     test_pca = augment_feature_vector(test_pca)


# # TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

# train_cube = cubic_features(train_pca_n)
# test_cube = cubic_features(test_pca_n)
# # train_cube (and test_cube) is a representation of our training (and test) data
# # after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# # TODO: Train your softmax regression model using (train_cube, train_y)
# #       and evaluate its accuracy on (test_cube, test_y).

# # Temperature = 4 - optimal for Cubic features
# theta, cost_function_history = softmax_regression(train_cube, train_y, temp_parameter=1, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
# test_error = compute_test_error(test_cube, test_y, theta, temp_parameter=1, bias=False)
# print("Test Error of Cubic Softmax = " + str(test_error))


# Cubic Softmax wia Scikit-learn:

# n_components = 10
# feature_means = np.mean(train_x, axis=0)

# pcs = principal_components(train_x - feature_means)
# train_pca_n = project_onto_PC(train_x, pcs, n_components, feature_means)
# test_pca_n = project_onto_PC(test_x, pcs, n_components, feature_means)

# #plot_PC(train_x[range(100)], pcs, train_y[range(100)], feature_means)

# # Polynomial svm:

# cubic_polynomial_svm_scklearn = svm.SVC(kernel='poly', degree=3, random_state=0, C=1)
# polynomial_classifier = cubic_polynomial_svm_scklearn.fit(train_pca_n, train_y)
# polynomial_predictions = polynomial_classifier.predict(test_pca_n)
# print("Scklearn cubic polynomial SVM error = " + str(1 - np.mean(polynomial_predictions == test_y)))

# # Rbf svm:

# rbf_svm_scklearn = svm.SVC(random_state=0, kernel='rbf', C=1)
# rbf_classifier = rbf_svm_scklearn.fit(train_pca_n, train_y)
# rbf_predictions = rbf_classifier.predict(test_pca_n)
# print("Scklearn rbf SVM error = " + str(1 - np.mean(rbf_predictions == test_y)))


# Kernel Softmax Functions:

def run_kernel_softmax_on_MNIST(type='poly3'):
    """
    Trains Softmax regression on kernel features.

    See run_softmax_on_MNIST for more info.
    """
    # SETUP

    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x = train_x[:4000]
    train_y = train_y[:4000]
    test_x = test_x[:1000]
    test_y = test_y[:1000]
    gamma = 0.08
    num_iterations = 100
    temp_parameter = 200

    n_components = 784
    feature_means = np.mean(train_x, axis=0)

    pcs = principal_components(train_x - feature_means)
    train_x = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_x = project_onto_PC(test_x, pcs, n_components, feature_means)


    train_kernel_matrix = None
    test_kernel_matrix = None
    
    if type == "poly3":
        train_kernel_matrix = polynomial_kernel(train_x, train_x, 1, 3)
    else:
        train_kernel_matrix = rbf_kernel(train_x, train_x, gamma)

    #print("train kernel matrix.shape = " + str(train_kernel_matrix.shape))

    # ALGORITHM 

    a_indices, cost_function_history = kernelised_softmax_regression(train_x, train_y, train_kernel_matrix, temp_parameter=temp_parameter, k=10, num_iterations=num_iterations, type=type, gamma=gamma)
    #plot_cost_function_over_time(cost_function_history)

    if type == "poly3":
        test_kernel_matrix = polynomial_kernel(train_x, test_x, 1, 3)
    else:
        test_kernel_matrix = rbf_kernel(train_x, test_x, gamma)

    # TEST ERROR

    test_error = kernelised_compute_test_error(test_kernel_matrix, a_indices, test_y, temp_parameter)

    #print(np.sum(a_indices, axis=1))
    #write_pickle_data(theta, "./theta.pkl.gz")
    return test_error


#print('Poly3 kernel softmax test_error = ', run_kernel_softmax_on_MNIST(type='poly3'))
#print('Poly3 kernel softmax test_error = ', run_kernel_softmax_on_MNIST(type='rbf'))