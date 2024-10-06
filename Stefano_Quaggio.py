'''
Template for Assignment 1
'''

import numpy as np # Is it version 2.1 the one you are running?
import matplotlib.pyplot as plt 
import torch # Is it version 2.4 the one you are running?
import torch.nn as nn
import torch.optim as optim

def plot_polynomial(coeffs, z_range, color='b'):
    pass
    

def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):
    X = None
    y = None
    return X, y

def visualize_data(X, y, coeffs, z_range, title=""):
    pass

def plot_polynomial(coeffs, z_range, color='b') :
    z_min, z_max = z_range

    # Generate z values in the range [z_min, z_max]
    z = np.linspace(z_min, z_max, 100)

    # Compute the polynomial p(z) = w_0 + w_1*z + w_2*z^2 + ... + w_n*z^n using the poly1d function
    p = np.poly1d(coeffs)

    # Plot the polynomial
    plt.plot(z, p(z), color)
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.title('Polynomial p(z) = ' + ' + '.join([f'{c}z^{i}' for i, c in enumerate(coeffs)]))
    plt.show()

def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):
    torch.manual_seed(seed)
    z_min, z_max = z_range

    # Generate z values in the range [z_min, z_max]
    z = torch.rand(sample_size)
    z = z * (z_max - z_min) + z_min
    
    # Initiliaze y_hat
    y_hat = torch.zeros(sample_size)

    # Compute the polynomial y_hat = w_0 + w_1*x + w_2*x^2 + ... + w_n*x^n
    for i, coeff in enumerate(coeffs):
        y_hat += coeff * z**i

    # Add Gaussian noise to y_hat
    y = y_hat + torch.normal(torch.zeros(sample_size), sigma*torch.ones(sample_size))

    return z, y

def visualize_data(X, y, coeffs, z_range, title=""):
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max, 100)
    p = np.poly1d(coeffs)
    plt.plot(z, p(z), 'b', label='True polynomial')
    plt.scatter(X, y, c='r', label='Noisy data')
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_polynomial([1, -1, 5, -0.1, 1/30], [-4, 4], 'r')

    z_range = [-2, 2]
    coeffs = [1, -1, 5, -0.1, 1/30]
    n_samples = 500
    sigma = 0.5
    
    X_train, y_train = create_dataset(coeffs, z_range, n_samples, sigma, 0)
    X_val, y_val= create_dataset(coeffs, z_range, n_samples, sigma, 1)

    visualize_data(X_train, y_train, coeffs, z_range, 'Training data')
    visualize_data(X_val, y_val, coeffs, z_range, 'Validation data')


    #assert np.version.version=="2.1"
    
