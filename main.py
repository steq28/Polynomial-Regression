from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def plot_polynomial(coeffs: List[float], z_range: Tuple[float, float], color: str = 'b'):
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max)

    p = np.poly1d(coeffs)

    plt.plot(z, p(z), color)
    plt.xlabel('z')
    plt.ylabel('p(z)')

    terms = []
    for i, c in enumerate(coeffs):
        term = f'{round(c, 3)}' if i == 4 else f'{round(c, 3)}z^{4 - i}'
        terms.append(term)

    polynomial = ' + '.join(terms)

    plt.title(r'$p(z) = ' + polynomial + '$')
    plt.show()

def create_dataset(coeffs: List[float], z_range: Tuple[float, float], sample_size: int, sigma: float, seed: int = 42):
    torch.manual_seed(seed)
    z_min, z_max = z_range

    z = torch.rand(sample_size)
    z = z * (z_max - z_min) + z_min

    p = np.poly1d(coeffs)
    
    y_hat = torch.tensor(p(z))

    y = y_hat + torch.normal(torch.zeros(sample_size), sigma * torch.ones(sample_size))

    X = torch.stack([z ** power for power in range(5)], dim=1)
    
    return X, y

def visualize_data(X: torch.Tensor, y: torch.Tensor, coeffs: List[float], z_range: Tuple[float, float], title: str = ""):
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max)
    p = np.poly1d(coeffs)

    plt.plot(z, p(z), 'b', label='True polynomial')
    plt.scatter(X[:, 1], y, c='r', label='Noisy data')
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.title(title)
    plt.legend()
    plt.show()

def create_train_model(X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, n_steps: int = 500, input_size: int = 5, bonus_question: bool = False):
    train_loss_vals = []
    val_loss_vals = []
    estimated_w = []

    model = nn.Linear(input_size, 1, bonus_question)
    loss_fn = nn.MSELoss()
    learning_rate = 0.027 if bonus_question else 0.03

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    if bonus_question:
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for step in range(n_steps):
        model.train()
        optimizer.zero_grad()

        y_hat = model(X_train)
        loss = loss_fn(y_hat, y_train)
        loss.backward()

        optimizer.step()
        model.eval()
        with torch.no_grad():
            y_hat_val = model(X_val)
            loss_val = loss_fn(y_hat_val, y_val)

            val_loss_vals.append(loss_val.item())
            train_loss_vals.append(loss.item())

        estimated_w.append(model.weight.detach().numpy().flatten())

    return model, train_loss_vals, val_loss_vals, estimated_w, loss_val, step

def visualize_loss(train_loss_vals: List[float], val_loss_vals: List[float], n_steps: int):
    plt.plot(range(n_steps + 1), train_loss_vals)
    plt.plot(range(n_steps + 1), val_loss_vals)
    plt.legend(["Training loss", "Validation loss"])
    plt.xlabel("Steps")
    plt.ylabel("Loss value")
    plt.title("Training and validation loss")
    plt.show()

'''
Bonus Question functions
'''
def bonus_create_dataset(X_range: Tuple[float, float], sample_size: int, sigma: float, seed: int = 42):
    torch.manual_seed(seed)
    X_min, X_max = X_range

    X = torch.rand(sample_size)
    X = X * (X_max - X_min) + X_min

    y_hat = 2 * torch.log(X + 1) + 3

    y = y_hat + torch.normal(torch.zeros(sample_size), sigma * torch.ones(sample_size))
    
    return X, y

def bonus_visualize_data(X_noisy: torch.Tensor, y_noisy: torch.Tensor, X_true: torch.Tensor, title: str = ""):
    y_true = 2 * np.log(X_true + 1) + 3

    plt.plot(X_true, y_true, 'b', label='True function')
    plt.scatter(X_noisy, y_noisy, c='r', alpha=.6, label='Noisy data')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.show()

def bonus_plot_function_compare(X: torch.Tensor, estimated_coeffs: np.ndarray, estimated_bias: float):
    y_hat = 2 * np.log(X + 1) + 3

    plt.plot(X, y_hat, 'b:', label='True function')
    p_estimated = X * estimated_coeffs[0] + estimated_bias

    plt.plot(X, p_estimated, 'r', label='Estimated function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title('True VS estimated function')
    plt.show()

if __name__ == "__main__":
    # Initialization of the parameters
    z_range = [-2, 2]
    coeffs = np.array([1/30,-0.1, 5, -1, 1])
    n_samples = 500
    sigma = 0.5
    n_steps = 600
    colors = ['r', 'g', 'b', 'y', 'm']

    '''
    Code for Q1
    '''
    assert np.version.version=="2.1"

    '''
    Code for Q2
    '''
    plot_polynomial(coeffs, [-4, 4], 'r')
    
    '''
    Code for Q3 and Q4
    '''
    X_train, y_train = create_dataset(coeffs, z_range, n_samples, sigma, 0)
    X_val, y_val= create_dataset(coeffs, z_range, n_samples, sigma, 1)

    '''
    Code for Q5
    '''
    visualize_data(X_train, y_train, coeffs, z_range, 'Training data')
    visualize_data(X_val, y_val, coeffs, z_range, 'Validation data')

    '''
    Code for Q6
    '''
    model, train_loss_vals, val_loss_vals, estimated_w, loss_val, step = create_train_model(X_train, y_train, X_val, y_val, n_steps)

    '''
    Code for Q7
    '''
    print("Training done, with an evaluation loss of {}".format(loss_val.item()))
    print("Final w:", model.weight)

    visualize_loss(train_loss_vals, val_loss_vals, step)

    '''
    Code for Q8
    '''
    estimated_coeffs = model.weight.detach().numpy().flatten()[::-1]
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max)

    p = np.poly1d(coeffs)
    plt.plot(z, p(z), 'b:', label='True polynomial')

    p_estimated = np.poly1d(estimated_coeffs)
    plt.plot(z, p_estimated(z), 'r', label='Estimated polynomial')

    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.legend()
    plt.title('True VS estimated polynomial')
    plt.show()

    '''
    Code for Q9
    '''
    estimated_w = np.array(estimated_w)
    coeffs = coeffs[::-1]
    
    plt.figure(figsize=(9, 7))

    for i in range(5):
        plt.plot(range(n_steps), estimated_w[:, i], label=f'Estimated $w_{{{i}}}$', color=colors[i])
        plt.axhline(y = coeffs[i], alpha=.6 , linestyle = 'dotted', label='_Hidden', color=colors[i])
        plt.text(n_steps, coeffs[i], f'$w_{{{i}}}$', fontsize=10, va='bottom', ha='left')
    
    plt.xlabel('Steps')
    plt.ylabel('W value')
    plt.legend(loc='upper left')
    plt.title('Estimated w over time')
    plt.show()

    '''
    Code for Bonus Question
    '''
    ranges = [[-0.05, 0.01], [-0.05, 10]]

    for value_range in ranges: 
        X_min, X_max = value_range
        X_true = np.linspace(X_min, X_max)

        bonus_X_train, bonus_y_train = bonus_create_dataset(value_range, n_samples, sigma, 0)
        bonus_X_val, bonus_y_val = bonus_create_dataset(value_range, n_samples, sigma, 1)

        bonus_visualize_data(bonus_X_train, bonus_y_train, X_true, 'Training data')
        bonus_visualize_data(bonus_X_val, bonus_y_val, X_true, 'Validation data')

        bonus_model, bonus_train_loss_vals, bonus_val_loss_vals, _, bonus_loss_val, bonus_step = create_train_model(bonus_X_train, bonus_y_train, bonus_X_val, bonus_y_val, 200, 1, True)

        print("Training done, with an evaluation loss of {}".format(bonus_loss_val.item()))
        print("Final w:", bonus_model.weight)
        visualize_loss(bonus_train_loss_vals, bonus_val_loss_vals, bonus_step)

        estimated_coeffs = bonus_model.weight.detach().numpy().flatten()[::-1]
        estimated_bias = bonus_model.bias.detach().numpy()

        bonus_plot_function_compare(X_true, estimated_coeffs, estimated_bias)