# Polynomial Regression with Gradient Descent

This repository contains Python code for performing polynomial regression using gradient descent, implemented with PyTorch, as required for **Assignment 1**. The code provides functions for polynomial plotting, dataset generation, model training, and visualization.

## Assignment Overview

The task involves performing polynomial regression on a dataset generated from a given polynomial. The regression is done using gradient descent with PyTorch's `torch.nn.Linear` model to estimate the coefficients of the polynomial. The steps include:

-   Plotting the polynomial.
-   Generating noisy datasets for training and evaluation.
-   Training a linear regression model using gradient descent.
-   Visualizing the model's performance and results.

## Features

-   **Polynomial Plotting**: A function to visualize any polynomial by defining its coefficients and plotting range.
-   **Dataset Generation**: Create noisy datasets for training and validation using polynomial values with added Gaussian noise.
-   **Polynomial Regression Model**: Train a neural network to learn the polynomial's coefficients using gradient descent.
-   **Loss Visualization**: Track the loss function over time to observe model convergence.
-   **Parameter Tracking**: Visualize how the model's parameters (coefficients) converge towards the true polynomial values.

## Requirements

Make sure you have the following Python packages installed:

-   `matplotlib`
-   `numpy`
-   `torch`

You can install them using:

```bash
pip install matplotlib numpy torch
```

## Neural Network Model for Polynomial Regression

In this project, we perform **polynomial regression** using a simple linear regression model in PyTorch. While the polynomial is of degree 4, the regression model estimates the coefficients by mapping the inputs to the polynomial's features. Here is how the model is structured:

### Model Definition

We use PyTorch's `torch.nn.Linear` module, which implements a basic fully connected layer (linear transformation). In this case, it is used to map the polynomial features (`z`, `z²`, `z³`, `z⁴`) to the target output (`y`).

-   **Input features**: The model receives a vector `x = [1, z, z², z³, z⁴]`, where `z` is the independent variable.
-   **Output**: The model outputs a scalar `y_hat`, which is the predicted value based on the learned coefficients.

### Model Architecture

The architecture is a single-layer linear model, defined as:

```python
class PolynomialRegressionModel(nn.Module):
    def __init__(self):
        super(PolynomialRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=5, out_features=1, bias=True)  # Linear layer

    def forward(self, x):
        return self.linear(x)
```

-   **Input**: A tensor of shape `(N, 5)`, where `N` is the batch size and `5` represents the 5 input features `[1, z, z², z³, z⁴]`.
-   **Output**: A tensor of shape `(N, 1)`, representing the predicted values.

### Training Process

We use **gradient descent** to minimize the **mean squared error (MSE)** between the predicted values (`y_hat`) and the true values (`y`). The training process involves iterating over the dataset and updating the model's parameters (coefficients) to reduce the loss function.

#### Steps:

1. **Initialize the Model**:
    ```python
    model = PolynomialRegressionModel()
    ```
2. **Define the Loss Function**: We use MSELoss for regression.
    ```python
    criterion = nn.MSELoss()
    ```
3. **Define the Optimizer**: We use the `Adam` optimizer to perform gradient-based optimization.
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ```
4. **Training Loop**: In each iteration, we:
    - Perform a forward pass to compute predictions.
    - Calculate the loss (error).
    - Backpropagate the error and update the model's parameters.
    - Track the loss over time to ensure the model is converging.

### Polynomial Features

To perform polynomial regression, we transform the original data points `z` into polynomial features `[1, z, z², z³, z⁴]`. This allows us to fit a higher-degree polynomial using linear regression.

### Learning Rate and Convergence

The **learning rate** controls how large the parameter updates are at each step of gradient descent. If the learning rate is too high, the model might overshoot the minimum of the loss function, causing instability. If it is too small, the model will take a long time to converge. During the training process, different learning rates are experimented with to find the optimal value.

## Functions

### 1. `plot_polynomial(coeffs: np.array, z_range: Tuple[float, float], color='b')`

Plots a polynomial based on the given coefficients and range.

**Parameters:**

-   `coeffs`: A numpy array of coefficients `[w0, w1, w2, w3, w4]`.
-   `z_range`: The interval `[z_min, z_max]` for the `z` variable.
-   `color`: The color of the plot (default: blue).

### 2. `create_dataset(coeffs: np.array, z_range: Tuple[float, float], sample_size: int, sigma: float, seed: int = 42)`

Generates a noisy dataset based on the polynomial defined by the coefficients.

**Parameters:**

-   `coeffs`: Polynomial coefficients.
-   `z_range`: Range of `z` values.
-   `sample_size`: Number of data points to generate.
-   `sigma`: Standard deviation of noise added to the data.
-   `seed`: Seed for random number generation (default: 42).

Returns two `torch.tensor` objects representing the dataset.

### 3. `visualize_data(X, y, coeffs, z_range, title="")`

Plots the true polynomial and the generated data (training or validation). Includes scatter plots for visualizing the dataset.

**Parameters:**

-   `X`, `y`: Dataset generated by `create_dataset`.
-   `coeffs`: Polynomial coefficients.
-   `z_range`: Range of `z` values.
-   `title`: Title of the plot.

### 4. **Polynomial Regression Using Gradient Descent**

This step trains a model to fit the generated dataset using gradient descent in PyTorch. The training loop involves:

-   Forward pass to compute the predicted polynomial values.
-   Calculation of the loss (MSE).
-   Backpropagation to compute gradients and update model parameters.

## Usage

1. **Plot a Polynomial:**

```python
from main import plot_polynomial
plot_polynomial(np.array([1, -1, 5, -0.1, 1/30]), (-4, 4))
```

2. **Generate a Dataset:**

```python
from main import create_dataset
train_data, train_labels = create_dataset(np.array([1, -1, 5, -0.1, 1/30]), (-2, 2), 500, sigma=0.5, seed=0)
```

3. **Visualize Data:**

```python
from main import visualize_data
visualize_data(train_data, train_labels, np.array([1, -1, 5, -0.1, 1/30]), (-2, 2), title="Training Data")
```

4. **Train the Model**:

```python
# Initialize the model
model = PolynomialRegressionModel()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    # Forward pass, loss calculation, backward pass, optimizer step
    pass  # Example loop
```

## License

This project is licensed under the MIT License.
