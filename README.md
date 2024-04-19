# Gradient-Descent

This repository explores the Gradient Descent optimization algorithm, widely used in training machine learning models. It provides a clear understanding through examples and code on how gradient descent works and how it can be implemented to optimize various types of functions.

## Prerequisites

- Python 3.6 or higher
- Basic understanding of calculus and optimization

## Installation

No specific installations are necessary for the basic examples, which use pure Python. For more advanced examples involving data visualization:

```bash
pip install numpy matplotlib
```

## Example - Simple Gradient Descent Implementation

Here's a simple Python example demonstrating gradient descent applied to a quadratic function to find its minimum.

### `gradient_descent.py`

```python
import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return x**2

def derivative(x):
    return 2*x

def gradient_descent(starting_point, learning_rate, num_iterations):
    x = starting_point
    for i in range(num_iterations):
        grad = derivative(x)
        x -= learning_rate * grad
        if i % 10 == 0:
            print(f"Iteration {i}: x = {x}, f(x) = {function(x)}")
    return x

# Parameters
starting_point = 10
learning_rate = 0.1
num_iterations = 100

min_point = gradient_descent(starting_point, learning_rate, num_iterations)
print(f"Minimum point at x = {min_point}, f(x) = {function(min_point)}")

# Plotting the function and the path of gradient descent
x_values = np.linspace(-10, 10, 400)
y_values = function(x_values)
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="f(x) = x^2")
plt.scatter(min_point, function(min_point), color='red', label=f"Min at x = {min_point}")
plt.title("Gradient Descent on f(x) = x^2")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
```

## Contributing

Contributions to enhance the understanding and implementation of gradient descent are welcome! Fork the repository, create your branch, and submit a pull request with your additions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
