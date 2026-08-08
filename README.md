# gradients-in-motion

Animated visualizations of classic machine learning algorithms, built with [Manim Community Edition](https://www.manim.community/). Each script trains a model from scratch in NumPy, then animates the training process — decision boundaries shifting, weights strengthening, loss dropping — rather than just showing a static end result.

## What's inside

| Script | Algorithm | What it shows |
|---|---|---|
| `gradient_descent.py` | Gradient descent | A ball rolling down a 3D loss surface, paired with a matching 2D contour view |
| `perceptron.py` | Perceptron | A linear decision boundary jumping/correcting each time it misclassifies a point |
| `logistic_regression.py` | Logistic regression | A linear boundary shifting *smoothly* every gradient step, with a probability heatmap |
| `poly_logistic_regression.py` | Polynomial logistic regression | A **curved** decision boundary bending around a non-linearly-separable "two moons" dataset |
| `neural_network.py` | Small neural network (MLP) | A network diagram (weights strengthening/weakening) synced with its decision boundary bending into shape |

Rendered videos for each live in [`videos/`](videos/) (or wherever you point `manim`'s output — see below).

## Notes on each script

- **Training is precomputed, not animated live.** Every script trains the model fully in NumPy first, saving a history of parameters at each step, *then* animates through that saved history. This keeps timing fully controllable and avoids re-running expensive backprop inside the render loop.
- **`poly_logistic_regression.py` and `neural_network.py`** share the same "two moons" synthetic dataset, so they're directly comparable — a good pair to render side by side if you want to show a polynomial model vs. a neural network solving the same non-linear problem.
- Regularization is used in the polynomial and neural network scripts to keep boundaries smooth rather than overfitting to individual points — tune the `reg` parameter if you want a wigglier or smoother result.


