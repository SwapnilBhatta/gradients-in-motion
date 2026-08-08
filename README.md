# gradients-in-motion

Animated visualizations of a few classic machine learning algorithms that I made with [Manim Community Edition](https://www.manim.community/). These were originally made in June-July, 2026. Each script trains a model from scratch in NumPy, then animates the training process rather than just showing a static end result.

## What's inside

| Script | Algorithm | What it shows |
|---|---|---|
| `graddescmanim.py` | Gradient descent | A ball rolling down a 3D loss surface, paired with a matching 2D contour view |
| `percepmanim.py` | Perceptron | A linear decision boundary jumping/correcting each time it misclassifies a point |
| `classifmanim.py` | Polynomial logistic regression | A decision boundary bending around a non-linearly-separable "two moons" dataset |
| `classnn.py` | Small neural network (MLP) | A network diagram (weights strengthening/weakening) synced with its decision boundary bending into shape |

Rendered videos for each live in [`videos/`](media/videos)

## Notes on each script

- **Training is precomputed, not animated live.** Every script trains the model fully in NumPy first, saving a history of parameters at each step, *then* animates through that saved history. This keeps timing fully controllable and avoids re-running expensive backprop inside the render loop.
- **`classifmanim.py` and `classnn.py`** share the same "two moons" synthetic dataset, so they're directly comparable, a good pair to render side by side if you want to show a polynomial model vs a neural network solving the same non-linear problem.
- Regularization is used in the polynomial and neural network scripts to keep boundaries smooth rather than overfitting to individual points.


