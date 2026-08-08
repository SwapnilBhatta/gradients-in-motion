# gradients-in-motion
# Learning, Boundaries

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

## Requirements

```bash
pip install manim numpy
```

Manim also needs [FFmpeg](https://ffmpeg.org/) and a working Pango/Cairo install for text rendering — see the [official Manim installation guide](https://docs.manim.community/en/stable/installation.html) for OS-specific steps (Windows in particular needs its Cairo/Pango setup done carefully).

## Usage

Each script defines one `Scene` class. Render with:

```bash
manim -pql <script>.py <SceneName>      # fast preview (low quality)
manim -pqh <script>.py <SceneName>      # final render (high quality)
```

For example:

```bash
manim -pqh perceptron.py PerceptronLearning
manim -pqh poly_logistic_regression.py PolyLogisticRegression
manim -pqh neural_network.py NeuralNetworkClassifier
```

Rendered output lands in `media/videos/<script_name>/<quality>/`.

## Notes on each script

- **Training is precomputed, not animated live.** Every script trains the model fully in NumPy first, saving a history of parameters at each step, *then* animates through that saved history. This keeps timing fully controllable and avoids re-running expensive backprop inside the render loop.
- **`poly_logistic_regression.py` and `neural_network.py`** share the same "two moons" synthetic dataset, so they're directly comparable — a good pair to render side by side if you want to show a polynomial model vs. a neural network solving the same non-linear problem.
- Regularization is used in the polynomial and neural network scripts to keep boundaries smooth rather than overfitting to individual points — tune the `reg` parameter if you want a wigglier or smoother result.

## Customizing

Common things you'll want to tweak per script:

- `np.random.seed(...)` — regenerate the dataset layout
- `lr` (learning rate) and `epochs` — training speed/duration, which affects how many animation frames get generated
- `checkpoints` / `frame_stride` — how many training steps actually get animated (fewer = faster to render, more = smoother motion)
- Color scheme — all scripts use Manim's built-in colors (`BLUE`, `RED`, `YELLOW`, etc.), easy to swap

## Troubleshooting

If you hit a `ParseError: no element found` on `Text(...)` objects, this is almost always a corrupted/stale SVG cache from Manim's text rendering pipeline, not a bug in these scripts. Try, in order:

1. `manim ... --disable_caching` to bypass the cache for one run
2. Delete Manim's text cache directory (`python -c "from manim import config; print(config.get_dir('text_dir'))"` to find it)
3. Temporarily exclude your project folder from antivirus real-time scanning
4. Move your project out of any cloud-synced folder (OneDrive, Dropbox, etc.) — sync clients can lock temp files mid-write

## License

MIT (or your preference — update this section).
