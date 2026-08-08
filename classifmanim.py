from manim import *
import numpy as np

np.random.seed(1)

def make_moons(n_per_class=25, noise=0.15):
    t1 = np.linspace(0, np.pi, n_per_class)
    x1, y1 = np.cos(t1), np.sin(t1)
    t2 = np.linspace(0, np.pi, n_per_class)
    x2, y2 = 1 - np.cos(t2), 1 - np.sin(t2) - 0.5

    x1 += np.random.normal(0, noise, n_per_class)
    y1 += np.random.normal(0, noise, n_per_class)
    x2 += np.random.normal(0, noise, n_per_class)
    y2 += np.random.normal(0, noise, n_per_class)

    return [(x, y, 1) for x, y in zip(x1, y1)] + [(x, y, 0) for x, y in zip(x2, y2)]

DATA = make_moons()



DEGREE = 6

def poly_features(x, y, degree=DEGREE):
    feats = []
    for total in range(1, degree + 1):
        for i in range(total + 1):
            feats.append((x ** (total - i)) * (y ** i))
    return np.array(feats)

X_raw = np.array([poly_features(x, y) for x, y, _ in DATA])
LABELS = np.array([label for _, _, label in DATA])

FEAT_MU = X_raw.mean(axis=0)
FEAT_SIGMA = X_raw.std(axis=0) + 1e-8
X_TRAIN = (X_raw - FEAT_MU) / FEAT_SIGMA


def featurize(x, y):
    raw = poly_features(x, y)
    return (raw - FEAT_MU) / FEAT_SIGMA



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(X, labels, lr=0.6, epochs=500, reg=0.008):
    n_feat = X.shape[1]
    w = np.zeros(n_feat)
    b = 0.0
    history = [(w.copy(), b, 0.0)]

    for _ in range(epochs):
        z = X @ w + b
        preds = sigmoid(z)
        error = preds - labels
        grad_w = X.T @ error / len(labels) + reg * w
        grad_b = np.mean(error)
        w = w - lr * grad_w
        b = b - lr * grad_b
        loss = (
            -np.mean(labels * np.log(preds + 1e-9) + (1 - labels) * np.log(1 - preds + 1e-9))
            + reg / 2 * np.sum(w ** 2)
        )
        history.append((w.copy(), b, loss))

    return history

HISTORY = train(X_TRAIN, LABELS)



def make_heatmap(axes, w, b, grid_n=22, x_span=(-1.6, 2.6), y_span=(-1.3, 1.8)):
    squares = VGroup()
    x_step = (x_span[1] - x_span[0]) / grid_n
    y_step = (y_span[1] - y_span[0]) / grid_n
    for i in range(grid_n):
        for j in range(grid_n):
            gx = x_span[0] + (i + 0.5) * x_step
            gy = y_span[0] + (j + 0.5) * y_step
            prob = sigmoid(featurize(gx, gy) @ w + b)
            confidence = abs(prob - 0.5) * 2
            color = interpolate_color(RED, BLUE, prob)
            sq = Rectangle(
                width=axes.x_axis.unit_size * x_step,
                height=axes.y_axis.unit_size * y_step,
            )
            sq.set_fill(color, opacity=0.35 * confidence)
            sq.set_stroke(width=0)
            sq.move_to(axes.c2p(gx, gy))
            squares.add(sq)
    return squares



def make_boundary(axes, w, b):
    def implicit_fn(x, y):
        return featurize(x, y) @ w + b

    curve = ImplicitFunction(
        implicit_fn,
        x_range=[-2.2, 3.2],
        y_range=[-2.0, 2.4],
        color=WHITE,
        stroke_width=4,
    )
    # Confine drawing to axes' plotted region visually
    curve.move_to(axes.c2p(0, 0))
    return curve



class PolyLogisticRegression(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-2, 3, 1],
            y_range=[-2, 2, 1],
            x_length=7.5,
            y_length=6,
            axis_config={"include_tip": False},
        )
        self.add(axes)

        title = Text("Polynomial Logistic Regression", font_size=30).to_edge(UP)
        self.play(Write(title))

        w0, b0, _ = HISTORY[0]
        heatmap = make_heatmap(axes, w0, b0)
        self.play(FadeIn(heatmap), run_time=1)

        dots = VGroup()
        for (x, y, label) in DATA:
            color = BLUE if label == 1 else RED
            dot = Dot(axes.c2p(x, y), color=color, radius=0.07, stroke_color=WHITE, stroke_width=1)
            dots.add(dot)
        self.play(*[FadeIn(d) for d in dots], run_time=1)

        full_legend = VGroup(
            VGroup(Dot(color=BLUE), Text("Class 1", font_size=22)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(color=RED), Text("Class 0", font_size=22)).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.6).to_edge(DOWN)
        self.play(FadeIn(full_legend))

        boundary = make_boundary(axes, w0, b0)
        self.play(Create(boundary))

        loss_text = Text("Loss: --", font_size=24).to_corner(UR)
        self.play(FadeIn(loss_text))

        checkpoints = list(range(10, 100, 15)) + list(range(100, 300, 25)) + list(range(300, 501, 40))

        for i in checkpoints:
            w, b, loss = HISTORY[i]

            new_heatmap = make_heatmap(axes, w, b)
            new_boundary = make_boundary(axes, w, b)
            new_loss_text = Text(f"Loss: {loss:.3f}", font_size=24).to_corner(UR)

            self.play(
                Transform(heatmap, new_heatmap),
                Transform(boundary, new_boundary),
                Transform(loss_text, new_loss_text),
                run_time=0.35,
                rate_func=linear,
            )

        final_label = Text("Converged", font_size=28, color=GREEN).next_to(title, DOWN)
        self.play(Write(final_label))
        self.wait(2)