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
X_DATA = np.array([[x, y] for x, y, _ in DATA])
LABELS = np.array([label for _, _, label in DATA]).reshape(-1, 1)


HIDDEN = 6

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def train_mlp(X, labels, hidden=HIDDEN, lr=0.3, epochs=600, reg=0.001, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((2, hidden)) * 0.8
    b1 = np.zeros((1, hidden))
    W2 = rng.standard_normal((hidden, 1)) * 0.8
    b2 = np.zeros((1, 1))

    history = []
    for _ in range(epochs):
        z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)
        loss = (
            -np.mean(labels * np.log(a2 + 1e-9) + (1 - labels) * np.log(1 - a2 + 1e-9))
            + reg / 2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        )
        history.append((W1.copy(), b1.copy(), W2.copy(), b2.copy(), loss))

        dz2 = a2 - labels
        dW2 = a1.T @ dz2 / len(X) + reg * W2
        db2 = np.mean(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1 ** 2)
        dW1 = X.T @ dz1 / len(X) + reg * W1
        db1 = np.mean(dz1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return history

HISTORY = train_mlp(X_DATA, LABELS)


def predict_grid(gx, gy, W1, b1, W2, b2):
    pt = np.array([[gx, gy]])
    _, _, _, a2 = forward(pt, W1, b1, W2, b2)
    return float(a2[0, 0])


def make_heatmap(axes, W1, b1, W2, b2, grid_n=20, x_span=(-1.6, 2.6), y_span=(-1.3, 1.8)):
    squares = VGroup()
    x_step = (x_span[1] - x_span[0]) / grid_n
    y_step = (y_span[1] - y_span[0]) / grid_n
    for i in range(grid_n):
        for j in range(grid_n):
            gx = x_span[0] + (i + 0.5) * x_step
            gy = y_span[0] + (j + 0.5) * y_step
            prob = predict_grid(gx, gy, W1, b1, W2, b2)
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



def make_node_positions():
    input_pos = [UP * 0.9, DOWN * 0.9]
    hidden_pos = [UP * 2.2 + RIGHT * 0, UP * 1.3, UP * 0.4, DOWN * 0.4, DOWN * 1.3, DOWN * 2.2]
    hidden_pos = [p + RIGHT * 2 for p in hidden_pos]
    output_pos = [RIGHT * 4]
    return input_pos, hidden_pos, output_pos


def make_network_diagram(origin, W1, b1, W2, b2):
    input_pos, hidden_pos, output_pos = make_node_positions()
    input_pos = [origin + p for p in input_pos]
    hidden_pos = [origin + p for p in hidden_pos]
    output_pos = [origin + p for p in output_pos]

    edges = VGroup()
    max_w = max(np.abs(W1).max(), np.abs(W2).max(), 1e-6)

    # input -> hidden
    for i, ip in enumerate(input_pos):
        for h, hp in enumerate(hidden_pos):
            w = W1[i, h]
            color = interpolate_color(RED, BLUE, (w / max_w + 1) / 2)
            edge = Line(ip, hp, stroke_color=color, stroke_width=1 + 4 * abs(w) / max_w)
            edges.add(edge)

    # hidden -> output
    for h, hp in enumerate(hidden_pos):
        w = W2[h, 0]
        color = interpolate_color(RED, BLUE, (w / max_w + 1) / 2)
        edge = Line(hp, output_pos[0], stroke_color=color, stroke_width=1 + 4 * abs(w) / max_w)
        edges.add(edge)

    nodes = VGroup()
    for p in input_pos:
        nodes.add(Circle(radius=0.18, color=WHITE, fill_color=GREY_D, fill_opacity=1).move_to(p))
    for p in hidden_pos:
        nodes.add(Circle(radius=0.18, color=WHITE, fill_color=GREY_D, fill_opacity=1).move_to(p))
    for p in output_pos:
        nodes.add(Circle(radius=0.2, color=WHITE, fill_color=GREY_D, fill_opacity=1).move_to(p))

    return edges, nodes



class NeuralNetworkClassifier(Scene):
    def construct(self):
        title = Text("Neural Network Classifier", font_size=30).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-2, 3, 1],
            y_range=[-2, 2, 1],
            x_length=5.5,
            y_length=4.5,
            axis_config={"include_tip": False},
        ).shift(RIGHT * 3.2 + DOWN * 0.3)

        diagram_origin = LEFT * 4.3 + DOWN * 0.3

        W1_0, b1_0, W2_0, b2_0, _ = HISTORY[0]
        edges, nodes = make_network_diagram(diagram_origin, W1_0, b1_0, W2_0, b2_0)
        heatmap = make_heatmap(axes, W1_0, b1_0, W2_0, b2_0)

        self.play(FadeIn(heatmap), run_time=0.8)
        self.add(axes)
        self.play(Create(edges), FadeIn(nodes), run_time=1.2)

        dots = VGroup()
        for (x, y, label) in DATA:
            color = BLUE if label == 1 else RED
            dot = Dot(axes.c2p(x, y), color=color, radius=0.06, stroke_color=WHITE, stroke_width=1)
            dots.add(dot)
        self.play(*[FadeIn(d) for d in dots], run_time=1)

        loss_text = Text("Loss: --", font_size=24).to_corner(UR)
        self.play(FadeIn(loss_text))

        layer_labels = VGroup(
            Text("Input", font_size=18).next_to(diagram_origin + UP * 0.9, LEFT, buff=0.35),
            Text("Hidden", font_size=18).next_to(diagram_origin + UP * 2.9 + RIGHT * 2, UP, buff=0.2),
            Text("Output", font_size=18).next_to(diagram_origin + RIGHT * 4, RIGHT, buff=0.3),
        )
        self.play(FadeIn(layer_labels))

        checkpoints = list(range(10, 100, 12)) + list(range(100, 300, 25)) + list(range(300, 600, 40))

        for i in checkpoints:
            W1, b1, W2, b2, loss = HISTORY[i]

            new_edges, _ = make_network_diagram(diagram_origin, W1, b1, W2, b2)
            new_heatmap = make_heatmap(axes, W1, b1, W2, b2)
            new_loss_text = Text(f"Loss: {loss:.3f}", font_size=24).to_corner(UR)

            self.play(
                Transform(edges, new_edges),
                Transform(heatmap, new_heatmap),
                Transform(loss_text, new_loss_text),
                run_time=0.3,
                rate_func=linear,
            )

        final_label = Text("Converged", font_size=28, color=GREEN).next_to(title, DOWN)
        self.play(Write(final_label))
        self.wait(2)
