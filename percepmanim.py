from manim import *
import numpy as np

np.random.seed(3)

def make_dataset(n_per_class=6, margin=0.15):
    pos, neg = [], []
    while len(pos) < n_per_class or len(neg) < n_per_class:
        x, y = np.random.uniform(-3, 3, 2)
        if y - x > margin and len(pos) < n_per_class:
            pos.append((x, y, 1))
        elif y - x < -margin and len(neg) < n_per_class:
            neg.append((x, y, -1))
    return pos + neg

DATA = make_dataset()




def train_perceptron(data, lr=0.3, max_epochs=30):
    w = np.array([1.0, 1.0])    
    b = 0.0                     
                                 
    history = [(w.copy(), b, None)]  

    for epoch in range(max_epochs):
        made_update = False
        for (x, y, label) in data:
            point = np.array([x, y])
            pred = np.sign(w @ point + b)
            if pred == 0:
                pred = -1
            if pred != label:
                w = w + lr * label * point
                b = b + lr * label
                history.append((w.copy(), b, (x, y, label)))
                made_update = True
        if not made_update:
            break

    history.append((w.copy(), b, "converged"))
    return history

HISTORY = train_perceptron(DATA)



def boundary_endpoints(w, b, span=4):
    w1, w2 = w
    if abs(w2) > 1e-6:
        x_vals = np.array([-span, span])
        y_vals = -(w1 * x_vals + b) / w2
    else:
        # near-vertical line
        y_vals = np.array([-span, span])
        x_vals = -(w2 * y_vals + b) / (w1 if abs(w1) > 1e-6 else 1e-6)
    return x_vals, y_vals



class PerceptronLearning(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=7,
            y_length=7,
            axis_config={"include_tip": False},
        )
        self.add(axes)

        title = Text("Perceptron Learning").to_edge(UP)
        self.play(Write(title))

        dots = VGroup()
        for (x, y, label) in DATA:
            color = BLUE if label == 1 else RED
            dot = Dot(axes.c2p(x, y), color=color, radius=0.09)
            dots.add(dot)
        self.play(*[FadeIn(d) for d in dots], run_time=1)

        legend = VGroup(
            Dot(color=BLUE).scale(1.2), Text("Class +1", font_size=22),
            Dot(color=RED).scale(1.2), Text("Class -1", font_size=22),
        ).arrange(RIGHT, buff=0.15)
        legend[1].next_to(legend[0], RIGHT, buff=0.1)
        legend2 = VGroup(legend[0], legend[1]).arrange(RIGHT, buff=0.1)
        full_legend = VGroup(
            VGroup(Dot(color=BLUE), Text("Class +1", font_size=22)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(color=RED), Text("Class -1", font_size=22)).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.6).to_edge(DOWN)
        self.play(FadeIn(full_legend))

        w0, b0, _ = HISTORY[0]
        x_vals, y_vals = boundary_endpoints(w0, b0)
        line = Line(
            axes.c2p(x_vals[0], y_vals[0]),
            axes.c2p(x_vals[1], y_vals[1]),
            color=YELLOW,
            stroke_width=4,
        )
        self.play(Create(line))

        update_counter = Text("Updates: 0").to_corner(UR)
        self.play(FadeIn(update_counter))
        n_updates = 0

        for (w, b, info) in HISTORY[1:]:
            x_vals, y_vals = boundary_endpoints(w, b)
            new_line = Line(
                axes.c2p(x_vals[0], y_vals[0]),
                axes.c2p(x_vals[1], y_vals[1]),
                color=YELLOW,
                stroke_width=4,
            )

            if info == "converged":
                check = Text("All points classified correctly!", font_size=26, color=GREEN)
                check.next_to(title, DOWN)
                self.play(Transform(line, new_line), run_time=0.6)
                self.play(Write(check))
                self.wait(2)
                break

            x, y, label = info
            flash_dot = Dot(axes.c2p(x, y), color=ORANGE, radius=0.16)
            self.play(FadeIn(flash_dot), run_time=0.2)
            self.play(Flash(flash_dot, color=ORANGE, flash_radius=0.35), run_time=0.3)
            self.play(Transform(line, new_line), FadeOut(flash_dot), run_time=0.5)

            n_updates += 1
            new_counter = Text(f"Updates: {n_updates}", font_size=24).to_corner(UR)
            self.play(Transform(update_counter, new_counter), run_time=0.15)

        self.wait(2)
