from manim import *
import numpy as np

def loss_fn(x, y):
    return 0.5 * (x ** 2 + 3 * y ** 2)


def grad_fn(x, y):
    return np.array([x, 6 * y])



def compute_path(start=(2.5, 1.8), lr=0.15, steps=40):
    path = [np.array(start, dtype=float)]
    for _ in range(steps):
        x, y = path[-1]
        g = grad_fn(x, y)
        new_point = path[-1] - lr * g
        path.append(new_point)
    return path


PATH_2D = compute_path()
PATH_3D = [np.array([p[0], p[1], loss_fn(p[0], p[1])]) for p in PATH_2D]




class GradientDescent3D(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 8, 2],
            x_length=7,
            y_length=7,
            z_length=5,
        )

        surface = Surface(
            lambda u, v: axes.c2p(u, v, loss_fn(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(36, 36),
            fill_opacity=0.6,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, distance=8)
        self.add(axes, surface)

        # Ball starts at the first point of the precomputed path
        start = axes.c2p(*PATH_3D[0])
        ball = Sphere(radius=0.12, color=YELLOW).move_to(start)
        ball.set_shade_in_3d(True)

        trail = TracedPath(ball.get_center, stroke_color=YELLOW, stroke_width=3)

        self.add(trail)
        self.play(FadeIn(ball), run_time=0.5)
        self.begin_ambient_camera_rotation(rate=0.05)

        for point in PATH_3D[1:]:
            target = axes.c2p(*point)
            self.play(ball.animate.move_to(target), run_time=0.15, rate_func=linear)

        self.wait(2)


class GradientDescent2D(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=7,
            y_length=7,
            axis_config={"include_tip": False},
        )
        self.add(axes)
        
        contours = VGroup()
        for c, color in zip([0.5, 1.5, 3, 5, 7.5], [GREEN, TEAL, YELLOW, ORANGE, RED]):
            ellipse = ParametricFunction(
                lambda t, c=c: axes.c2p(
                    np.sqrt(2 * c) * np.cos(t),
                    np.sqrt(2 * c / 3) * np.sin(t),
                ),
                t_range=[0, TAU],
                color=color,
                stroke_width=2,
            )
            contours.add(ellipse)

        self.play(*[Create(c) for c in contours], run_time=2)

        dot = Dot(axes.c2p(*PATH_2D[0]), color=YELLOW, radius=0.08)
        trail = TracedPath(dot.get_center, stroke_color=YELLOW, stroke_width=3)
        self.add(trail)
        self.play(FadeIn(dot))

        # Arrow showing the current negative-gradient step direction
        step_label = MathTex(r"-\nabla L").scale(0.6).next_to(dot, UP)

        for point in PATH_2D[1:]:
            target = axes.c2p(*point)
            arrow = Arrow(dot.get_center(), target, buff=0, stroke_width=2, color=RED)
            self.play(
                Create(arrow),
                run_time=0.1,
            )
            self.play(
                dot.animate.move_to(target),
                FadeOut(arrow),
                run_time=0.15,
                rate_func=linear,
            )

        final_label = Text("Converged", font_size=28, color=GREEN).next_to(dot, DOWN)
        self.play(Write(final_label))
        self.wait(2)