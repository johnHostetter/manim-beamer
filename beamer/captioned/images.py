from manim import (
    Scene,
    ORIGIN,
    SVGMobject,
    Text,
    BLACK,
    VGroup,
    ImageMobject,
    Group,
    FadeIn,
    DOWN,
    Create,
    Write,
)


class CaptionedSVG(Scene):
    def __init__(self, path, caption, **kwargs):
        self.path = path
        self.caption = caption
        super().__init__(**kwargs)

    def construct(self, origin=ORIGIN, scale=1.0):
        self.draw(origin, scale)

    def draw(self, origin, scale, target_scene=None, animate=True):
        svg = SVGMobject(self.path).scale(2)
        text = (
            Text(self.caption, font="TeX Gyre Termes", color=BLACK)
            .scale(0.7)
            .next_to(svg, DOWN)
        )
        group = VGroup(svg, text)
        group.scale(scale_factor=scale).move_to(origin)
        if target_scene is None:
            target_scene = self
        if animate:
            target_scene.play(Create(svg, run_time=3), Write(text, run_time=3))
        else:
            target_scene.add(group)


class CaptionedJPG(Scene):
    def __init__(self, path, caption, original_image_scale: float = 0.25, **kwargs):
        self.path = path
        self.caption = caption
        self.original_image_scale = original_image_scale
        super().__init__(**kwargs)

    def construct(self, origin=ORIGIN, scale=1.0):
        self.draw(origin, scale)

    def draw(self, origin, scale, target_scene=None, animate=True):
        jpg = ImageMobject(self.path).scale(self.original_image_scale)
        text = (
            Text(self.caption, font="TeX Gyre Termes", color=BLACK)
            .scale(0.7)
            .next_to(jpg, DOWN)
        )
        group = Group(jpg, text)
        group.scale(scale_factor=scale).move_to(origin)
        if target_scene is None:
            target_scene = self
        if animate:
            target_scene.play(FadeIn(group, run_time=3))
        else:
            target_scene.add(group)
