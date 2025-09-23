from typing import Union

from manim import BLACK, ITALIC, ORIGIN, FadeOut, Group, SVGMobject, Text, Write
from manim_slides import Slide


class PromptSlide(Slide):
    def __init__(
        self,
        prompt: str,
        skip: bool = False,
        default_m_object: Union[
            None, SVGMobject
        ] = None,  # allows for either Text or MathTex
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_str: str = prompt
        self.skip: bool = skip  # whether to not focus on the slide
        self.default_m_object = Text if default_m_object is None else default_m_object

    def construct(self):
        self.draw(origin=ORIGIN, scale=1.0)
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)

    def draw(self, origin, scale, target_scene=None, animate=True):
        if target_scene is None:
            target_scene = self

        prompt_text = (
            self.default_m_object(self.prompt_str, color=BLACK, slant=ITALIC)
            .move_to(origin)
            .scale(scale)
        )

        if animate:
            target_scene.play(Write(prompt_text))
            target_scene.wait(2)
        else:
            target_scene.add(prompt_text)
