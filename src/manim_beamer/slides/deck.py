from typing import List, Type

from manim import FadeOut, MovingCameraScene
from manim_slides import Slide

from manim_beamer.slides.base import BeamerSlide


class SlideShow(BeamerSlide, MovingCameraScene):
    """
    A class to create a slide show of multiple Slide objects.
    """

    def __init__(self, slides, zoom_with_height: bool = False, **kwargs):
        super().__init__(title="", subtitle="", **kwargs)
        self.slides: List[Type[Slide]] = slides
        self.zoom_with_height: bool = zoom_with_height

    def construct(self):
        for slide in self.slides:
            # draw the slide but ignore the returned content
            _ = slide.draw(origin=None, scale=1.0, target_scene=self, animate=True)
            # fade out the slide content
            self.play(*[FadeOut(m_object) for m_object in self.mobjects])
