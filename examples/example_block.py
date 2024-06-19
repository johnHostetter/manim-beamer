from manim import *

from mbeamer.blocks import ExampleBlock

config.background_color = WHITE
light_theme_style = {
    "fill_color": BLACK,
    "background_stroke_color": WHITE,
}


class BlockScene(Scene):
    def construct(self):
        block = ExampleBlock(title="Example Block", content="This is an example block")
        self.play(block.get_animation(scale_factor=1.0, below=None, animate=True))
        self.wait(3)


if __name__ == "__main__":
    c = BlockScene()
    c.render()
