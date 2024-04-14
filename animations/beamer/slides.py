from typing import Union as U, Type

from manim import *
from manim_slides import Slide

from animations.beamer.blocks import Block


class PromptSlide(Slide):
    def __init__(self, prompt: str, skip: bool = False, **kwargs):
        super().__init__(**kwargs)
        # self.title_str: str = title
        self.prompt_str: str = prompt
        self.skip: bool = skip  # whether to not focus on the slide

        # # create the manim objects for the slide title
        # self.title_text: Text = Text(
        #     self.title_str,
        #     font="TeX Gyre Termes",
        #     color=BLACK,
        #     font_size=60,
        #     weight=BOLD,
        # ).to_edge(UP)
        # # create the overall contents of the slide
        # self.contents: VGroup = VGroup(self.title_text)

    def construct(self):
        self.draw(origin=ORIGIN, scale=1.0)
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)

    def draw(self, origin, scale, target_scene=None):
        if target_scene is None:
            target_scene = self
        target_scene.play(
            Write(Text(self.prompt_str, color=BLACK, slant=ITALIC).move_to(origin).scale(scale))
        )
        target_scene.wait(2)


class SlideWithBlocks(MovingCameraScene, Slide):
    def __init__(self, title: str, blocks: List[Type[Block]], **kwargs):
        super().__init__(**kwargs)
        self.title_str: str = title
        self.blocks: List[Type[Block]] = blocks

        # create the manim objects for the slide title
        self.title_text: Text = Text(
            self.title_str,
            font="TeX Gyre Termes",
            color=BLACK,
            font_size=60,
            weight=BOLD,
        ).to_edge(UP)
        # create the overall contents of the slide
        self.contents: VGroup = VGroup(self.title_text)

    def make_block_and_focus(
        self,
        block: Block,
        scale: float,
        below: U[None, Text, Block],
        target_scene: U[None, Slide],
    ):
        if target_scene is None:
            target_scene = self
        target_scene.play(
            block.get_animation(scale_factor=scale, below=below),
            target_scene.camera.frame.animate.move_to(
                block.block_background.get_center()
            ).set(
                width=block.block_background.width + 3,
                # height=block.block_background.height + 3
            ),
        )

    def construct(self):
        self.draw(ORIGIN, 1.0, target_scene=self)

    def draw(self, origin, scale, target_scene: U[None, Slide]):
        if target_scene is None:
            target_scene = self

        # position the slide correctly
        self.contents.move_to(origin)
        self.contents.scale(scale)

        target_scene.wait(1)
        target_scene.next_slide()
        target_scene.play(Write(self.title_text))

        self.wait(1)
        self.next_slide()
        m_object_to_be_below = self.title_text
        # iterate over the blocks and create them
        for block in self.blocks:
            # for block in self.contents[1:]:
            if isinstance(block, Block):
                self.make_block_and_focus(
                    block,
                    scale=scale,
                    below=m_object_to_be_below,
                    target_scene=target_scene,
                )
                self.contents.add(block.get_vgroup())
                m_object_to_be_below = block.block_background
            else:
                # raise an error if the block is not a 'Block' object
                raise ValueError("Invalid block type. Must be a 'Block' object")
            target_scene.wait(1)
            target_scene.next_slide()
        # focus the camera on the entire slide
        target_scene.play(
            target_scene.camera.frame.animate.move_to(self.contents.get_center()).set(
                height=self.contents.height + 1
            )
        )
        target_scene.wait(3)
