from typing import Union as U, Type

from manim import *
from manim_slides import Slide

from animations.beamer.blocks import Block
from animations.beamer.lists import BeamerList


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

    def draw(self, origin, scale, target_scene=None, animate=True):
        if target_scene is None:
            target_scene = self

        prompt_text = (
            Text(self.prompt_str, color=BLACK, slant=ITALIC)
            .move_to(origin)
            .scale(scale)
        )

        if animate:
            target_scene.play(Write(prompt_text))
            target_scene.wait(2)
        else:
            target_scene.add(prompt_text)


class BeamerSlide(MovingCameraScene, Slide):
    def __init__(self, title: str, subtitle: U[None, str], **kwargs):
        super().__init__(**kwargs)
        self.title_str: str = title
        self.subtitle_str: str = subtitle

        # create the manim objects for the slide title
        self.title_text: Text = Text(
            self.title_str,
            font="TeX Gyre Termes",
            color=BLACK,
            font_size=60,
            weight=BOLD,
        ).to_edge(UP)
        if self.subtitle_str is not None:
            self.subtitle_text: Text = Text(
                self.subtitle_str,
                font="TeX Gyre Termes",
                color=BLACK,
                font_size=30,
                slant=ITALIC,
            ).next_to(self.title_text, DOWN)
        # create the overall contents of the slide
        self.contents: VGroup = VGroup(self.title_text)
        if self.subtitle_str is not None:
            self.contents.add(self.subtitle_text)

    def inner_draw(self, origin, scale, target_scene=None, animate=True) -> VGroup:
        """
        Draw the slide contents (title and subtitle - if applicable) on the scene
        and then return the last displayed text object.

        Args:
            origin: The origin of the slide.
            scale: The scale factor to apply to the slide contents.
            target_scene: The scene to draw the slide on. If None, the current scene is used.
            animate: Whether to animate the drawing of the slide.

        Returns:
            The current text objects displayed on the scene.
        """
        if target_scene is None:
            target_scene = self

        # position the slide correctly
        self.contents.move_to(origin)
        self.contents.scale(scale)

        if animate:
            target_scene.wait(1)
            target_scene.next_slide()
            # position the camera correctly
            self.play(
                Succession(
                    target_scene.camera.frame.animate.set(
                        width=self.contents.width
                        + 2,  # height=self.contents.height + 1
                    ),
                    Write(self.title_text),
                )
            )
        else:
            target_scene.add(self.title_text)

        if self.subtitle_str is not None:
            if animate:
                target_scene.play(Write(self.subtitle_text))
                target_scene.wait(1)
                target_scene.next_slide()
            else:
                target_scene.add(self.subtitle_text)

        if animate:
            target_scene.wait(1)
            self.next_slide()
        return self.contents


class SlideWithList(BeamerSlide):
    def __init__(self, title: str, subtitle: U[None, str], beamer_list: BeamerList):
        super().__init__(title=title, subtitle=subtitle)
        self.beamer_list: BeamerList = beamer_list

    def construct(self):
        self.draw(ORIGIN, 1.0, target_scene=self)

    def draw(self, origin, scale: float, target_scene: U[None, Slide], animate=True):
        m_object_to_be_below = self.inner_draw(origin, scale, target_scene=target_scene)
        # create the list object
        list_group = self.beamer_list.get_list(scale_factor=scale)
        buffer_with_prev_object = 0.5
        list_group.scale(scale_factor=scale).next_to(
            m_object_to_be_below, DOWN, buff=buffer_with_prev_object * scale
        )
        all_content = VGroup(m_object_to_be_below, list_group)
        if animate:
            target_scene.play(
                Create(list_group),
                self.camera.frame.animate.move_to(all_content.get_center()).set(
                    width=all_content.width + 2, height=all_content.height + 2
                ),
            )
            target_scene.wait(2)
            target_scene.next_slide()
            target_scene.wait(2)
        else:
            target_scene.add(list_group)


class SlideWithBlocks(BeamerSlide):
    def __init__(self, title: str, subtitle: U[None, str], blocks: List[Type[Block]]):
        super().__init__(title=title, subtitle=subtitle)
        self.blocks: List[Type[Block]] = blocks

    def make_block_and_focus(
        self,
        block: Block,
        scale: float,
        below: U[None, Text, Block],
        target_scene: U[None, Slide],
        animate=True,
    ):
        if target_scene is None:
            target_scene = self
        if animate:
            target_scene.play(
                block.get_animation(scale_factor=scale, below=below, animate=animate),
                target_scene.camera.frame.animate.move_to(
                    block.block_background.get_center()
                ).set(
                    width=block.block_background.width + 3,
                    # height=block.block_background.height + 3
                ),
            )
        else:
            # this returns a VGroup instead of an animation and then adds it to the scene
            block_vgroup: VGroup = block.get_animation(
                scale_factor=scale, below=below, animate=animate
            )
            target_scene.add(block_vgroup[0])  # add the background first
            target_scene.add(block_vgroup[1])  # add the text group

    def construct(self):
        animate = True
        self.draw(ORIGIN, 1.0, target_scene=self, animate=animate)
        if not animate:
            self.play(self.camera.frame.animate.move_to(ORIGIN))

    def draw(self, origin, scale, target_scene: U[None, Slide], animate=True):
        m_object_to_be_below = self.inner_draw(
            origin, scale, target_scene=target_scene, animate=animate
        )
        # iterate over the blocks and create them
        for block in self.blocks:
            # for block in self.contents[1:]:
            if isinstance(block, Block):
                self.make_block_and_focus(
                    block,
                    scale=scale,
                    below=m_object_to_be_below,
                    target_scene=target_scene,
                    animate=animate,
                )
                self.contents.add(block.get_vgroup())
                m_object_to_be_below = block.block_background
            else:
                # raise an error if the block is not a 'Block' object
                raise ValueError("Invalid block type. Must be a 'Block' object")
            if animate:
                target_scene.wait(1)
                target_scene.next_slide()

        if animate:
            # focus the camera on the entire slide
            target_scene.play(
                target_scene.camera.frame.animate.move_to(
                    self.contents.get_center()
                ).set(height=self.contents.height + 1)
            )
            target_scene.wait(3)
