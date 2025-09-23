from typing import List, Type, Union

from manim import DOWN, ORIGIN, AnimationGroup, MathTex, SVGMobject, Text, VGroup, Write
from manim_slides import Slide

from manim_beamer.blocks import Block
from manim_beamer.slides.base import BeamerSlide


class SlideWithBlocks(BeamerSlide):
    def __init__(
        self,
        title: str,
        subtitle: Union[None, str],
        blocks: List[Union[Type[Block], VGroup]],
        width_buffer: float = 3.0,
        height_buffer: float = 1.0,
    ):
        super().__init__(
            title=title,
            subtitle=subtitle,
            width_buffer=width_buffer,
            height_buffer=height_buffer,
        )
        self.blocks: List[Type[Block]] = blocks

    def make_block_and_focus(
        self,
        block: Block,
        scale: float,
        below: Union[None, SVGMobject, Block],
        target_scene: Union[None, Slide],
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
                    width=block.block_background.width + self.width_buffer,
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

    def draw(self, origin, scale, target_scene: Union[None, Slide], animate=True):
        if target_scene is None:
            target_scene = self

        if origin is None:
            origin = ORIGIN

        content, animations = self.inner_draw(
            origin, scale, target_scene=target_scene, animate=animate
        )
        m_object_to_be_below = content

        for animation in animations:
            target_scene.play(animation)
            target_scene.wait(1)
            target_scene.next_slide()

        # iterate over the blocks and create them
        for block in self.blocks:
            # for block in content[1:]:
            if isinstance(block, Block):
                self.make_block_and_focus(
                    block,
                    scale=scale,
                    below=m_object_to_be_below,
                    target_scene=target_scene,
                    animate=animate,
                )
                content.add(block.get_vgroup())
                m_object_to_be_below = block.block_background
            elif (
                isinstance(block, Text)
                or isinstance(block, MathTex)
                or isinstance(block, VGroup)
            ):
                block.scale(scale_factor=scale).next_to(
                    m_object_to_be_below, DOWN, buff=0.5
                )
                if animate:
                    target_scene.play(
                        Write(block),
                        self.move_and_fit_camera_to_content(
                            content=block,
                            target_scene=target_scene,
                            animate_camera=True,
                        ),
                    )
                    target_scene.wait(1)
                else:
                    target_scene.add(block)
                content.add(block)
                m_object_to_be_below = block
            else:
                # raise an error if the block is not a 'Block' object
                raise ValueError("Invalid block type. Must be a 'Block' object")
            if animate:
                target_scene.wait(1)
                target_scene.next_slide()

        # display the entire slide

        camera_animation: Union[None, AnimationGroup] = (
            self.move_and_fit_camera_to_content(
                content=content, target_scene=target_scene, animate_camera=animate
            )
        )

        if camera_animation is not None:
            target_scene.play(camera_animation)
            target_scene.wait(1)
