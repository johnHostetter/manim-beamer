from typing import Any, List, Tuple, Type, Union

import numpy as np
from manim import (
    BLACK,
    BOLD,
    DOWN,
    ITALIC,
    ORIGIN,
    RIGHT,
    UP,
    Animation,
    AnimationGroup,
    Circumscribe,
    Create,
    FadeOut,
    Group,
    MathTex,
    MovingCameraScene,
    SVGMobject,
    Table,
    Text,
    VGroup,
    Write,
)
from manim_slides import Slide

from manim_beamer import MANIM_BLUE
from manim_beamer.blocks import Block
from manim_beamer.images import CaptionedJPG
from manim_beamer.lists import BeamerList


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


class BeamerSlide(MovingCameraScene, Slide):
    def __init__(
        self,
        title: str,
        subtitle: Union[None, str],
        width_buffer: float = 3.0,
        height_buffer: float = 1.0,
        default_m_object: Union[
            None, SVGMobject
        ] = None,  # allows for either Text or MathTex
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title_str: str = title
        self.subtitle_str: str = subtitle
        self.width_buffer = width_buffer
        self.height_buffer = height_buffer
        self.default_m_object = Text if default_m_object is None else default_m_object
        # create the manim objects for the slide title
        self.title_text: SVGMobject = self.default_m_object(
            self.title_str,
            font="TeX Gyre Termes",
            color=BLACK,
            font_size=60,
            weight=BOLD,
        ).to_edge(UP)
        if self.subtitle_str is not None:
            self.subtitle_text: SVGMobject = self.default_m_object(
                self.subtitle_str,
                font="TeX Gyre Termes",
                color=BLACK,
                font_size=30,
                slant=ITALIC,
            ).next_to(self.title_text, DOWN)

    def inner_draw(
        self, origin, scale, target_scene=None, animate=True, animate_camera=True
    ) -> Tuple[VGroup, List[Any]]:
        """
        Draw the slide content (title and subtitle - if applicable) on the scene
        and then return the last displayed text object.

        Args:
            origin: The origin of the slide.
            scale: The scale factor to apply to the slide content.
            target_scene: The scene to draw the slide on. If None, the current scene is used.
            animate: Whether to animate the drawing of the slide.
            animate_camera: Whether to move the camera in the animation, or have the content already in frame. Too much
                movement of the camera can be disorienting during presentations.

        Returns:
            The current text objects displayed on the scene.
        """
        if target_scene is None:
            target_scene = self

        # make local copies to avoid modifying the original objects
        title_text = self.title_text.copy()
        subtitle_text = (
            self.subtitle_text.copy() if self.subtitle_str is not None else None
        )
        content = (
            VGroup(title_text, subtitle_text)
            if subtitle_text is not None
            else VGroup(title_text)
        )

        # position and scale the content
        content.move_to(origin)
        content.scale(scale)

        # position the camera correctly
        animations = []
        camera_animation = self.move_and_fit_camera_to_content(
            content=content, target_scene=target_scene, animate_camera=animate_camera
        )
        if camera_animation is not None:
            animations.append(camera_animation)

        if animate:
            animations.append(Write(title_text))
        else:
            target_scene.add(title_text)

        if subtitle_text is not None:
            if animate:
                animations.append(Write(subtitle_text))
            else:
                target_scene.add(subtitle_text)

        return content, animations

    @staticmethod
    def fit_camera_to_content(
        content: VGroup,
        target_scene: Slide,
        animate_camera: bool,
    ) -> Union[None, Animation]:
        """
        Given the content of interest, the camera will zoom in/out to match the size of the content's scale either with
        or without an animation. The scaling effect of the camera with respect to the content is based on the longest
        dimension of the content's boundary. If an animation is requested, an animation will be returned, but has not
        yet been played. Otherwise, the effect occurs immediately.

        Args:
            content: The content that the camera should fit within its frame.
            target_scene: The scene/slide that the method should be applied on.
            animate_camera: Whether to produce an animation illustrating this effect.

        Returns:
            None if no animation is requested. Otherwise, an animation is returned that performs the adjustments.
        """

        content_dimensions: List[str] = ["width", "height"]
        previous_camera_settings = dict(
            zip(
                content_dimensions,
                map(
                    lambda x: getattr(target_scene.camera.frame, x), content_dimensions
                ),
            )
        )

        padding: float = 0.1
        for idx, content_dimension in enumerate(content_dimensions):
            content_dimension_val: np.float64 = getattr(content, content_dimension)
            if (
                getattr(target_scene.camera.frame, content_dimension)
                <= content_dimension_val
            ):
                scale_camera_kwargs = {
                    content_dimension: content_dimension_val
                    + (content_dimension_val * padding)
                }
                target_scene.camera.frame.set(**scale_camera_kwargs)

        if animate_camera:
            # restore it to the previous camera settings so we can animate the effect
            for key, value in previous_camera_settings.items():
                target_scene.camera.frame.set(key=value)

            scale_camera_kwargs = {
                "width": target_scene.camera.frame.width,
                "height": target_scene.camera.frame.height,
            }

            # create an animation illustrating the effect of fitting the camera to the content
            return target_scene.camera.frame.animate.set(**scale_camera_kwargs)

        # camera is already fit to the content without creating an animation
        return None

    @staticmethod
    def move_and_fit_camera_to_content(
        content,
        target_scene,
        animate_camera: bool,
    ) -> Union[None, List[Animation]]:
        """
        Given the content of interest, the camera will zoom in/out as well as move to display the content's entirety
        either with or without an animation. The scaling effect of the camera with respect to the content is based on
        the longest dimension of the content's boundary. If an animation is requested, an animation will be returned,
        but has not yet been played. Otherwise, the effect occurs immediately.

        Args:
            content: The content that the camera should fit within its frame.
            target_scene: The scene/slide that the method should be applied on.
            animate_camera: Whether to produce an animation illustrating this effect.

        Returns:
            None if no animation is requested. Otherwise, an animation is returned that performs the adjustments.
        """
        camera_fit_animation: Union[None, Animation] = (
            BeamerSlide.fit_camera_to_content(
                content=content,
                target_scene=target_scene,
                animate_camera=animate_camera,
            )
        )
        if camera_fit_animation is not None and animate_camera:
            return [
                camera_fit_animation,
                target_scene.camera.frame.animate.move_to(content),
            ]
        elif camera_fit_animation is None and not animate_camera:
            # the camera has already been fit to the content, only thing left to do is to move the camera
            target_scene.camera.frame.move_to(content.get_center())
        else:
            raise ValueError(
                "An unexpected situation has occurred. An animation was produced even though it was not required, "
                "or an animation was not produced despite it being required."
            )
        return None


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


class SlideWithList(BeamerSlide):
    def __init__(
        self,
        title: str,
        subtitle: Union[None, str],
        beamer_list: BeamerList,
        width_buffer: float = 3.0,
        height_buffer: float = 1.0,
    ):
        super().__init__(
            title=title,
            subtitle=subtitle,
            width_buffer=width_buffer,
            height_buffer=height_buffer,
        )
        self.beamer_list: BeamerList = beamer_list

    def construct(self):
        self.draw(ORIGIN, 1.0, target_scene=self)

    def draw(
        self,
        origin=None,
        scale: float = 1.0,
        target_scene: Union[None, Slide] = None,
        animate=True,
        animate_camera=True,
    ) -> VGroup:
        if target_scene is None:
            target_scene = self

        if origin is None:
            origin = ORIGIN

        content, animations = self.inner_draw(
            origin,
            scale,
            target_scene=target_scene,
            animate=animate,
            animate_camera=animate_camera,
        )
        # create the list object
        list_group = self.beamer_list.get_list(scale_factor=scale)
        buffer_with_prev_object = 0.5
        list_group.scale(scale_factor=0.75).next_to(
            content, DOWN, buff=buffer_with_prev_object * scale
        )
        content.add(list_group)
        if animate:
            camera_animation: Union[None, List[Animation]] = (
                self.move_and_fit_camera_to_content(
                    content=content,
                    target_scene=target_scene,
                    animate_camera=animate_camera,
                )
            )
            if camera_animation is not None:
                # move camera while writing the list
                camera_animation.append(Create(list_group))
                animations.append(AnimationGroup(*camera_animation))
            else:
                # only animate writing the list
                animations.append(Create(list_group))

            for animation in animations:
                target_scene.play(animation)
                target_scene.wait(1)
                target_scene.next_slide()
        else:
            target_scene.add(list_group)
        return content


def light_themed_table(table: Table) -> Table:
    """
    Apply a light theme to the table.

    Args:
        table: A manim Table object.

    Returns:
        The table with a light theme applied.
    """
    # make lines & text black
    table.get_col_labels().set_weight("bold")
    table.get_horizontal_lines().set_color(BLACK)
    table.get_vertical_lines().set_color(BLACK)
    for entry in table.get_entries():
        entry.set_color(BLACK)
    return table


class SlideWithTable(BeamerSlide):
    def __init__(
        self,
        title: str,
        subtitle: Union[None, str],
        table: Table,
        caption: str,
        highlighted_columns: List[int],
        width_buffer: float = 3.0,
        height_buffer: float = 1.0,
    ):
        super().__init__(
            title=title,
            subtitle=subtitle,
            width_buffer=width_buffer,
            height_buffer=height_buffer,
        )
        self.table: Table = light_themed_table(table)
        self.caption = caption
        self.highlighted_columns = highlighted_columns

    def construct(self):
        self.draw(ORIGIN, 1.0, target_scene=self)

    def draw(
        self, origin, scale: float, target_scene: Union[None, Slide], animate=True
    ) -> VGroup:
        if target_scene is None:
            target_scene = self
        content: VGroup = self.inner_draw(origin, scale, target_scene=target_scene)
        buffer_with_prev_object = 0.5
        table = self.table.copy()
        caption = self.default_m_object(self.caption, color=BLACK).scale(0.5)
        caption.next_to(table, DOWN, buff=0.5)
        captioned_table = VGroup(table, caption)
        captioned_table.scale(scale_factor=scale).next_to(
            content, DOWN, buff=buffer_with_prev_object * scale
        )
        content.add(captioned_table)
        if animate:
            target_scene.play(
                Write(captioned_table),
                target_scene.camera.frame.animate.move_to(content.get_center()).set(
                    width=content.width
                    + self.width_buffer,  # height=all_content.height + 2
                ),
            )
            animations = []
            for col_idx in self.highlighted_columns:
                animations.append(
                    Circumscribe(
                        table.get_columns()[col_idx],
                        color=MANIM_BLUE,
                        stroke_width=15 * scale,
                        run_time=1,
                    ),
                )
            if len(animations) > 0:
                target_scene.wait(1)
                target_scene.next_slide(loop=True)
                target_scene.play(AnimationGroup(*animations))
            target_scene.wait(1)
        else:
            target_scene.add(content)
        return content


class SlideWithTables(BeamerSlide):
    """
    A slide that shows multiple tables side by side.

    Identical copy to the above but with some minor changes.
    Duplicated here due to presentation deadline.
    """

    def __init__(
        self,
        title: str,
        subtitle: Union[None, str],
        tables: Table,
        captions: str,
        highlighted_columns: List[int],
        width_buffer: float = 3.0,
        height_buffer: float = 1.0,
    ):
        super().__init__(
            title=title,
            subtitle=subtitle,
            width_buffer=width_buffer,
            height_buffer=height_buffer,
        )
        self.tables: List[Table] = []
        for table in tables:
            self.tables.append(light_themed_table(table))
        self.captions = captions
        self.highlighted_columns = highlighted_columns

    def construct(self):
        self.draw(ORIGIN, 1.0, target_scene=self)

    def draw(
        self, origin, scale: float, target_scene: Union[None, Slide], animate=True
    ) -> VGroup:
        if target_scene is None:
            target_scene = self
        content: VGroup = self.inner_draw(origin, scale, target_scene=target_scene)
        len_of_titles = len(content)
        buffer_with_prev_object = 0.5
        captioned_tables: List[VGroup] = []
        prev_table = None
        for caption, table in zip(self.captions, self.tables):
            table_copy = table.copy()
            caption_text = self.default_m_object(caption, color=BLACK)
            caption_text.next_to(table_copy, DOWN, buff=0.5)
            captioned_table = VGroup(table_copy, caption_text)
            captioned_table.scale(scale_factor=scale).next_to(
                content, DOWN, buff=buffer_with_prev_object * scale
            )
            captioned_tables.append(captioned_table)
            if prev_table is not None:
                captioned_table.next_to(prev_table, RIGHT)
            content.add(captioned_table)
            prev_table = table_copy

        # adjust all tables to be centered beneath the title
        content[len_of_titles:].next_to(
            content[:len_of_titles], DOWN, buff=buffer_with_prev_object * scale
        )

        if animate:
            target_scene.play(
                AnimationGroup(
                    *[Write(captioned_table) for captioned_table in captioned_tables]
                ),
                target_scene.camera.frame.animate.move_to(content.get_center()).set(
                    width=content.width
                    + self.width_buffer,  # height=all_content.height + 2
                ),
            )
            # animations = []
            # for col_idx in self.highlighted_columns:
            #     animations.append(
            #         Circumscribe(
            #             table.get_columns()[col_idx], color=MANIM_BLUE,
            #             stroke_width=15 * scale, run_time=1
            #         ),
            #     )
            # if len(animations) > 0:
            #     target_scene.wait(1)
            #     target_scene.next_slide(loop=True)
            #     target_scene.play(AnimationGroup(*animations))
            target_scene.wait(1)
        else:
            target_scene.add(content)
        return content


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


class SlideDiagram(Slide):
    def __init__(self, path, caption, original_image_scale, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.caption = caption
        self.original_image_scale = original_image_scale
        self.captioned_jpg: CaptionedJPG = self.get_diagram()

    def construct(self, origin=ORIGIN, scale=1.0):
        self.draw(origin, scale, target_scene=self)

    def draw(self, origin, scale, target_scene=None, animate=True):
        self.captioned_jpg.draw(
            origin, scale, target_scene=target_scene, animate=animate
        )

    def get_diagram(self) -> CaptionedJPG:
        """
        Create a slide showing the diagram of the CEW systematic design process of NFNs.

        Returns:
            The slide with the diagram shown.
        """
        return CaptionedJPG(
            path=self.path,
            caption=self.caption,
            original_image_scale=self.original_image_scale,
        )
