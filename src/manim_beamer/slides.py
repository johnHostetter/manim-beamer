from typing import Union, List, Type

from manim import (
    ORIGIN,
    MovingCameraScene,
    FadeOut,
    Text,
    ITALIC,
    BOLD,
    VGroup,
    Succession,
    Write,
    Circumscribe,
    Group,
    Create,
    AnimationGroup,
    BLACK,
    UP,
    DOWN,
    RIGHT,
    Table,
    MathTex,
)
from manim_slides import Slide

from manim_beamer import MANIM_BLUE
from manim_beamer.blocks import Block
from manim_beamer.lists import BeamerList
from manim_beamer.images import CaptionedJPG


class SlideShow(Slide, MovingCameraScene):
    """
    A class to create a slide show of multiple Slide objects.
    """

    def __init__(self, slides, zoom_with_height: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.slides: List[Type[Slide]] = slides
        self.zoom_with_height: bool = zoom_with_height

    def construct(self):
        for slide in self.slides:
            # see what the content will be like in advance
            content = slide.draw(
                origin=ORIGIN, scale=1.0, target_scene=None, animate=False
            )
            if content is not None:
                # focus the camera on the entire slide
                self.camera.frame.move_to(content.get_center()).set(
                    width=content.width * 3.0,  # height=content.height + 3
                )
                # if self.zoom_with_height:
                #     self.camera.frame.set(height=content.height * 7.0)
            # draw the slide but ignore the returned content
            _ = slide.draw(origin=ORIGIN, scale=1.0, target_scene=self, animate=True)
            self.wait(1)
            self.next_slide()
            # fade out the slide content
            self.play(*[FadeOut(m_object) for m_object in self.mobjects])


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
        # # create the overall my_config of the slide
        # self.content: VGroup = VGroup(self.title_text)

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
    def __init__(
        self,
        title: str,
        subtitle: Union[None, str],
        width_buffer: float = 3.0,
        height_buffer: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title_str: str = title
        self.subtitle_str: str = subtitle
        self.width_buffer = width_buffer
        self.height_buffer = height_buffer

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

    def inner_draw(self, origin, scale, target_scene=None, animate=True) -> VGroup:
        """
        Draw the slide content (title and subtitle - if applicable) on the scene
        and then return the last displayed text object.

        Args:
            origin: The origin of the slide.
            scale: The scale factor to apply to the slide content.
            target_scene: The scene to draw the slide on. If None, the current scene is used.
            animate: Whether to animate the drawing of the slide.

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

        if animate:
            target_scene.wait(1)
            target_scene.next_slide()
            # position the camera correctly
            target_scene.play(
                Succession(
                    target_scene.camera.frame.animate.set(
                        width=content.width
                        + self.width_buffer,  # height=content.height + 1
                    ),
                    Write(title_text),
                )
            )
        else:
            target_scene.add(title_text)

        if subtitle_text is not None:
            if animate:
                target_scene.play(Write(subtitle_text))
                target_scene.wait(1)
                target_scene.next_slide()
            else:
                target_scene.add(subtitle_text)

        if animate:
            target_scene.wait(1)
            target_scene.next_slide()
        return content


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
        self, origin, scale: float, target_scene: Union[None, Slide], animate=True
    ) -> VGroup:
        if target_scene is None:
            target_scene = self
        content: VGroup = self.inner_draw(origin, scale, target_scene=target_scene)
        # create the list object
        list_group = self.beamer_list.get_list(scale_factor=scale)
        buffer_with_prev_object = 0.5
        list_group.scale(scale_factor=scale).next_to(
            content, DOWN, buff=buffer_with_prev_object * scale
        )
        content.add(list_group)
        if animate:
            target_scene.play(
                Create(list_group),
                self.camera.frame.animate.move_to(content.get_center()).set(
                    width=content.width + 2,  # height=all_content.height + 2
                ),
            )
            target_scene.wait(2)
            target_scene.next_slide()
            target_scene.wait(2)
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
        caption = Text(self.caption, color=BLACK).scale(0.5)
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
            caption_text = Text(caption, color=BLACK)
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
        blocks: List[Type[Block]],
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
        below: Union[None, Text, Block],
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
        content: VGroup = self.inner_draw(
            origin, scale, target_scene=target_scene, animate=animate
        )
        m_object_to_be_below = content
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
                    target_scene.play(Write(block))
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

        if animate:
            # focus the camera on the entire slide
            target_scene.play(
                target_scene.camera.frame.animate.move_to(content.get_center()).set(
                    height=content.height + self.height_buffer
                )
            )
            target_scene.wait(3)


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
