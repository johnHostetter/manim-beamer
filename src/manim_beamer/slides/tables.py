from typing import List, Union

from manim import (
    BLACK,
    DOWN,
    ORIGIN,
    RIGHT,
    AnimationGroup,
    Circumscribe,
    Table,
    VGroup,
    Write,
)
from manim_slides import Slide

from manim_beamer import MANIM_BLUE
from manim_beamer.slides.base import BeamerSlide


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
