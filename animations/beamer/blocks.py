from typing import Union as U
from abc import abstractmethod

from manim import *

from animations.beamer.lists import BeamerList

config.background_color = WHITE
light_theme_style = {
    "fill_color": BLACK,
    "background_stroke_color": WHITE,
}


class BlockTitle(Title):
    def __init__(self, text, underline_color: str, **kwargs):
        super().__init__(text, **kwargs)
        # override the default underline color from white to #bf0040
        self.underline.set_color(ManimColor(underline_color))
        # Access the main text and set its alignment to left
        main_text = self.submobjects[0]
        main_text.align_to(self.get_left(), LEFT)


class Block:
    def __init__(self, title: str, content: U[str, BeamerList]):
        if isinstance(title, str):
            # automatically convert the str title to a RemarkTitle object
            self.title = BlockTitle(
                title,
                underline_color=self.get_foreground_color(),
                color=ManimColor(self.get_foreground_color()),
                underline_buff=0.1,
            )
        else:
            assert isinstance(
                title, Title
            ), "The argument 'title' must be a 'string' or a 'Title' object"

        self.content = content
        if isinstance(content, str):
            # automatically convert the str content to a Text object
            self.content = Text(
                content, font="TeX Gyre Termes", color=BLACK, font_size=30
            )
        elif isinstance(content, BeamerList):
            self.update_beamer_list_color(content)
            self.content = content.get_list()

        # begin positioning the content below the title
        self.content.next_to(self.title, DOWN)
        self.text_group = VGroup(self.title, self.content)
        # left align the title and content
        self.text_group.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        # create a surrounding rectangle around the title and content
        self.block_background = SurroundingRectangle(
            self.text_group,
            color=ManimColor(self.get_foreground_color()),
            fill_color=ManimColor(self.get_background_color()),
            fill_opacity=1,
            corner_radius=0.25,
            buff=0.1,  # controls the top and bottom buffer
        ).scale(
            1.025
        )  # controls the left and right buffer

    @abstractmethod
    def get_foreground_color(self) -> str:
        raise NotImplementedError("This method must be implemented in a subclass")

    @abstractmethod
    def get_background_color(self) -> str:
        raise NotImplementedError("This method must be implemented in a subclass")

    def update_beamer_list_color(self, beamer_list: BeamerList) -> None:
        beamer_list.list_color = self.get_foreground_color()
        for item in beamer_list.items:
            if isinstance(item, BeamerList):
                item.list_color = self.get_foreground_color()
                self.update_beamer_list_color(item)

    def get_vgroup(self) -> VGroup:
        return VGroup(self.block_background, self.text_group)

    def get_animation(self, scale: float, below=None) -> LaggedStart:
        if below is None:
            return LaggedStart(
                Create(self.block_background),  # .scale(scale)),
                Create(self.text_group),  # .scale(scale)),
            )
        elif isinstance(below, Block):
            return LaggedStart(
                Create(
                    self.block_background.next_to(
                        below.block_background, DOWN, buff=0.5
                    )  # .scale(scale)
                ),
                Create(
                    self.text_group.next_to(
                        below.block_background, DOWN, buff=0.65
                    )  # .scale(scale)
                ),
            )
        else:  # e.g. isinstance(below, Text)
            return LaggedStart(
                Create(
                    self.block_background.next_to(
                        below, DOWN, buff=0.5
                    )  # .scale(scale)
                ),
                Create(
                    self.text_group.next_to(below, DOWN, buff=0.65)
                ),  # .scale(scale)),
            )


class RemarkBlock(Block):
    def get_foreground_color(self) -> str:
        return "#bf0040"

    def get_background_color(self) -> str:
        return "#f9e6ec"


class ExampleBlock(Block):
    def get_foreground_color(self) -> str:
        return "#007f5f"

    def get_background_color(self) -> str:
        return "#e5f9f6"


class AlertBlock(Block):
    def get_foreground_color(self) -> str:
        # return "#ffa600"  # orange yellow
        return "#ff0f20"

    def get_background_color(self) -> str:
        # return "#fff2e6"  # orange yellow
        return "#ffe6e6"
