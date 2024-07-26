from typing import Union
from abc import abstractmethod

from manim import (
    WHITE,
    BLACK,
    DOWN,
    LEFT,
    VGroup,
    SurroundingRectangle,
    LaggedStart,
    Create,
    Text,
    ManimColor,
    Title,
    config,
)

from manim_beamer.lists import BeamerList

config.background_color = WHITE
light_theme_style = {
    "fill_color": BLACK,
    "background_stroke_color": WHITE,
}


class BlockTitle(Title):
    def __init__(
        self, text, underline_color: str, underline_thickness: float = 4.0, **kwargs
    ):
        super().__init__(text, **kwargs)
        # override the default underline color from white to #bf0040
        self.underline.set_color(ManimColor(underline_color))
        self.underline.set_stroke(width=underline_thickness)
        # Access the main text and set its alignment to left
        main_text = self.submobjects[0]
        main_text.align_to(self.get_left(), LEFT)


class Block:
    def __init__(self, title: Union[None, str], content: Union[str, BeamerList]):
        if title is None or isinstance(title, str):
            # automatically convert the str title to a RemarkTitle object
            self.title_str: Union[None, str] = title
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

        # create the manim objects for the block
        # these objects are created in the update_position_and_scale method
        # they are not created here to avoid scaling issues
        self.title = None
        self.text_group = None
        self.block_background = None

        # some settings to control the amount of buffer between the elements
        self.title_header_buff = 0.5
        self.content_block_buff = 0.65

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

    def update_position_and_scale(self, scale_factor: float) -> None:
        if (
            self.title is None
            and self.text_group is None
            and self.block_background is None
        ):
            if self.title_str is not None:
                self.title = BlockTitle(
                    self.title_str,
                    underline_color=self.get_foreground_color(),
                    underline_thickness=4.0 * scale_factor,
                    color=ManimColor(self.get_foreground_color()),
                    underline_buff=0.1,
                )

            content = self.content
            if isinstance(self.content, BeamerList):
                content = self.content.get_list(scale_factor=scale_factor)

            if self.title is not None:
                content.next_to(self.title, (DOWN * scale_factor))
                self.text_group = VGroup(self.title, content)
            else:
                self.text_group = VGroup(content)
            # left align the title and content
            self.text_group.arrange((DOWN * scale_factor), aligned_edge=LEFT, buff=0.25)
            # create a surrounding rectangle around the title and content
            self.block_background = SurroundingRectangle(
                self.text_group,
                color=ManimColor(self.get_foreground_color()),
                fill_color=ManimColor(self.get_background_color()),
                fill_opacity=1,
                corner_radius=0.25,
                stroke_width=4.0 * scale_factor,
                buff=0.1,  # controls the top and bottom buffer
            ).scale(
                1.025
            )  # controls the left and right buffer

    def get_vgroup(self) -> VGroup:
        return VGroup(self.block_background, self.text_group)

    def get_animation(
        self, scale_factor: float, below=None, animate=True
    ) -> Union[LaggedStart, VGroup]:
        title_header_buff = self.title_header_buff  # * scale_factor
        content_block_buff = self.content_block_buff  # * scale_factor
        self.update_position_and_scale(scale_factor)

        if below is None:
            block_background = self.block_background.scale(scale_factor)
            text_group = self.text_group.scale(scale_factor)
            if animate:
                return LaggedStart(
                    Create(block_background),
                    Create(text_group),
                )
            else:
                return VGroup(block_background, text_group)
        # scale the block FIRST before positioning it below the previous block
        # otherwise it will not be positioned correctly
        elif isinstance(below, Block):
            block_background = self.block_background.scale(scale_factor).next_to(
                below.block_background,
                (DOWN * scale_factor),
                buff=title_header_buff,
            )
            text_group = self.text_group.scale(scale_factor).next_to(
                below.block_background,
                (DOWN * scale_factor),
                buff=content_block_buff,
            )
            if animate:
                return LaggedStart(
                    Create(block_background),
                    Create(text_group),
                )
            else:
                return VGroup(block_background, text_group)
        else:  # e.g. isinstance(below, Text)
            block_background = self.block_background.scale(scale_factor).next_to(
                below, (DOWN * scale_factor), buff=title_header_buff
            )
            text_group = self.text_group.scale(scale_factor).next_to(
                below, (DOWN * scale_factor), buff=content_block_buff
            )
            if animate:
                return LaggedStart(
                    Create(block_background),
                    Create(text_group),
                )
            else:
                return VGroup(block_background, text_group)


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
