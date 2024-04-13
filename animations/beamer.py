from abc import abstractmethod

from manim import *


config.background_color = WHITE
light_theme_style = {
    "fill_color": BLACK,
    "background_stroke_color": WHITE,
}


class BeamerList:
    def __init__(self, items, font_size=30, list_color=BLACK):
        super().__init__()
        self.items = items
        self.font_size = font_size
        self._list_color = list_color
        self.arrow = Arrow(
            LEFT,
            RIGHT,
            color=self.list_color,
            max_stroke_width_to_length_ratio=0.0,
            stroke_opacity=0.1
        ).scale(0.1)
        self.max_allowed_lists = 3  # this includes the main list and all sublists

    @property
    def list_color(self):
        return self._list_color

    @list_color.setter
    def list_color(self, value):
        self._list_color = value
        self.arrow = Arrow(
            LEFT,
            RIGHT,
            color=self.list_color,
            max_stroke_width_to_length_ratio=0.0,
            stroke_opacity=0.1
        ).scale(0.1)

    @list_color.deleter
    def list_color(self):
        del self._list_color

    def get_list(self, depth=0):
        # Create a VGroup to contain the items and arrows
        list_group_lst = []
        list_group = VGroup()
        for index, item in enumerate(self.items):
            if isinstance(item, str):
                text = Text(f"{item}", color=BLACK, font_size=self.font_size)
                arrow = self.arrow.copy()
                arrow.set_opacity(1.0 - (depth / (self.max_allowed_lists + 1)))
                arrow.next_to(text, LEFT, buff=0.25)
                item_group = VGroup(text, arrow)
                # item_group.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
                list_group_lst.append(item_group)
            elif isinstance(item, BeamerList):
                item_group = item.get_list(depth=depth + 1)
                # item_group = item
            else:
                raise ValueError("Invalid item type. Must be a string or a BeamerList object")
            list_group.add(item_group)

        # Arrange the items vertically
        list_group.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        for m_object in list_group:
            if isinstance(m_object, VGroup):
                m_object.shift(0.5 * RIGHT)
                # m_object[-1].set_opacity(0.75)
                for sub_object in m_object:
                    if isinstance(sub_object, VGroup):
                        sub_object.shift(RIGHT)
                        # sub_object[-1].set_opacity(0.5)

        return list_group


class BlockTitle(Title):
    def __init__(self, text, underline_color: str, **kwargs):
        super().__init__(text, **kwargs)
        # override the default underline color from white to #bf0040
        self.underline.set_color(ManimColor(underline_color))
        # Access the main text and set its alignment to left
        main_text = self.submobjects[0]
        main_text.align_to(self.get_left(), LEFT)


class Block:
    def __init__(self, title: str, content: str):
        if isinstance(title, str):
            # automatically convert the str title to a RemarkTitle object
            self.title = BlockTitle(
                title,
                underline_color=self.get_foreground_color(),
                color=ManimColor(self.get_foreground_color()),
                underline_buff=0.1
            )
        else:
            assert isinstance(title, Title), \
                "The argument \'title\' must be a \'string\' or a \'Title\' object"

        self.content = content
        if isinstance(content, str):
            # automatically convert the str content to a Text object
            self.content = Text(content, font="TeX Gyre Termes", color=BLACK, font_size=30)
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
            buff=0.1  # controls the top and bottom buffer
        ).scale(1.025)  # controls the left and right buffer

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

    def add_to_scene(self, scene: Scene) -> None:
        # order matters, add the block background first otherwise the text will be hidden
        scene.add(self.block_background)
        scene.add(self.text_group)


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
