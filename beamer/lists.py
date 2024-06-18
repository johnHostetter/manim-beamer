from abc import abstractmethod

from manim import (
    VGroup,
    Text,
    Arrow,
    LEFT,
    RIGHT,
    DOWN,
    BLACK,
    MathTex,
    Cross,
    StealthTip,
)


class TextWithMath(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BeamerList:
    def __init__(self, items, font_size=30, list_color=BLACK):
        super().__init__()
        self.items = items
        self.font_size = font_size
        self._list_color = list_color
        self.max_allowed_lists = 3  # this includes the main list and all sublists
        self.item_vertical_spacing = 0.25  # vertical spacing between items in the list

    @property
    def list_color(self):
        return self._list_color

    @list_color.setter
    def list_color(self, value):
        self._list_color = value

    @list_color.deleter
    def list_color(self):
        del self._list_color

    @abstractmethod
    def get_item_marker(self, scale_factor: float):
        raise NotImplementedError("This method must be implemented in a subclass")

    def get_list(self, scale_factor: float, depth=0):
        # Create a VGroup to contain the items and item_markers
        list_group = VGroup()
        for index, item in enumerate(self.items):
            # default values for the font color and opacity of the item marker
            font_color = BLACK
            item_marker_opacity: float = 1.0 - (depth / (self.max_allowed_lists + 1))

            if isinstance(item, tuple):
                # if the item is a tuple, it should contain the text, font color, and
                # opacity of the item marker
                item, font_color, item_marker_opacity = item[0], item[1], item[2]
            if isinstance(item, str) or isinstance(item, VGroup):
                if isinstance(item, str):
                    # if the item is a string, create a Text object
                    text = Text(f"{item}", color=font_color, font_size=self.font_size)
                else:
                    text = self.parse_vgroup(font_color, item)

                item_marker = self.get_item_marker(
                    scale_factor=scale_factor,
                ).copy()
                # set the opacity of the item_marker based on the depth of the list
                item_marker.set_opacity(item_marker_opacity)
                item_marker.next_to(text, LEFT, buff=0.25)
                item_group = VGroup(text, item_marker)
            elif isinstance(item, BeamerList):
                item_group = item.get_list(
                    scale_factor=scale_factor,
                    depth=depth + 1,
                )
            else:
                raise ValueError(
                    f"Invalid item type {type(item)}. Must be a string or a BeamerList object"
                )
            list_group.add(item_group)

        # Arrange the items vertically, and appropriately indent if it's a sublist
        list_group.arrange(
            self.item_vertical_spacing * DOWN, aligned_edge=LEFT, buff=0.5
        )
        for m_object in list_group:
            if isinstance(m_object, VGroup) and not isinstance(m_object, TextWithMath):
                m_object.shift(
                    0.5 * RIGHT
                )  # shift only the nested list (not TextWithMath)
                for sub_object in m_object:
                    if isinstance(sub_object, VGroup) and not isinstance(
                        sub_object, TextWithMath
                    ):
                        sub_object.shift(
                            RIGHT
                        )  # shift only the nested list (not TextWithMath)

        return list_group

    def parse_vgroup(self, font_color, item):
        text = TextWithMath()
        if isinstance(item, Cross):
            # useful for substituting \cross with a cross symbol
            item.set_color(font_color)
            text.add(item)
            return text
        prev_sub_text = None
        for sub_item in item:
            if isinstance(sub_item, str):
                # if the item is a string, create a Text object
                sub_text = Text(f"{item}", color=font_color, font_size=self.font_size)
            elif isinstance(sub_item, Text) or isinstance(sub_item, MathTex):
                sub_text = sub_item
                sub_text.set_color(font_color)
                sub_text.set_font_size(self.font_size)
                if isinstance(sub_text, MathTex):
                    # math text is smaller than regular text
                    sub_text.set_font_size(self.font_size * 1.5)
            elif isinstance(sub_item, VGroup):
                # recursive call
                sub_text = self.parse_vgroup(font_color, sub_item)
            else:
                raise ValueError(
                    f"Invalid item type {type(item)}. Must be a string or a BeamerList object"
                )
            if prev_sub_text is not None:
                sub_text.next_to(prev_sub_text, RIGHT, buff=0.2)
            prev_sub_text = sub_text
            text.add(sub_text)
        return text


class ItemizedList(BeamerList):
    def get_item_marker(self, scale_factor: float = 1.0):
        return Arrow(
            LEFT * scale_factor,
            RIGHT * scale_factor,
            color=self.list_color,
            max_stroke_width_to_length_ratio=0.0,
            max_tip_length_to_length_ratio=(0.1 * scale_factor),
            stroke_opacity=0.1,
            tip_shape=StealthTip,
            buff=0,
        ).scale(0.1 * scale_factor)


class BulletedList(BeamerList):
    def get_item_marker(self, scale_factor: float = 1.0):
        return Text("•", color=self.list_color, font_size=self.font_size).scale(1.5)


class AdvantagesList(BeamerList):
    def get_item_marker(self, scale_factor: float = 1.0):
        return Text("+", color=self.list_color, font_size=self.font_size).scale(1.25)


class DisadvantagesList(BeamerList):
    def get_item_marker(self, scale_factor: float = 1.0):
        return Text("-", color=self.list_color, font_size=self.font_size).scale(1.25)
