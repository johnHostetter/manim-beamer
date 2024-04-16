from abc import abstractmethod

from manim import *


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
    def get_item_marker(self, scale_factor: float, scale_down_text: bool = False):
        raise NotImplementedError("This method must be implemented in a subclass")

    def get_list(self, scale_factor: float, depth=0, scale_down_text: bool = False):
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
            if isinstance(item, str):
                text = Text(f"{item}", color=font_color, font_size=self.font_size)
                if scale_down_text:
                    # beamer blocks already scale the text down, so we only need to scale it down
                    # if using a list outside a block
                    text.scale(
                        scale_factor=scale_factor
                    )
                item_marker = self.get_item_marker(
                    scale_factor=scale_factor,
                    scale_down_text=scale_down_text,
                ).copy()
                # set the opacity of the item_marker based on the depth of the list
                item_marker.set_opacity(item_marker_opacity)
                buffer_with_marker = 0.25
                if scale_down_text:
                    buffer_with_marker *= scale_factor
                item_marker.next_to(text, LEFT, buff=buffer_with_marker)
                item_group = VGroup(text, item_marker)
            elif isinstance(item, BeamerList):
                item_group = item.get_list(
                    scale_factor=scale_factor, depth=depth + 1, scale_down_text=scale_down_text
                )
            else:
                raise ValueError(
                    "Invalid item type. Must be a string or a BeamerList object"
                )
            list_group.add(item_group)

        # Arrange the items vertically, and appropriately indent if it's a sublist
        item_vertical_spacing = self.item_vertical_spacing * DOWN
        aligned_edge = LEFT
        list_group_buff = 0.5

        if scale_down_text:
            # item_vertical_spacing *= scale_factor
            aligned_edge *= scale_factor
            list_group_buff *= scale_factor

        list_group.arrange(
            item_vertical_spacing, aligned_edge=aligned_edge, buff=list_group_buff
        )
        right_shift = RIGHT
        if scale_down_text:
            right_shift *= scale_factor
        for m_object in list_group:
            if isinstance(m_object, VGroup):
                m_object.shift(0.5 * right_shift)
                for sub_object in m_object:
                    if isinstance(sub_object, VGroup):
                        sub_object.shift(right_shift)

        return list_group


class ItemizedList(BeamerList):
    def get_item_marker(self, scale_factor: float = 1.0, scale_down_text: bool = False):
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
    def get_item_marker(self, scale_factor: float = 1.0, scale_down_text: bool = False):
        marker = Text("•", color=self.list_color, font_size=self.font_size).scale(1.5)
        if scale_down_text:
            marker.scale(scale_factor)
        return marker


class AdvantagesList(BeamerList):
    def get_item_marker(self, scale_factor: float = 1.0, scale_down_text: bool = False):
        marker = Text("+", color=self.list_color, font_size=self.font_size).scale(1.25)
        if scale_down_text:
            marker.scale(scale_factor)
        return marker


class DisadvantagesList(BeamerList):
    def get_item_marker(self, scale_factor: float = 1.0, scale_down_text: bool = False):
        marker = Text("-", color=self.list_color, font_size=self.font_size).scale(1.25)
        if scale_down_text:
            marker.scale(scale_factor)
        return marker