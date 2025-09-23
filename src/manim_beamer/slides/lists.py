from typing import List, Union

from manim import DOWN, ORIGIN, Animation, AnimationGroup, Create, VGroup
from manim_slides import Slide

from manim_beamer.lists import BeamerList
from manim_beamer.slides.base import BeamerSlide


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
