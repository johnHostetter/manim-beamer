from typing import Any, List, Tuple, Union

import numpy as np
from manim import (
    BLACK,
    BOLD,
    DOWN,
    ITALIC,
    UP,
    Animation,
    MovingCameraScene,
    SVGMobject,
    Text,
    VGroup,
    Write,
)
from manim_slides import Slide


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
