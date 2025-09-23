"""
The LaTeX beamer slides module.
"""

from .base import BeamerSlide
from .blocks import SlideWithBlocks
from .deck import SlideShow
from .diagram import SlideDiagram
from .lists import SlideWithList
from .prompt import PromptSlide
from .tables import SlideWithTable, SlideWithTables

__all__ = [
    "BeamerSlide",
    "SlideWithBlocks",
    "SlideShow",
    "SlideDiagram",
    "SlideWithList",
    "PromptSlide",
    "SlideWithTable",
    "SlideWithTables",
]
