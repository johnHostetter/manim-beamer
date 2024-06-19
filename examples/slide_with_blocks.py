from pathlib import Path

from manim import config, WHITE, BLACK

from mbeamer.bibtex import BibTexManager
from mbeamer.slides import SlideWithBlocks
from mbeamer.blocks import AlertBlock, ExampleBlock
from mbeamer.lists import ItemizedList, AdvantagesList, DisadvantagesList

config.background_color = WHITE
light_theme_style = {
    "fill_color": BLACK,
    "background_stroke_color": WHITE,
}


def pros_and_cons() -> SlideWithBlocks:
    """
    Create a slide with two blocks: one for the advantages and one for the disadvantages of
    deep neural networks (DNNs).

    Returns:
        The slide with the two blocks.
    """
    bib = BibTexManager(path=Path(__file__).parent / "references.bib")

    example_block = ExampleBlock(
        title="Advantages of DNNs",
        content=AdvantagesList(
            items=[
                "Reliable",
                ItemizedList(
                    items=[
                        "Assist students in learning",
                        bib.slide_short_cite("abdelshiheed2023leveraging"),
                        "Predict septic shock",
                        bib.slide_short_cite("dqn_septic_shock"),
                        "Able to solve complex games (e.g., Go)",
                        bib.slide_short_cite("silver2016mastering"),
                    ]
                ),
                "Flexible",
                ItemizedList(
                    items=[
                        "Network morphism (i.e., neurogenesis)",
                        bib.slide_short_cite("draelos_neurogenesis_2016"),
                        bib.slide_short_cite("maile_when_2022"),
                    ]
                ),
                "Generalizable",
                ItemizedList(
                    items=[
                        "Supervised learning",
                        bib.slide_short_cite(
                            "bolat_interpreting_2020"
                        ),  # nfn paper that uses dnn
                        "Online reinforcement learning",
                        bib.slide_short_cite("jaderberg_reinforcement_2016"),
                        "Offline reinforcement learning",
                        bib.slide_short_cite("levine_offline_2020"),
                        "and more...",
                    ]
                ),
            ]
        ),
    )
    alert_block = AlertBlock(
        title="Disadvantages of DNNs",
        content=DisadvantagesList(
            items=[
                "Relies upon large quantities of data",
                bib.slide_short_cite("efficient_processing_of_dnns"),
                "Difficult to interpret (i.e., black-box)",
                bib.slide_short_cite("wang_explaining_2021"),
            ]
        ),
    )
    return SlideWithBlocks(
        title="Deep Neural Networks (DNNs)",
        subtitle=None,
        blocks=[example_block, alert_block],
    )


if __name__ == "__main__":
    beamer_slide = pros_and_cons()
    beamer_slide.render()
