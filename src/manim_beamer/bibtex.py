"""
Implements the necessary classes and features to process and handle .bib references.
"""

from pathlib import Path
from typing import Union, Tuple, List, NamedTuple

from manim import DARK_BLUE, Tex, Text, RIGHT, DOWN, TexTemplate
from manim.utils.tex import _DEFAULT_PREAMBLE

import bibtexparser
from bibtexparser.model import Entry


class CitedTex(Tex):
    """
    This class is for LaTeX text that has been cited. It is assumed that there is no punctuation used.
    """

    template = TexTemplate(
        documentclass="\documentclass[preview]{standalone}",
        preamble=_DEFAULT_PREAMBLE + r"""\usepackage{ragged2e}\usepackage{adjustbox}""",
    )

    def __init__(
        self, *tex_strings, arg_separator="", tex_environment="center", **kwargs
    ):
        super().__init__(
            *tex_strings,
            arg_separator=arg_separator,
            tex_environment=tex_environment,
            tex_template=CitedTex.template,
            **kwargs,
        )
        for tex_string in self[1:]:
            tex_string.set_color(DARK_BLUE)


class BibTexManager:
    """
    The BibTexManager will allow convenient management, access, query and display of references
    stored in a .bib file.
    """

    def __init__(self, path: Path):
        """
        Given the path to a .bib file containing the references, an instance of this class will be
        created to efficiently manage and query it.
        """
        self.path: Path = path

        # We want to add three new middleware layers to our parse stack:
        layers = [
            bibtexparser.middlewares.MonthIntMiddleware(),
            # Months should be represented as int (0-12)
            bibtexparser.middlewares.SeparateCoAuthors(),  # Co-authors should be separated
            bibtexparser.middlewares.SplitNameParts(),
            # Names should be split into first, von, last, jr parts
        ]

        self.library = bibtexparser.parse_file(str(self.path), append_middleware=layers)

    def __getitem__(self, item: str):
        return self.get_entry_by_key(item)

    def get_entry_by_key(self, key: str) -> Union[None, Entry]:
        """
        Get a bibtex entry by its key. If the key is not found, return None.

        Args:
            key: The key of the entry to find.

        Returns:
            The entry if found, otherwise None.
        """
        for entry in self.library.entries:
            if entry.key == key:
                return entry
        return None

    @staticmethod
    def get_author_last_names_only(entry: Entry) -> str:
        """
        Get the last names of the authors of a bibtex entry. If there are more than two authors,
        only the first author's last name is returned followed by "et al.".

        Args:
            entry: The bibtex entry.

        Returns:
            The last names of the authors.
        """
        # the result is that within Entry, the author field is a list of NameParts objects

        # note: the author last name is still a list of strings, hence the [0] at the end
        if len(entry["author"]) == 1:
            return entry["author"][0].last[0].replace("{", "").replace("}", "")
        if len(entry["author"]) == 2:
            return (
                " and ".join([name_parts.last[0] for name_parts in entry["author"]])
                .replace("{", "")
                .replace("}", "")
            )
        return entry["author"][0].last[0] + " et al.".replace("{", "").replace("}", "")

    @staticmethod
    def cite_short_entry_no_brackets(entry: Entry) -> str:
        """
        Convert a bibtex entry to a citation string (for presentation slides),
        but do not add the left & right square brackets.

        Args:
            entry: The bibtex entry.

        Returns:
            The citation string for the entry. Format is "Author et al. (Year)".
        """
        return f"{BibTexManager.get_author_last_names_only(entry)} ({entry['year']})"

    @staticmethod
    def cite_short_entry(entry: Entry) -> str:
        """
        Convert a bibtex entry to a citation string (for presentation slides).

        Args:
            entry: The bibtex entry.

        Returns:
            The citation string for the entry. Format is "[Author et al. (Year)]".
        """
        return f"[{BibTexManager.cite_short_entry_no_brackets(entry=entry)}]"

    @staticmethod
    def wrap_by_word(string_to_parse, num_of_words: int) -> str:
        """
        Return a string where \\n is inserted between every n words.
        https://www.reddit.com/r/learnpython/comments/4i2z4u/how_to_add_a_new_line_after_every_nth_word/

        Args:
            string_to_parse: The string to wrap.
            num_of_words: The number of words to wrap by.
        """
        a: List[str] = string_to_parse.split()
        wrapped_text: str = ""
        for i in range(0, len(a), num_of_words):
            wrapped_text += " ".join(a[i : i + num_of_words]) + "\n"

        return wrapped_text

    @staticmethod
    def cite_entry(entry: Entry, num_of_words: int = 6) -> str:
        """
        Convert a bibtex entry to a citation string.

        Args:
            entry: The bibtex entry.
            num_of_words: The number of words to wrap by.

        Returns:
            The citation string for the entry. Format is "Author et al. (Year)".
        """
        # cite the paper as "Paper title (Author et al., Year)"
        title = BibTexManager.wrap_by_word(
            entry["title"].replace("{", "").replace("}", ""), num_of_words=num_of_words
        )
        if "year" not in entry:
            return f"{title} ({BibTexManager.get_author_last_names_only(entry)})"
        return f"{title} ({BibTexManager.get_author_last_names_only(entry)}, {entry['year']})"

    def slide_short_cite(
        self, *keys: str, item_marker_opacity: float = 0.0
    ) -> List[str]:
        """
        Get the citation string for a bibtex entry in a format suitable for a slide using a BeamerList.

        Args:
            keys: The keys of the bibtex entries.
            item_marker_opacity: The opacity of the item marker within the BeamerList.

        Returns:
            The citation for the entry. Format is "[Author et al., Year]".
        """
        return [self.cite_short_entry_no_brackets(self[key]) for key in keys]

    def slide_short_cite_after_join_with_brackets(self, *keys: str) -> str:
        # the \\scalebox offers true geometric scaling, both width and height
        # single argument (e.g., \scalebox{0.75}) applies to both width and height
        # double argument (e.g., \scalebox{0.75}[2]) changes them differently
        # return r"\scalebox{0.75}{[" + ", ".join(self.slide_short_cite(*keys)) + "]}"

        # however, the above does not work with the justifying environment, and won't allow line breaks
        # therefore, I scale only font size (with baseline alignment preserved);
        # first arg = font size (pt) & second arg = line spacing (baseline skip)

        return (
            r"{\fontsize{8}{9}\selectfont ["
            + ", ".join(self.slide_short_cite(*keys))
            + "]}"
        )
