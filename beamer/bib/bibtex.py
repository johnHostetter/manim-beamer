"""
Implements the necessary classes and features to process and handle .bib references.
"""

from pathlib import Path
from typing import Union, Tuple, List

from manim import DARK_BLUE

import bibtexparser
from bibtexparser.model import Entry


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
    def cite_short_entry(entry: Entry) -> str:
        """
        Convert a bibtex entry to a citation string (for presentation slides).

        Args:
            entry: The bibtex entry.

        Returns:
            The citation string for the entry. Format is "[Author et al. (Year)]".
        """
        return f"[{BibTexManager.get_author_last_names_only(entry)} ({entry['year']})]"

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
        self, key: str, item_marker_opacity: float = 0.0
    ) -> Tuple[str, str, float]:
        """
        Get the citation string for a bibtex entry in a format suitable for a slide using
        a BeamerList.

        Args:
            key: The key of the bibtex entry.
            item_marker_opacity: The opacity of the item marker within the BeamerList.

        Returns:
            The citation string for the entry. Format is "[Author et al., Year]".
        """
        return self.cite_short_entry(self[key]), DARK_BLUE, item_marker_opacity
