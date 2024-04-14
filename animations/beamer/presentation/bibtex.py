from pathlib import Path
from typing import Union

import bibtexparser
from bibtexparser.model import Entry

from soft.utilities.reproducibility import path_to_project_root


class BibTexManager:
    def __init__(self, path: Union[None, Path] = None):
        if path is None:
            path = path_to_project_root() / "animations" / "beamer" / "presentation" / "ref.bib"
        self.path: Path = path

        # We want to add three new middleware layers to our parse stack:
        layers = [
            bibtexparser.middlewares.MonthIntMiddleware(),
            # Months should be represented as int (0-12)
            bibtexparser.middlewares.SeparateCoAuthors(),  # Co-authors should be separated
            bibtexparser.middlewares.SplitNameParts()
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
        if len(entry["author"]) == 1:
            return entry["author"][0].last
        elif len(entry["author"]) == 2:
            return " and ".join([name_parts.last[0] for name_parts in entry["author"]])
        else:
            return entry["author"][0].last + " et al."

    @staticmethod
    def convert_entry_to_citation(entry: Entry) -> str:
        """
        Convert a bibtex entry to a citation string (for presentation slides).

        Args:
            entry: The bibtex entry.

        Returns:
            The citation string for the entry. Format is "[Author et al. (Year)]".
        """
        return f"[{BibTexManager.get_author_last_names_only(entry)} ({entry['year']})]"
