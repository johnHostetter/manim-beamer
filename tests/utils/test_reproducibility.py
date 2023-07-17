"""
Test the functions related to reproducibility.
"""
import unittest

import numpy as np

from soft.utils.reproducibility import load_configuration, parse_configuration


class TestConfiguration(unittest.TestCase):
    """
    Test the functions related to the configuration settings.
    """
    def test_load_configuration(self) -> None:
        """
        Test that the configuration is loaded correctly.

        Returns:
            None
        """
        config = load_configuration(convert_data_types=False)
        self.assertEqual(config.fuzzy.t_norm.yager, "golden")

    def test_parse_configuration(self) -> None:
        """
        Test that the configuration is parsed correctly. For example, the user may specify the string "golden"
        for the Yager t-norm, but the function should convert this to the Golden ratio.

        Returns:
            None
        """
        config = load_configuration(convert_data_types=False)
        config = parse_configuration(config)
        self.assertEqual(  # by default, the Golden ratio is used
            config.fuzzy.t_norm.yager, (1 + 5**0.5) / 2
        )

    def test_modified_configuration(self) -> None:
        """
        Test that the configuration is modified when the user specifies a different value for the
        Yager t-norm. For example, the user may specify the Euler number instead of the Golden
        ratio.

        Returns:
            None
        """
        config = load_configuration(convert_data_types=False)
        for euler_spelling in ["euler", "Euler", "EULER"]:
            with config.unfreeze():
                config.fuzzy.t_norm.yager = euler_spelling
            config = parse_configuration(config)
            self.assertEqual(  # by default, the Golden ratio is used
                config.fuzzy.t_norm.yager, np.e
            )
