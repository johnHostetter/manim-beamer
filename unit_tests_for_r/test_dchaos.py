"""
Test that the Dchaos package written in R can be imported and used from Python.

The following may be necessary on Windows to set the 'R_HOME' for rpy2 correctly:

    from rpy2 import situation
    import os
    os.environ['R_HOME'] = situation.r_home_from_registry()
    situation.get_r_home()
"""
import unittest

from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr


class TestDChaos(unittest.TestCase):
    """
    Test the DChaos package written in R.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("DChaos")
        self.chaos_package = importr("DChaos")

    def test_dchaos(self) -> None:
        """
        Test the DChaos package. The following code is adapted from the
        DChaos package documentation. See:

            https://cran.r-project.org/web/packages/DChaos/DChaos.pdf

        Specifically, this is a reproduction of the example on page 3 of the
        documentation just beneath the dchaos function within the Details section.

        Returns:
            None
        """
        time_series_data = self.chaos_package.logistic_sim(a=4, n=1000)
        help(self.chaos_package.embedding)
        # in the following, m=5 is the embedding dimension and lag=2 is the time lag
        data = self.chaos_package.embedding(time_series_data, m=5, lag=2, timelapse="FIXED")
        print(data)
