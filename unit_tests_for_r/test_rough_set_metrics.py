"""
Test that the RoughSets package written in R can be imported and used from Python.
"""
import unittest

from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr, data


class TestRoughSets(unittest.TestCase):
    """
    Test the RoughSets package written in R.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages('RoughSets')
        self.rs_package = importr('RoughSets')
        self.rs_data = data(self.rs_package).fetch("RoughSetData")["RoughSetData"]
        print(f"Available datasets: {list(self.rs_data.names)}")

    def test_rough_sets(self) -> None:
        """
        Test the rough sets package. The following code is adapted from the
        RoughSets package documentation. See:
            https://cran.r-project.org/web/packages/RoughSets/RoughSets.pdf

        Specifically, this is a reproduction of the example on page 20 of the
        documentation for the BC.boundary.reg.RST function.

        Note that typically the RST functions take a data.frame as input, but
        the data.frame is not a native Python object. Instead, we use the
        SF_asDecisionTable function to convert the data.frame into a native
        Python object, which is a list of lists. The first list is the header
        and the remaining lists are the rows of the data.frame.

        Also, the RST functions typically take a feature set as input, but
        the feature set is not a native Python object. Instead, we use the
        r['c'] function to convert a Python list into an R vector.

        Lastly, most of the functions are nested within the RoughSets package using
        a dot notation. For example, the BC.IND.relation.RST function is nested
        within the BC function. To access the BC.IND.relation.RST function in Python,
        we replace the dot notation with underscores. For example, the following:

            BC_IND_relation_RST

        is equivalent to the following:

            self.rs_package.BC.IND.relation.RST

        where self.rs_package is the RoughSets package imported from R. This is
        because the RoughSets package is imported as a Python object, which is
        a dictionary. The keys of the dictionary are the functions in the
        RoughSets package. The values of the dictionary are the functions in the
        RoughSets package as Python objects.

        Returns:
            None
        """
        rs_hiring_data = self.rs_data[0]  # index 0 is the hiring.dt dataset
        rs_hiring_table = self.rs_package.SF_asDecisionTable(rs_hiring_data)
        indiscernibility_relation = self.rs_package.BC_IND_relation_RST(
            rs_hiring_table, feature_set=robjects.r['c'](2)
        )
        roughset = self.rs_package.BC_LU_approximation_RST(
            rs_hiring_data, indiscernibility_relation
        )
        print(f"Possible approximations for the rough set: {list(roughset.names)}")
        # the zero index refers to the first option, which is the lower approximation
        print(f"Possible outcomes for the lower approximation: {list(roughset[0].names)}")
        # the indexing of [0][0] refers to the first option, which is Accept
        # and the indexing of [0][1] refers to the second option, which is Reject
        self.assertTrue(list(roughset[0][0]), [2, 3, 4])  # Accept
        self.assertTrue(list(roughset[0][1]), [2, 3, 4])  # Reject

        pos_boundary = self.rs_package.BC_boundary_reg_RST(
            rs_hiring_data, roughset
        )
        print(f"Possible variables for the positive boundary: {list(pos_boundary.names)}")
        self.assertTrue(list(pos_boundary[0]), [1, 7])
        self.assertTrue(pos_boundary[1][0], 0.25)
