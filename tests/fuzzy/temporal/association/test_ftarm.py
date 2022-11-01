import unittest
import numpy as np

from soft.fuzzy.temporal.association.ftarm import FTARM


def make_example():
    trans_1 = [('A', 5), ('C', 4)]
    trans_2 = [('A', 3), ('B', 2)]
    trans_3 = [('C', 4)]
    trans_4 = [('A', 3), ('B', 2), ('D', 4)]
    trans_5 = [('A', 3), ('B', 5), ('D', 4), ('E', 2)]
    P_1 = [trans_1, trans_2, trans_3]
    P_2 = [trans_4, trans_5]
    P = [P_1, P_2]
    D = P  # D should be a dictionary, but this is simpler to work with

    term_1 = {'label': 'LOW', 'a': 0.0, 'b': 3.0, 'c': 6.0}
    term_2 = {'label': 'MODERATE', 'a': 3.0, 'b': 6.0, 'c': 9.0}
    term_3 = {'label': 'HIGH', 'a': 6.0, 'b': 9.0, 'c': 11.0}
    terms = [term_1, term_2, term_3]
    return P, D, terms


class TestFTARM(unittest.TestCase):
    def test_step_1(self):
        P, D, terms = make_example()
        ftarm = FTARM(P, D, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
        actual_output = ftarm._FTARM__step_1()
        with open('data/step_1_output.npy', 'rb') as file:
            expected_output = np.load(file)
        assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()

    def test_step_2(self):
        P, D, terms = make_example()
        ftarm = FTARM(P, D, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
        f = ftarm._FTARM__step_1()
        actual_output = ftarm._FTARM__step_2(f)
        with open('data/step_2_output.npy', 'rb') as file:
            expected_output = np.load(file)
        assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()

    def test_step_3(self):
        P, D, terms = make_example()
        ftarm = FTARM(P, D, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
        f = ftarm._FTARM__step_1()
        count = ftarm._FTARM__step_2(f)
        actual_output = ftarm._FTARM__step_3(count)
        with open('data/step_3_output.npy', 'rb') as file:
            expected_output = np.load(file)
        assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()

    def test_step_4(self):
        P, D, terms = make_example()
        ftarm = FTARM(P, D, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
        f = ftarm._FTARM__step_1()
        count = ftarm._FTARM__step_2(f)
        tFuzzySupport = ftarm._FTARM__step_3(count)
        actual_output = ftarm._FTARM__step_4(tFuzzySupport)
        with open('data/step_4_output.npy', 'rb') as file:
            expected_output = np.load(file)
        assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()
