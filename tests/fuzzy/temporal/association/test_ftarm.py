import unittest
import numpy as np

from soft.fuzzy.temporal.association.ftarm import FTARM
from examples.fuzzy.temporal.association.ftarm.sample import make_example


class TestFTARM(unittest.TestCase):
    def test_step_1(self):
        dataframe, terms = make_example()
        ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
        actual_output = ftarm._FTARM__step_1()
        with open('data/step_1_output.npy', 'rb') as file:
            expected_output = np.load(file)
        assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()

    def test_step_2(self):
        dataframe, terms = make_example()
        ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
        f = ftarm._FTARM__step_1()
        actual_output = ftarm._FTARM__step_2(f)
        with open('data/step_2_output.npy', 'rb') as file:
            expected_output = np.load(file)
        assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()

    def test_step_3(self):
        dataframe, terms = make_example()
        ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
        f = ftarm._FTARM__step_1()
        count = ftarm._FTARM__step_2(f)
        actual_output = ftarm._FTARM__step_3(count)
        with open('data/step_3_output.npy', 'rb') as file:
            expected_output = np.load(file)
        assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()

    def test_step_4(self):
        dataframe, terms = make_example()
        ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
        f = ftarm._FTARM__step_1()
        count = ftarm._FTARM__step_2(f)
        tFuzzySupport = ftarm._FTARM__step_3(count)
        actual_output = ftarm._FTARM__step_4(tFuzzySupport)
        with open('data/step_4_output.npy', 'rb') as file:
            expected_output = np.load(file)
        assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()
