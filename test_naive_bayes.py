__author__ = 'niklas'

import unittest
import naive_bayes as nb
import pandas as pd


class AccuracyTest(unittest.TestCase):
    def testAccuray(self):
        """"accuracy is calculated correctly"""
        predicted = pd.Series([0, 0, 0])
        real = pd.Series([0, 0, 1])
        self.assertEqual(nb.calc_accuracy(predicted, real), 1/3)


if __name__ == "__main__":
    unittest.main()