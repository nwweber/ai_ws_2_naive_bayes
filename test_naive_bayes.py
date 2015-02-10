__author__ = 'niklas'

import unittest
import naive_bayes as nb
import pandas as pd


class AccuracyTest(unittest.TestCase):
    def testAccuray(self):
        """"accuracy is calculated correctly"""
        predicted = pd.Series([0, 0, 0])
        real = pd.Series([0, 0, 1])
        self.assertEqual(nb.calc_accuracy(predicted, real), 1 / 3)


class BernoulliTest(unittest.TestCase):
    def testBernoulli(self):
        """Bernoulli pmf is correct"""
        p = 0.3
        self.assertEqual(nb.calc_bernoulli(1, p), p)
        self.assertEqual(nb.calc_bernoulli(0, p), (1 - p))


class NaiveBayesTest(unittest.TestCase):
    def setUp(self):
        self.classifier = nb.NBClassifier()

    def testCalculatePhiY(self):
        label_series = pd.Series([0, 0, 1])
        self.assertEqual(self.classifier._calc_phi_y(label_series), 1/5)

    def testCalculatePhiXY1(self):
        labels = pd.Series([0,1])
        df = pd.DataFrame([{"test0": 0, "test1": 1}], index=[0,1])
        df["the label"] = labels
        spam_count = labels.sum()
        phi_dict = {}
        phi_dict["test0"] = 1 / (spam_count+2)
        phi_dict["test1"] = 2 / (spam_count+2)
        test_dict = self.classifier._calculatePhiXY1(df,labels)
        self.assertEqual(len(phi_dict), len(test_dict))
        for key in phi_dict:
            self.assertEqual(phi_dict[key], test_dict[key])

if __name__ == "__main__":
    unittest.main()