import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../A-B-Hypothesis-Testing/")))

from scripts import data_cleaner

df_g = pd.read_csv("./tests/sample_data.csv")

class TestDataClean(unittest.TestCase):
    def setUp(self) -> None:
        self.df = df_g.copy()
        self.cleaner = data_cleaner.DataCleaner()
    
    def test_date_to_day(self):
        self.df = df_g.copy()
        self.assertEqual(
            [str(x) for x in self.cleaner.date_to_day(self.df)['day_of_week'].tolist()],
            ['Friday', 'Tuesday', 'Sunday', 'Friday', 'Friday']
        )

    def test_drop_unresponsive(self):
        self.df = df_g.copy()
        self.assertEqual(self.cleaner.drop_unresponsive(self.df).shape, (1,9))
    
    def test_drop_columns(self):
        self.df = df_g.copy()
        self.assertEqual(self.cleaner.drop_columns(self.df).shape, (5,4))
    
    def test_merge_response_columns(self):
        self.df = df_g.copy()
        self.assertEqual(self.cleaner.merge_response_columns(self.df)['response'].tolist(), [1, 1, 0, 1, 1])
    def test_convert_to_brands(self):
        self.df = df_g.copy()
        self.assertEqual(self.cleaner.convert_to_brands(self.df)['brand'].tolist(),
         ['generic', 'generic', 'generic', 'samsung', 'generic'])


if __name__ == "__main__":
    unittest.main()