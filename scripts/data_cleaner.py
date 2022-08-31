import pandas as pd
import numpy as np

class DataCleaner():
    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This checkes if there are any duplicated entries for a user
        And remove the duplicated rows
        """
        df = df.drop_duplicates(subset='auction_id')

        return df

    def date_to_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This converts the date column into the day of the week
        """
        df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name().values

        return df

    def drop_unresponsive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This drops rows where users didn't repond to the questioneer.
        Meaning, rows where both yes and no columns have 0
        """
        df = df.query("yes==1 | no==1")

        return df

    def drop_columns(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Drops columns that are not essesntial for modeling
        """
        if not columns:
            columns = ['auction_id', 'date', 'yes', 'no', 'device_make']
        df.drop(columns=columns, inplace=True)

        return df

    def merge_response_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This merges the one-hot-encoded target columns into
        a single column named response, and drop the yes and no columns
        """
        df['response'] = [1] * df.shape[0]
        df.loc[df['no'] == 1, 'response'] = 0

        return df

    def convert_to_brands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This converts the device model column in to 
        `known` and `generic` brands. It then removes
        the device_make column.
        """
        known_brands = ['samsung', 'htc', 'nokia',
                        'moto', 'lg', 'oneplus',
                        'iphone', 'xiaomi', 'huawei',
                        'pixel']
        makers = ["generic"]*df.shape[0]
        for idx, make in enumerate(df['device_make'].values):
            for brand in known_brands:
                if brand in make.lower():
                    makers[idx] = "known brand"
                    break
        df['brand'] = makers

        return df

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This runs a series of cleaner methods on the df passed to it. 
        """
        df = self.drop_duplicates(df)
        df = self.drop_unresponsive(df)
        df = self.date_to_day(df)
        df = self.convert_to_brands(df)
        df = self.merge_response_columns(df)
        df = self.drop_columns(df)
        df.reset_index(drop=True, inplace=True)

        return df