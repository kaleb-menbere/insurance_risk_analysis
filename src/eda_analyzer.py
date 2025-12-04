import pandas as pd
import numpy as np

class InsuranceEDA:
    """
    A class to perform Exploratory Data Analysis and Statistical Summaries.
    """
    
    def __init__(self, df):
        self.df = df

    def get_data_summary(self):
        """Returns descriptive statistics for numerical columns."""
        return self.df.describe()

    def check_missing_values(self):
        """Returns the count and percentage of missing values."""
        missing = self.df.isnull().sum()
        percentage = (self.df.isnull().sum() / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': percentage})
        return missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage', ascending=False)

    def calculate_loss_ratio(self):
        """
        Calculates the Loss Ratio (TotalClaims / TotalPremium).
        Adds a new column 'LossRatio' to the dataframe.
        """
        # Avoid division by zero
        self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium'].replace(0, np.nan)
        return self.df

    def get_outliers_iqr(self, column):
        """
        Identifies outliers using the IQR method.
        Returns the subset of dataframe containing outliers.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers

    def aggregate_by_geography(self, geo_col='PostalCode'):
        """Aggregates Premium and Claims by geography."""
        agg_df = self.df.groupby(geo_col)[['TotalPremium', 'TotalClaims']].mean().reset_index()
        return agg_df