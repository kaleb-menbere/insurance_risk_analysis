import pandas as pd
import numpy as np

class InsuranceDataLoader:
    """
    A class to handle loading and basic processing of insurance data.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Loads data from CSV."""
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
        except FileNotFoundError:
            print("Error: File not found.")
        return self.df

    def optimize_types(self):
        """
        Optimizes data types: 
        - Converts date columns to datetime objects.
        - Ensures categorical columns are optimal.
        """
        if self.df is None:
            print("Data not loaded.")
            return

        # List of potential date columns based on the problem description
        date_cols = ['TransactionMonth', 'VehicleIntroDate']
        
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Ensure numerical columns are numeric
        num_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm']
        for col in num_cols:
             if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        print("Data types optimized.")
        return self.df