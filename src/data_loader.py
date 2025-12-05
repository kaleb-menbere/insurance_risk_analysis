import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class InsuranceDataLoader:
    """Load and preprocess insurance data"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or "data/raw/MachineLearningRating_v3.txt"
        self.data = None
        
    def load_data(self):
        """Load the insurance data from pipe-separated text file"""
        try:
            # Load the data with pipe separator
            self.data = pd.read_csv(
                self.data_path, 
                sep='|', 
                encoding='utf-8',
                parse_dates=['TransactionMonth'],
                dayfirst=False,  # The sample shows YYYY-MM-DD format
                low_memory=False
            )
            
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Convert column names to lowercase with underscores for consistency
            self.data.columns = [col.strip().lower().replace(' ', '_') for col in self.data.columns]
            
            # Initial data cleaning
            self._clean_data()
            
            return self.data
            
        except FileNotFoundError:
            print(f"Data file not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def _clean_data(self):
        """Perform initial data cleaning"""
        print("Performing initial data cleaning...")
        
        # Replace empty strings and special characters with NaN
        self.data = self.data.replace(['', ' ', '  ', 'N/A', 'n/a', 'NA', 'na', 'NULL', 'null', '.', '-'], np.nan)
        
        # Clean specific columns
        if 'customvalueestimate' in self.data.columns:
            # Convert to numeric, coerce errors
            self.data['customvalueestimate'] = pd.to_numeric(self.data['customvalueestimate'], errors='coerce')
        
        if 'totalclaims' in self.data.columns:
            # Sample shows '.000000000000' which should be 0
            self.data['totalclaims'] = pd.to_numeric(self.data['totalclaims'], errors='coerce')
            # Fill NaN claims with 0 (assuming no claim)
            self.data['totalclaims'] = self.data['totalclaims'].fillna(0)
        
        if 'totalpremium' in self.data.columns:
            self.data['totalpremium'] = pd.to_numeric(self.data['totalpremium'], errors='coerce')
        
        # Clean categorical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.data[col].astype(str).str.strip()
            # Replace 'nan' strings with actual NaN
            self.data[col] = self.data[col].replace('nan', np.nan)
        
        print("Data cleaning completed.")
    
    def get_data_info(self):
        """Get basic information about the dataset"""
        if self.data is None:
            print("Please load data first using load_data()")
            return
        
        print("=" * 70)
        print("DATASET INFORMATION")
        print("=" * 70)
        
        # Basic info
        print(f"Shape: {self.data.shape}")
        print(f"Rows: {self.data.shape[0]:,}")
        print(f"Columns: {self.data.shape[1]}")
        
        # Data types
        print("\n" + "=" * 70)
        print("DATA TYPES SUMMARY")
        print("=" * 70)
        dtypes_summary = self.data.dtypes.value_counts()
        for dtype, count in dtypes_summary.items():
            print(f"{dtype}: {count} columns")
        
        # First few column names
        print(f"\nFirst 10 columns: {self.data.columns[:10].tolist()}")
        
        # Missing values
        print("\n" + "=" * 70)
        print("MISSING VALUES ANALYSIS")
        print("=" * 70)
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        }).sort_values('missing_percentage', ascending=False)
        
        # Show columns with missing values
        missing_cols = missing_df[missing_df['missing_count'] > 0]
        print(f"\nColumns with missing values: {len(missing_cols)}")
        
        if len(missing_cols) > 0:
            print("\nTop 20 columns by missing percentage:")
            print(missing_cols.head(20).to_string())
        
        # Memory usage
        print("\n" + "=" * 70)
        print("MEMORY USAGE")
        print("=" * 70)
        memory_mb = self.data.memory_usage(deep=True).sum() / 1024**2
        print(f"Total memory usage: {memory_mb:.2f} MB")
        
    def get_descriptive_stats(self):
        """Get descriptive statistics for numerical columns"""
        if self.data is None:
            print("Please load data first using load_data()")
            return
            
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        print("\n" + "=" * 70)
        print("DESCRIPTIVE STATISTICS (Numerical Columns)")
        print("=" * 70)
        print(f"Numerical columns found: {len(numerical_cols)}")
        
        if len(numerical_cols) > 0:
            stats = self.data[numerical_cols].describe().T
            stats['missing'] = self.data[numerical_cols].isnull().sum()
            stats['missing_pct'] = (stats['missing'] / len(self.data)) * 100
            
            # Add skewness and kurtosis
            stats['skew'] = self.data[numerical_cols].skew()
            stats['kurtosis'] = self.data[numerical_cols].kurt()
            
            print("\nKey financial metrics:")
            if 'totalclaims' in numerical_cols and 'totalpremium' in numerical_cols:
                total_claims = self.data['totalclaims'].sum()
                total_premium = self.data['totalpremium'].sum()
                loss_ratio = total_claims / total_premium if total_premium > 0 else np.nan
                print(f"Total Claims: R {total_claims:,.2f}")
                print(f"Total Premium: R {total_premium:,.2f}")
                print(f"Overall Loss Ratio: {loss_ratio:.2%}")
                print(f"Average Claim: R {self.data['totalclaims'].mean():,.2f}")
                print(f"Average Premium: R {self.data['totalpremium'].mean():,.2f}")
            
            return stats
        else:
            print("No numerical columns found.")
            return None
    
    def get_categorical_stats(self, top_n: int = 10):
        """Get statistics for categorical columns"""
        if self.data is None:
            print("Please load data first using load_data()")
            return
            
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        print("\n" + "=" * 70)
        print("CATEGORICAL COLUMNS SUMMARY")
        print("=" * 70)
        print(f"Categorical columns found: {len(categorical_cols)}")
        
        for col in categorical_cols[:20]:  # Limit to first 20 for brevity
            unique_vals = self.data[col].nunique()
            print(f"\n{col.upper()}:")
            print(f"  Unique values: {unique_vals}")
            print(f"  Missing values: {self.data[col].isnull().sum():,} ({(self.data[col].isnull().sum()/len(self.data)*100):.1f}%)")
            
            if unique_vals <= 20:
                print(f"  Value distribution:")
                value_counts = self.data[col].value_counts(dropna=False)
                for value, count in value_counts.items():
                    percentage = (count / len(self.data)) * 100
                    print(f"    {value}: {count:,} ({percentage:.1f}%)")
            else:
                print(f"  Top {top_n} values:")
                top_values = self.data[col].value_counts().head(top_n)
                for value, count in top_values.items():
                    percentage = (count / len(self.data)) * 100
                    print(f"    {value}: {count:,} ({percentage:.1f}%)")
    
    def get_time_period_info(self):
        """Get information about the time period covered"""
        if self.data is None or 'transactionmonth' not in self.data.columns:
            print("Transaction month data not available")
            return
        
        print("\n" + "=" * 70)
        print("TIME PERIOD ANALYSIS")
        print("=" * 70)
        
        min_date = self.data['transactionmonth'].min()
        max_date = self.data['transactionmonth'].max()
        date_range = max_date - min_date
        
        print(f"Data period: {min_date.date()} to {max_date.date()}")
        print(f"Total months: {(date_range.days / 30.44):.1f}")
        print(f"\nRecords per month:")
        monthly_counts = self.data['transactionmonth'].dt.to_period('M').value_counts().sort_index()
        print(monthly_counts.to_string())
        
        return monthly_counts
    
    def get_premium_claim_analysis(self):
        """Analyze premium and claim patterns"""
        if self.data is None:
            return None
        
        print("\n" + "=" * 70)
        print("PREMIUM AND CLAIM ANALYSIS")
        print("=" * 70)
        
        # Policies with vs without claims
        policies_with_claims = (self.data['totalclaims'] > 0).sum()
        policies_without_claims = (self.data['totalclaims'] == 0).sum()
        
        print(f"Policies with claims: {policies_with_claims:,} ({(policies_with_claims/len(self.data)*100):.1f}%)")
        print(f"Policies without claims: {policies_without_claims:,} ({(policies_without_claims/len(self.data)*100):.1f}%)")
        
        if policies_with_claims > 0:
            avg_claim_amount = self.data.loc[self.data['totalclaims'] > 0, 'totalclaims'].mean()
            max_claim = self.data['totalclaims'].max()
            print(f"\nFor policies with claims:")
            print(f"  Average claim amount: R {avg_claim_amount:,.2f}")
            print(f"  Maximum claim amount: R {max_claim:,.2f}")
        
        return {
            'with_claims': policies_with_claims,
            'without_claims': policies_without_claims,
            'claim_rate': policies_with_claims / len(self.data)
        }