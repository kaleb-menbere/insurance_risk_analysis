import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InsuranceEDA:
    """Perform exploratory data analysis on insurance data"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.set_plot_style()
        
    def set_plot_style(self):
        """Set consistent plot style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def calculate_loss_ratio(self, groupby_cols: List[str] = None):
        """Calculate loss ratio (TotalClaims / TotalPremium)"""
        if groupby_cols:
            grouped = self.data.groupby(groupby_cols).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            }).reset_index()
            grouped['LossRatio'] = grouped['TotalClaims'] / grouped['TotalPremium']
            return grouped
        else:
            total_claims = self.data['TotalClaims'].sum()
            total_premium = self.data['TotalPremium'].sum()
            return total_claims / total_premium
    
    def plot_numerical_distributions(self, columns: List[str], figsize: Tuple = (15, 10)):
        """Plot distributions of numerical columns"""
        n_cols = min(3, len(columns))
        n_rows = int(np.ceil(len(columns) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                ax = axes[idx]
                self.data[col].hist(bins=50, ax=ax, edgecolor='black')
                ax.set_title(f'Distribution of {col}', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def plot_categorical_distributions(self, columns: List[str], figsize: Tuple = (15, 10)):
        """Plot distributions of categorical columns"""
        n_cols = min(2, len(columns))
        n_rows = int(np.ceil(len(columns) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                ax = axes[idx]
                value_counts = self.data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'Top 10 Categories - {col}', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.show()
    
    def plot_temporal_trends(self, date_col: str = 'TransactionMonth'):
        """Plot temporal trends in claims and premiums"""
        if date_col in self.data.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
                self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
            
            # Aggregate by month
            monthly_data = self.data.groupby(pd.Grouper(key=date_col, freq='M')).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'nunique'  # Count unique policies
            }).reset_index()
            
            monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Total Claims Over Time
            axes[0, 0].plot(monthly_data[date_col], monthly_data['TotalClaims'], 
                           marker='o', linewidth=2, color='red')
            axes[0, 0].set_title('Total Claims Over Time', fontsize=12)
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Total Claims')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Total Premium Over Time
            axes[0, 1].plot(monthly_data[date_col], monthly_data['TotalPremium'], 
                           marker='o', linewidth=2, color='green')
            axes[0, 1].set_title('Total Premium Over Time', fontsize=12)
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Total Premium')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Loss Ratio Over Time
            axes[1, 0].plot(monthly_data[date_col], monthly_data['LossRatio'], 
                           marker='o', linewidth=2, color='purple')
            axes[1, 0].set_title('Loss Ratio Over Time', fontsize=12)
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Loss Ratio')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Policy Count Over Time
            axes[1, 1].plot(monthly_data[date_col], monthly_data['PolicyID'], 
                           marker='o', linewidth=2, color='blue')
            axes[1, 1].set_title('Number of Policies Over Time', fontsize=12)
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Number of Policies')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return monthly_data
        else:
            print(f"Column {date_col} not found in data")
            return None
    
    def plot_geographic_analysis(self):
        """Analyze geographic patterns"""
        if 'Province' in self.data.columns:
            province_stats = self.data.groupby('Province').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'nunique'
            }).reset_index()
            
            province_stats['LossRatio'] = province_stats['TotalClaims'] / province_stats['TotalPremium']
            province_stats['AvgClaim'] = province_stats['TotalClaims'] / province_stats['PolicyID']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Loss Ratio by Province
            sorted_provinces = province_stats.sort_values('LossRatio', ascending=False)
            axes[0, 0].barh(sorted_provinces['Province'], sorted_provinces['LossRatio'], 
                           color='lightcoral', edgecolor='black')
            axes[0, 0].set_title('Loss Ratio by Province', fontsize=12)
            axes[0, 0].set_xlabel('Loss Ratio')
            axes[0, 0].grid(True, alpha=0.3, axis='x')
            
            # Plot 2: Total Claims by Province
            sorted_provinces_claims = province_stats.sort_values('TotalClaims', ascending=False)
            axes[0, 1].barh(sorted_provinces_claims['Province'], sorted_provinces_claims['TotalClaims'], 
                           color='lightblue', edgecolor='black')
            axes[0, 1].set_title('Total Claims by Province', fontsize=12)
            axes[0, 1].set_xlabel('Total Claims')
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Plot 3: Number of Policies by Province
            sorted_provinces_policies = province_stats.sort_values('PolicyID', ascending=False)
            axes[1, 0].barh(sorted_provinces_policies['Province'], sorted_provinces_policies['PolicyID'], 
                           color='lightgreen', edgecolor='black')
            axes[1, 0].set_title('Number of Policies by Province', fontsize=12)
            axes[1, 0].set_xlabel('Number of Policies')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            
            # Plot 4: Average Claim Amount by Province
            sorted_provinces_avg = province_stats.sort_values('AvgClaim', ascending=False)
            axes[1, 1].barh(sorted_provinces_avg['Province'], sorted_provinces_avg['AvgClaim'], 
                           color='gold', edgecolor='black')
            axes[1, 1].set_title('Average Claim Amount by Province', fontsize=12)
            axes[1, 1].set_xlabel('Average Claim Amount')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.show()
            
            return province_stats
        else:
            print("Province column not found in data")
            return None
    
    def plot_vehicle_analysis(self):
        """Analyze vehicle-related patterns"""
        if 'Make' in self.data.columns:
            vehicle_stats = self.data.groupby('Make').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'nunique',
                'CustomValueEstimate': 'mean'
            }).reset_index()
            
            vehicle_stats['LossRatio'] = vehicle_stats['TotalClaims'] / vehicle_stats['TotalPremium']
            vehicle_stats = vehicle_stats.sort_values('PolicyID', ascending=False).head(15)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Top 15 Vehicle Makes by Policy Count
            axes[0, 0].barh(vehicle_stats['Make'][::-1], vehicle_stats['PolicyID'][::-1], 
                           color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Top 15 Vehicle Makes by Policy Count', fontsize=12)
            axes[0, 0].set_xlabel('Number of Policies')
            axes[0, 0].grid(True, alpha=0.3, axis='x')
            
            # Plot 2: Loss Ratio for Top Makes
            vehicle_stats_lr = vehicle_stats.sort_values('LossRatio', ascending=False)
            axes[0, 1].barh(vehicle_stats_lr['Make'], vehicle_stats_lr['LossRatio'], 
                           color='lightcoral', edgecolor='black')
            axes[0, 1].set_title('Loss Ratio by Vehicle Make', fontsize=12)
            axes[0, 1].set_xlabel('Loss Ratio')
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Plot 3: Average Custom Value Estimate
            vehicle_stats_value = vehicle_stats.sort_values('CustomValueEstimate', ascending=False)
            axes[1, 0].barh(vehicle_stats_value['Make'], vehicle_stats_value['CustomValueEstimate'], 
                           color='gold', edgecolor='black')
            axes[1, 0].set_title('Average Vehicle Value by Make', fontsize=12)
            axes[1, 0].set_xlabel('Average Custom Value Estimate')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            
            # Plot 4: Total Claims by Make
            axes[1, 1].barh(vehicle_stats['Make'][::-1], vehicle_stats['TotalClaims'][::-1], 
                           color='lightgreen', edgecolor='black')
            axes[1, 1].set_title('Total Claims by Vehicle Make', fontsize=12)
            axes[1, 1].set_xlabel('Total Claims')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.show()
            
            return vehicle_stats
        else:
            print("Make column not found in data")
            return None
    
    def plot_outlier_detection(self, columns: List[str]):
        """Detect and visualize outliers using box plots"""
        n_cols = min(3, len(columns))
        n_rows = int(np.ceil(len(columns) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                ax = axes[idx]
                bp = ax.boxplot(self.data[col].dropna(), patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['medians'][0].set_color('red')
                ax.set_title(f'Box Plot - {col}', fontsize=12)
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
                
                # Calculate outlier statistics
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)][col]
                print(f"{col}: {len(outliers)} outliers detected")
                
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.show()
    
    def create_insight_plot_1(self):
        """Creative Plot 1: Interactive Sunburst of Risk Profile"""
        # Prepare data for sunburst plot
        if all(col in self.data.columns for col in ['Province', 'Make', 'Gender']):
            # Aggregate data
            sunburst_data = self.data.groupby(['Province', 'Make', 'Gender']).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            
            sunburst_data['LossRatio'] = sunburst_data['TotalClaims'] / sunburst_data['TotalPremium']
            
            # Create sunburst plot
            fig = px.sunburst(
                sunburst_data,
                path=['Province', 'Make', 'Gender'],
                values='PolicyID',
                color='LossRatio',
                color_continuous_scale='RdYlGn_r',  # Red for high loss ratio, green for low
                title='Risk Profile Sunburst: Province → Vehicle Make → Gender',
                width=800,
                height=800
            )
            
            fig.update_traces(textinfo="label+percent parent")
            fig.show()
            
            # Save to reports directory
            fig.write_html("reports/figures/sunburst_risk_profile.html")
            
            return fig
        else:
            print("Required columns not found for sunburst plot")
            return None
    
    def create_insight_plot_2(self):
        """Creative Plot 2: Animated Scatter Plot Over Time"""
        if 'TransactionMonth' in self.data.columns and 'Province' in self.data.columns:
            # Prepare time-series data
            time_data = self.data.groupby(['TransactionMonth', 'Province']).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            
            time_data['LossRatio'] = time_data['TotalClaims'] / time_data['TotalPremium']
            
            # Create animated scatter plot
            fig = px.scatter(
                time_data,
                x='PolicyID',
                y='LossRatio',
                size='TotalClaims',
                color='Province',
                hover_name='Province',
                animation_frame='TransactionMonth',
                title='Risk Evolution Over Time: Policy Count vs Loss Ratio',
                labels={
                    'PolicyID': 'Number of Policies',
                    'LossRatio': 'Loss Ratio',
                    'TotalClaims': 'Total Claims Amount'
                },
                width=1000,
                height=600
            )
            
            fig.update_layout(showlegend=True)
            fig.show()
            
            # Save to reports directory
            fig.write_html("reports/figures/animated_risk_evolution.html")
            
            return fig
        else:
            print("Required columns not found for animated plot")
            return None
    
    def create_insight_plot_3(self):
        """Creative Plot 3: Parallel Coordinates Plot for Risk Segmentation"""
        if all(col in self.data.columns for col in ['Province', 'VehicleType', 'Gender', 'MaritalStatus']):
            # Prepare sample data for parallel coordinates
            sample_data = self.data.groupby(['Province', 'VehicleType', 'Gender', 'MaritalStatus']).agg({
                'TotalClaims': 'mean',
                'TotalPremium': 'mean',
                'CustomValueEstimate': 'mean',
                'PolicyID': 'count'
            }).reset_index()
            
            sample_data['LossRatio'] = sample_data['TotalClaims'] / sample_data['TotalPremium']
            
            # Take top 100 combinations for clarity
            sample_data = sample_data.nlargest(100, 'PolicyID')
            
            # Create parallel coordinates plot
            dimensions = [
                dict(label='Province', values=sample_data['Province']),
                dict(label='VehicleType', values=sample_data['VehicleType']),
                dict(label='Gender', values=sample_data['Gender']),
                dict(label='MaritalStatus', values=sample_data['MaritalStatus']),
                dict(label='Avg Premium', values=sample_data['TotalPremium']),
                dict(label='Avg Claims', values=sample_data['TotalClaims']),
                dict(label='Loss Ratio', values=sample_data['LossRatio'])
            ]
            
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=sample_data['LossRatio'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title='Loss Ratio')
                    ),
                    dimensions=dimensions
                )
            )
            
            fig.update_layout(
                title='Parallel Coordinates: Risk Profile Segmentation',
                width=1200,
                height=600
            )
            
            fig.show()
            
            # Save to reports directory
            fig.write_html("reports/figures/parallel_coordinates_segmentation.html")
            
            return fig
        else:
            print("Required columns not found for parallel coordinates plot")
            return None