import matplotlib.pyplot as plt
import seaborn as sns

class InsurancePlotter:
    """
    A class to handle visualization of insurance data.
    """
    
    def __init__(self, df):
        self.df = df
        sns.set_theme(style="whitegrid")

    def plot_histogram(self, column, title, color='skyblue'):
        """Plots a histogram for a numerical column."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], kde=True, color=color)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_bar(self, x_col, y_col, title):
        """Plots a bar chart."""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=x_col, y=y_col, data=self.df)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.show()

    def plot_box(self, x_col, y_col, title):
        """Plots a box plot to detect outliers."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=x_col, y=y_col, data=self.df)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.show()

    def plot_correlation_heatmap(self, columns):
        """Plots a correlation heatmap for selected columns."""
        plt.figure(figsize=(10, 8))
        corr = self.df[columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

    def plot_scatter_geo_risk(self, df_agg, x_col, y_col, hue_col=None):
        """
        Specific scatter plot for ZipCode risk analysis.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_agg, x=x_col, y=y_col, hue=hue_col, size=y_col, sizes=(20, 200))
        plt.title(f'Relationship between {x_col} and {y_col}')
        plt.show()