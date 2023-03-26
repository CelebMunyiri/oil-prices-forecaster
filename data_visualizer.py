import matplotlib.pyplot as plt
import seaborn as sns


def plot_oil_prices(df):
    """
    Create a line plot of historical oil prices.

    Parameters:
        df (pandas.DataFrame): the dataframe containing the oil price data.
    """
    plt.plot(df['date'], df['oil_price'])
    plt.title('Historical Oil Prices')
    plt.xlabel('Year')
    plt.ylabel('Price (USD/barrel)')
    plt.show()


def plot_correlation_matrix(df):
    """
    Create a heatmap of the correlation matrix for the input dataframe.

    Parameters:
        df (pandas.DataFrame): the input dataframe.
    """
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Correlation Matrix')
    plt.show()
