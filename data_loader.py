import pandas as pd

def load_data():
    """
    Load the historical oil prices, economic indicators, and geopolitical events data.
    """
    # Load the historical oil price data
    oil_prices_df = pd.read_csv('oili_prices.csv', parse_dates=['date'])
    
    # Load the economic indicators data
    economic_indicators_df = pd.read_csv('economic_indicators.csv', parse_dates=['date'])
    
    # Load the geopolitical events data
    geopolitical_events_df = pd.read_csv('geopolitical_events2.csv', parse_dates=['date'])
    
    return oil_prices_df, economic_indicators_df, geopolitical_events_df

