import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from fbprophet import Prophet
import warnings
warnings.filterwarnings("ignore")

def preprocess_data(oil_prices_df, economic_indicators_df, geopolitical_events_df):
    """
    Preprocess the data by merging and scaling the features and target variable.
    """
    # Merge the dataframes on the 'date' column
    merged_df = oil_prices_df.merge(economic_indicators_df, on='date', how='left')
    merged_df = merged_df.merge(geopolitical_events_df, on='date', how='left')
    
    # Fill any missing values with 0
    merged_df.fillna(0, inplace=True)
    
    # Scale the features and target variable
    scaler = StandardScaler()
    X = scaler.fit_transform(merged_df.drop(['date', 'oil_price'], axis=1))
    y = scaler.fit_transform(merged_df[['oil_price']])
    
    return X, y, merged_df.columns[1:-1]

def run_linear_regression(X, y):
    """
    Train a linear regression model on the preprocessed data.
    """
    # Split the data into train and test sets
    split_index = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f'Linear regression test score: {score:.2f}')
    
    return model

def run_prophet_model(merged_df):
    """
    Train a Prophet model on the preprocessed data.
    """
    # Rename the columns to fit Prophet's naming conventions
    prophet_df = merged_df.rename(columns={'date': 'ds', 'oil_price': 'y'})
    
    # Create and train the model
    model = Prophet()
    model.fit(prophet_df)
    
    # Make predictions
    future_df = model.make_future_dataframe(periods=365)
    forecast_df = model.predict(future_df)
    
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def run_time_series_decomposition(oil_prices_df):
    """
    Run a time series decomposition on the historical oil prices.
    """
    # Set the 'date' column as the index
    oil_prices_df.set_index('date', inplace=True)
    
    # Run the decomposition
    decomposition = seasonal_decompose(oil_prices_df, model='additive', freq=365)
    
    # Extract the seasonal, trend, and residual components
    seasonal = decomposition.seasonal
    trend = decomposition.trend
    residual = decomposition.resid
    
    return seasonal, trend, residual

   
