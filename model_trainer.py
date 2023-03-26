import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_linear_regression(df):
    """
    Train a linear regression model to predict oil prices.

    Parameters:
        df (pandas.DataFrame): the input dataframe.

    Returns:
        (sklearn.linear_model.LinearRegression, float, float): a tuple containing the trained model, the
        training error (mean squared error), and the testing error (mean squared error).
    """
    X = df.drop(['date', 'oil_price'], axis=1)
    y = df['oil_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_error = mean_squared_error(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_error = mean_squared_error(y_test, y_test_pred)

    return model, train_error, test_error
