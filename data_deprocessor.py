import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Preprocess the data by filling missing values, scaling the data, and creating new features.

    Parameters:
        df (pandas.DataFrame): the input dataframe to preprocess.

    Returns:
        pandas.DataFrame: the preprocessed dataframe.
    """
    # Fill missing values
    imputer = SimpleImputer()
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Scale the data
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    # Create new features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df
