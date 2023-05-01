import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet

# Load oil price data
oil_prices = pd.read_csv('oili_prices.csv')
oil_prices['date'] = pd.to_datetime(oil_prices['date'])
oil_prices = oil_prices.rename(columns={'value': 'oil_price'})

# Load economic indicators data
economic_data = pd.read_csv('economic_indicators.csv')
economic_data['date'] = pd.to_datetime(economic_data['date'])

# Load geopolitical events data
geopolitical_data = pd.read_csv('geopolitical_events2.csv')
geopolitical_data['start_date'] = pd.to_datetime(geopolitical_data['start_date'])
geopolitical_data['end_date'] = pd.to_datetime(geopolitical_data['end_date'])

# Combine data into a single DataFrame
data = pd.merge_asof(oil_prices, economic_data, on='date', direction='nearest')
data['geopolitical_event'] = False
for _, row in geopolitical_data.iterrows():
    mask = (data['date'] >= row['start_date']) & (data['date'] <= row['end_date'])
    data.loc[mask, 'geopolitical_event'] = True

# Split the data into training and testing sets
train_data = data[data['date'] < '2021-01-01']
test_data = data[data['date'] >= '2021-01-01']

# Train the model
model = Prophet()
model.add_regressor('gdp')
model.add_regressor('inflation')
model.add_regressor('interest_rate')
model.add_regressor('geopolitical_event')
model.fit(train_data[['date', 'oil_price', 'gdp', 'inflation', 'interest_rate', 'geopolitical_event']])

# Make predictions on the testing set
future = test_data[['date', 'gdp', 'inflation', 'interest_rate', 'geopolitical_event']]
forecast = model.predict(future)

# Evaluate the model
mae = np.mean(np.abs(forecast['yhat'].values - test_data['oil_price'].values))
mse = np.mean((forecast['yhat'].values - test_data['oil_price'].values) ** 2)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Plot the results
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=train_data, x='date', y='oil_price', ax=ax)
sns.lineplot(data=test_data, x='date', y='oil_price', ax=ax)
sns.lineplot(data=forecast, x='ds', y='yhat', ax=ax)
ax.set(title='Oil Prices Forecast', xlabel='Date', ylabel='Price')
ax.legend(['Training Data', 'Testing Data', 'Forecast'])
plt.show()






