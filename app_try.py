import streamlit as st
from datetime import date

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.title('Stock Forecast App')

# Set up Alpha Vantage API key
API_KEY = "02UFWIM2I3UK5Y9N"

# Define stocks and time period
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load data from Alpha Vantage API
ts = TimeSeries(key=API_KEY, output_format='pandas')
data, meta_data = ts.get_daily(symbol=selected_stock, outputsize='full')
data.reset_index(inplace=True)
data.columns = ['ds', 'y']
data['ds'] = pd.to_datetime(data['ds'])

st.subheader('Raw data')
st.write(data.tail())

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Create linear regression model and fit it to the training data
lr = LinearRegression()
lr.fit(train_data[['ds']], train_data[['y']])

# Predict on the test set
y_pred = lr.predict(test_data[['ds']])

# Calculate the root mean squared error (RMSE) of the model
rmse = mean_squared_error(test_data[['y']], y_pred, squared=False)

st.subheader('Test set prediction')
st.write(test_data)
st.write(f'RMSE: {rmse:.2f}')

# Predict future values
future_dates = pd.date_range(start=data['ds'].iloc[-1], periods=period, freq='D')[1:]
future_data = pd.DataFrame({'ds': future_dates})

y_pred = lr.predict(future_data[['ds']])
forecast = pd.DataFrame({'ds': future_dates, 'y': y_pred})

st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig = px.line(forecast, x='ds', y='y', title=f'{selected_stock} stock price forecast')
st.plotly_chart(fig)
