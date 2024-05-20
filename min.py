import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas_datareader as web
from keras.models import load_model
import streamlit as st
import yfinance as yf   
import plotly.graph_objects as go
from keras.layers import Dense, LSTM

st.title('TradeInsight: Navigating Stock Space')

user_input= st.sidebar.text_input('Enter Stock Ticker','AAPL')
start = st.sidebar.date_input('Start Date')
end = st.sidebar.date_input('End Date')
df = web.DataReader(user_input, 'stooq', start, end)

fin= px.line(df, x=df.index, y= df['Close'])
st.plotly_chart(fin)

pricing_data, fundamental_data, news, time_charts, values= st.tabs(["Pricing data", "Fundamental Data", "Top 10 news", "Time Charts", "Next15"])
with pricing_data:
  st.header("Price Movements")
  df2= df
  df2["%Change"]= df['Close']/df['Close'].shift(1)-1
  df2.dropna(inplace=True)
  st.write(df2)
  annual_return= df2["%Change"].mean()*252*100
  st.write(annual_return)
  st.header("Price Overview")
  st.write(df.describe())

from alpha_vantage.fundamentaldata import FundamentalData          
with fundamental_data:
  key='QR45E646545AE'
  fd= FundamentalData(key, output_format='pandas')
  
  st.subheader('Balance Sheet')
  balance_sheet= fd.get_balance_sheet_annual(user_input)[0]
  bs= balance_sheet.T[2:]
  bs.columns = list(balance_sheet.T.iloc[0])
  st.write(bs)

  st.header('Income Statement')
  income_statement= fd.get_income_statement_annual(user_input)[0]
  is1= income_statement.T[2:]
  is1.columns = list(income_statement.T.iloc[0])
  st.write(is1)

  st.header('Cash Flow Statement')
  cash_flow= fd.get_cash_flow_annual(user_input)[0]
  cs= cash_flow.T[2:]
  cs.columns = list(cash_flow.T.iloc[0])
  st.write(cs)

from stocknews import StockNews
with news:
  st.header(f'News of {user_input}')
  sn = StockNews(user_input, save_news=False)
  df_news= sn.read_rss()
  for i in range(10):
    st.subheader(f'News {i+1}')
    st.write(df_news['published'][i])
    st.write(df_news['title'][i])
    st.write(df_news['summary'][i])
    title_sentiment= df_news['sentiment_title'][i]
    st.write(f'Title sentiment {title_sentiment}')
    news_sentiment= df_news['sentiment_summary'][i]
    st.write(f'News sentiment {news_sentiment}')

    
with time_charts: 
  st.subheader('Closing price vs Time chart with 100MA')
  ma100= df.Close.rolling(100).mean()
  fig= plt.figure(figsize =(12,6))
  plt.plot(ma100)
  plt.plot(df.Close)
  st.pyplot(fig)


  st.subheader('Closing price vs Time chart with 100MA and 200MA')
  ma100= df.Close.rolling(100).mean()
  ma200= df.Close.rolling(200).mean()
  fig= plt.figure(figsize =(12,6))
  plt.plot(ma100, 'r')
  plt.plot(ma200, 'g')
  plt.plot(df.Close, 'y')
  st.pyplot(fig)


  data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
  data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler(feature_range=(0,1))

  training_array= scaler.fit_transform(data_training)

  x_train=[]
  y_train=[]

  for i in range(100, training_array.shape[0]):
    x_train.append(training_array[i-100:i])
    y_train.append(training_array[i,0])

  x_train, y_train= np.array(x_train), np.array(y_train)
  
  model= load_model("keras_model.h5")

  past_100_days=data_training.tail(100)
  final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
  input_data=scaler.fit_transform(final_df)

  x_test=[]
  y_test=[]

  for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

  x_test, y_test= np.array(x_test), np.array(y_test)
  y_predicted=model.predict(x_test)

  scaler = scaler.scale_

  scal_fac= 1/scaler[0]
  y_predicted= y_predicted*scal_fac
  y_test=y_test*scal_fac


  st.subheader('Predictions vs Original')
  fig2= plt.figure(figsize=(12,6))
  plt.plot(y_test, 'b', label='Original Price')
  plt.plot(y_predicted, 'r', label= 'Predicted Price')
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.legend()
  st.pyplot(fig2)

  # Generate the input sequence for prediction
  last_10_days = data_testing[-10:].values.reshape(1, -1, 1)
  sequence = []
  for i in range(15):  # Predicting for the next 15 days
      prediction = model.predict(last_10_days)
      sequence.append(prediction)
      last_10_days = np.roll(last_10_days, -1)  # Shift the array
      last_10_days[0, -1, 0] = prediction  # Update the last value with the predicted one

  # Scale the predicted values back to the original scale using MinMaxScaler
  predicted_prices = np.array(sequence).flatten()
  predicted_prices_scaled = (predicted_prices - min(predicted_prices)) / (max(predicted_prices) - min(predicted_prices))

  # Generate dates for the next 15 days
  next_15_days_dates = pd.date_range(start=end, periods=15)

  # Plot the predictions
  plt.figure(figsize=(12, 6))
  plt.plot(df.index, df['Close'], label='Actual Prices')
  plt.plot(next_15_days_dates, predicted_prices_scaled, label='Predicted Prices')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.title('Predictions for the next 15 days')
  plt.legend()
  st.pyplot(plt)

with values:

  next_15_days_dates = pd.date_range(start=end, periods=15)
  prediction_table_data = {'Date': next_15_days_dates, 'Predicted Price': predicted_prices+ df['Close'].head(15)}
  prediction_table = pd.DataFrame(prediction_table_data)

  # Display the prediction table
  st.subheader('Predicted Prices for the Next 15 Days')
  st.table(prediction_table)

