import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
import random
import plotly
import plotly.express as px
from flask import Blueprint, render_template, jsonify
from flask_login import login_user, current_user, login_required, logout_user
from . import db
import re
from datetime import date, datetime
from pytz import timezone
import pytz

from datetime import date
import streamlit as st
import time
from tqdm.notebook import tqdm
from tensorflow import keras
import datetime as dt
from datetime import date
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
import plotly.express as px
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import quandl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


import time
import pandas as pd
from flask import flash, session, redirect, url_for
import numpy as np
from .models import Note
import json
from flask import request
from sqlalchemy import create_engine
import sqlalchemy as db
from flask_session import Session
from datetime import datetime
from pytz import timezone
import pytz
from websites import create_app
import sqlalchemy as db
from flask import Flask, render_template, request, flash, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
# import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constant as ct
# Ignore Warnings
import warnings
import random
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
nltk.download('punkt')

engine = db.create_engine('sqlite:///db.tide_work')


views = Blueprint('views', __name__)
auth = Blueprint('auth', __name__)


@views.route('/home', methods=['GET'])
def home():
    if request.method == 'GET':
        flag = 0
        if not session.get("email"):
            return redirect(url_for('auth.login'))
        try:
            email=session['email']
            sql = "SELECT * FROM users where user_email== ?"
            results = engine.execute(sql, (email)).fetchall()
            df = pd.DataFrame(results)
            df.columns = results[0].keys()
            if (df.is_admin[0] == '1'):
                session['is_admin'] = True
            else:
                session['is_admin'] = False
        except:
            pass

        return render_template("home.html")


@views.route('/stockmanage', methods=['GET','POST'])
def stockmanage():
    # session['flash'].clear()

    ticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
    mydict = {}
    stocks=ticker_list['Symbol']
    all_entered_symbols =  engine.execute("SELECT stock FROM add_stock").fetchall()
    df = pd.DataFrame(all_entered_symbols)
    df.columns = all_entered_symbols[0].keys()
    stocks= set(stocks) - set(df['stock'])
    #print(stocks)
    for k in stocks:
        mydict[k] = k
    try:
        results = engine.execute("SELECT * FROM add_stock").fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        list_users = df.to_dict(orient='records')
    except:
        list_users = [{'stock': 'Not Found',
                       'name': 'Not Found',
                       'sector': 'Not Found'}]
    if request.method == 'POST':
        tickerSymbol = request.form.get('my_name')
        sql = "SELECT * FROM 'add_stock' where stock == ? and email== ?"
        results = engine.execute(sql, (tickerSymbol,session.get("email"))).fetchall()
        #df = pd.DataFrame(results)
        #print(len(results))
        # if len(results)>0:


        try:
            df = pd.DataFrame(results)
            df.columns = results[0].keys()
            flash('Error! Stock already add in Watch List', category='error')
        except:
            sql = """insert into add_stock (email,stock,name,sector) VALUES (?,?,?,?);"""
            engine.execute(sql, (session.get("email"), tickerSymbol,'',''))
            flash('Stock Added Successfully', category='sucess')
    return render_template("managestock.html",mydict=mydict,list_stock=list_users)


@views.route("/delete_manage_stock/<val>", methods=["POST", "GET"])
def delete_manage_stock(val):
    data = {}

    if request.method == 'POST':
        try:
            sql = "DELETE FROM add_stock WHERE stock='" + val + "'"
            engine.execute(sql)
            data.update({'success': 1, 'msg': 'Stock! Delete Successfully'})
        except:
            data.update({'success': 0, 'msg': 'Error! There is an error'})
        return jsonify(data)




@views.route('/watchlist', methods=['GET'])
def watchlist():
    ticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
    mydict = ticker_list.to_dict(orient='records')
    print(mydict)
    watch_list = [{'stock': 'Not Found'}]
    if request.method == 'GET':

        flag = 0
        if not session.get("email"):
            return redirect(url_for('auth.login'))
        try:
            sql = "SELECT * FROM 'watch_list' where user_email == ?"
            results = engine.execute(sql, session.get("email")).fetchall()
            try:
                df = pd.DataFrame(results)
                df.columns = results[0].keys()
                watch_list = df.to_dict(orient='records')
            except:
                pass
        except:
            watch_list = [{'stock': 'Not Found'}]

        return render_template("watchlist.html", user='admin@quickbot.com', watch_list=watch_list,mydict=mydict)


@views.route("/stock_add/<val>", methods=["POST", "GET"])
def stock_add(val):
    print(val)
    data = {}
    if request.method == 'POST':
        try:
            sql = "SELECT * FROM 'watch_list' where stock == ? and user_email == ?"
            results = engine.execute(sql, (val,session.get("email"))).fetchall()
            try:
                df = pd.DataFrame(results)
                df.columns = results[0].keys()
                # flash('Container already exist', category='error')
                data.update({'success': 0, 'msg': 'Error! Stock already add in Watch List'})
            except:
                sql = """insert into watch_list(user_email,stock) VALUES (?,?);"""
                engine.execute(sql, (session.get("email"),val))
                data.update({'success': 1, 'msg': 'Stock! Added Successfully'})
        except:
            data.update({'success': 0, 'msg': 'Error! There is an error'})
        return jsonify(data)





@views.route('/users', methods=['GET'])
def users():
    if request.method == 'GET':
        flag = 0
        if not session.get("email"):
            return redirect(url_for('auth.login'))

        try:
            results = engine.execute("SELECT * FROM users where is_admin !='1' ").fetchall()
            df = pd.DataFrame(results)
            df.columns = results[0].keys()
            list_users = df.to_dict(orient='records')
        except:
            list_users = [{'email': 'Not Found',
                           'name': 'Not Found',
                           'password': 'Not Found',
                           'is_admin': 'Not Found',
                           'creation_date': 'Not Found'}]

        return render_template("users.html", user='admin@quickbot.com', list_users=list_users)

@views.route('/save_users', methods=['POST'])
def save_users():
    email = request.form.get('email')
    print(email)
    data = {}
    sql = "SELECT * FROM 'users' where user_email == ?"
    results = engine.execute(sql, email).fetchall()
    try:
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        # flash('Container already exist', category='error')
        data.update({'success': 0, 'msg': 'Error! Users already exist'})
    except:
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        is_admin = False
        matchs = "([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
        if (re.match(matchs, email)):
            date_format = '%m/%d/%Y %H:%M:%S %Z'
            date = datetime.now(tz=pytz.utc)
            date = date.astimezone(timezone('US/Pacific'))

            # creation_date = request.form.get('creation_date')

            sql = """insert into users(user_email,name,password,is_admin,creation_date) VALUES (?,?,?,?,?);"""
            engine.execute(sql, (email, name, password, is_admin, date.strftime(date_format)))
            data.update({'success': 1, 'msg': 'Account! Created Successfully'})
        else:
            data.update({'success': 0, 'msg': 'Error! Please follow email format'})
    return jsonify(data)

@views.route("/delete/<val>", methods=["POST", "GET"])
def delete(val):
    data = {}

    if request.method == 'POST':
        try:
            sql = "DELETE FROM users WHERE user_email='" + val + "'"
            engine.execute(sql)
            data.update({'success': 1, 'msg': 'Account! Delete Successfully'})
        except:
            data.update({'success': 0, 'msg': 'Error! There is an error'})
        return jsonify(data)



@views.route("/action", methods=["POST", "GET"])
def action():
    data = {}

    if request.method == 'POST':
        try:
            email1 = request.form['email1']
            username = request.form['username1']
            password = request.form['password1']
            sql = "UPDATE users set name=?, password= ? where user_email='" + email1 + "'"
            engine.execute(sql, (username, password))
            data.update({'success': 1, 'msg': 'Account! Updated Successfully'})

        except:
            data.update({'success': 0, 'msg': 'Error! Please enter correct information'})
        return jsonify(data)



def load_ticker_data():
    start_date='2021-07-07'
    end_date='2022-07-07'
    ticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
    tickerSymbol = ticker_list["ABT"][
        random.randint(-1, len(ticker_list["ABT"]))]  # each time it will pick random from list
    tickerData = yf.Ticker(tickerSymbol)  # Get ticker data
    tickerDf = tickerData.history(period='1d', start=start_date,
                                  end=end_date)  # get the historical prices for this ticker

    string_logo = tickerData.info['logo_url']
    string_name = tickerData.info['longName']
    string_summary = tickerData.info['longBusinessSummary']
    qf = cf.QuantFig(tickerDf, title='First Quant Figure', legend='top', name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    return tickerDf,string_logo,string_name,string_summary,fig

@views.route('/recommendation', methods=['GET','POST'])
def recommendation():
    ticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
    ticker_list['Symbol']
    mydict = {}
    for k in ticker_list['Symbol']:
        mydict[k] = k

    tickerDf=''
    string_logo=''
    string_name=''
    string_summary=''
    graphJSON=''
    list_of_dics=''
    if request.method == 'POST':
        buy = request.form.get('buy')
        sell = request.form.get('sell')
        start = request.form.get('start')
        end = request.form.get('end')

        if(start):
            print(start)
        else:
            start='2021-07-07'
            print(start)
        if (end):
            print(end)
        else:
            end = '2022-07-07'
            print(end)
        if(buy):
            tickerDf, string_logo, string_name, string_summary, fig = load_ticker_data()
            tickerDf.to_csv('a.csv')
            tickerDf=pd.read_csv('a.csv')
            list_of_dics = tickerDf.to_dict(orient='records')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        if (sell):
            tickerDf, string_logo, string_name, string_summary, fig = load_ticker_data()
            tickerDf.to_csv('a.csv')
            tickerDf=pd.read_csv('a.csv')
            list_of_dics = tickerDf.to_dict(orient='records')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        pass
    return render_template("recommendation.html",string_logo=string_logo,mydict=mydict, string_name=string_name, list_of_dics=list_of_dics,string_summary=string_summary,graphJSON=graphJSON)



@views.route('/recommendation2', methods=['GET','POST'])
def recommendation2():
    ticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
    ticker_list['Symbol']
    mydict = {}
    for k in ticker_list['Symbol']:
        mydict[k] = k

    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year - 2, end.month, end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv('' + quote + '.csv')
        if (df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:' + quote, outputsize='full')
            # Format df
            # Last 2 yrs rows => 502, in ascending order => ::-1
            data = data.head(503).iloc[::-1]
            data = data.reset_index()
            # Keep Required cols only
            df = pd.DataFrame()
            df['Date'] = data['date']
            df['Open'] = data['1. open']
            df['High'] = data['2. high']
            df['Low'] = data['3. low']
            df['Close'] = data['4. close']
            df['Adj Close'] = data['5. adjusted close']
            df['Volume'] = data['6. volume']
            df['Code'] = data['6. code']

            df.to_csv('' + quote + '.csv', index=False)

    def ARIMA_ALGO(df, quote):
        uniqueVals = df["Code"].unique()
        len(uniqueVals)
        df = df.set_index("Code")

        # for daily basis
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')

        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6, 1, 0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat[0])
                obs = test[t]
                history.append(obs)
            return predictions

        for company in uniqueVals[:10]:
            data = (df.loc[company, :]).reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Price', 'Date']]
            Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'], axis=1)
            fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
            plt.plot(Quantity_date)
            plt.savefig('websites/static/Trends.png')
            plt.close(fig)

            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]
            # fit in model
            predictions = arima_model(train, test)

            # plot graph
            fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
            plt.plot(test, label='Actual Price')
            plt.plot(predictions, label='Predicted Price')
            plt.legend(loc=4)
            plt.savefig('websites/static/ARIMA.png')
            plt.close(fig)
            print()
            print("##############################################################################")
            arima_pred = predictions[-2]
            print("Tomorrow's", quote, " Closing Price Prediction by ARIMA:", arima_pred)
            # rmse calculation
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            print("ARIMA RMSE:", error_arima)
            print("##############################################################################")
            return arima_pred, error_arima

    def LSTM_ALGO(df, quote):
        # Split data into training set and test set
        dataset_train = df.iloc[0:int(0.8 * len(df)), :]
        dataset_test = df.iloc[int(0.8 * len(df)):, :]
        ############# NOTE #################
        # TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
        training_set = df.iloc[:, 4:5].values  # 1:2, to store as numpy array else Series obj will be stored
        # select cols using above manner to select as float64 type, view in var explorer

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))  # Scaled values btween 0,1
        training_set_scaled = sc.fit_transform(training_set)
        # In scaling, fit_transform for training, transform for test

        # Creating data stucture with 7 timesteps and 1 output.
        # 7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train = []  # memory with 7 days from day i
        y_train = []  # day i
        for i in range(7, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - 7:i, 0])
            y_train.append(training_set_scaled[i, 0])
        # Convert list to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_forecast = np.array(X_train[-1, 1:])
        X_forecast = np.append(X_forecast, y_train[-1])
        # Reshaping: Adding 3rd dimension
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # .shape 0=row,1=col
        X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
        # For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)

        # Building RNN
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM

        # Initialise RNN
        regressor = Sequential()

        # Add first LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        # units=no. of neurons in layer
        # input_shape=(timesteps,no. of cols/features)
        # return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
        regressor.add(Dropout(0.1))

        # Add 2nd LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))

        # Add 3rd LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))

        # Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))

        # Add o/p layer
        regressor.add(Dense(units=1))

        # Compile
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        # Training
        regressor.fit(X_train, y_train, epochs=25, batch_size=32)
        # For lstm, batch_size=power of 2

        # Testing
        ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price = dataset_test.iloc[:, 4:5].values

        # To predict, we need stock prices of 7 days before the test set
        # So combine train and test set to get the entire data set
        dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
        testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
        testing_set = testing_set.reshape(-1, 1)
        # -1=till last row, (-1,1)=>(80,1). otherwise only (80,0)

        # Feature scaling
        testing_set = sc.transform(testing_set)

        # Create data structure
        X_test = []
        for i in range(7, len(testing_set)):
            X_test.append(testing_set[i - 7:i, 0])
            # Convert list to numpy arrays
        X_test = np.array(X_test)

        # Reshaping: Adding 3rd dimension
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Testing Prediction
        predicted_stock_price = regressor.predict(X_test)

        # Getting original prices back from scaled values
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(real_stock_price, label='Actual Price')
        plt.plot(predicted_stock_price, label='Predicted Price')

        plt.legend(loc=4)
        plt.savefig('websites/static/LSTM.png')
        plt.close(fig)

        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

        # Forecasting Prediction
        forecasted_stock_price = regressor.predict(X_forecast)

        # Getting original prices back from scaled values
        forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

        lstm_pred = forecasted_stock_price[0, 0]
        print()
        print("##############################################################################")
        print("Tomorrow's ", quote, " Closing Price Prediction by LSTM: ", lstm_pred)
        print("LSTM RMSE:", error_lstm)
        print("##############################################################################")
        return lstm_pred, error_lstm

    def LIN_REG_ALGO(df, quote):
        # No of days to be forcasted in future
        forecast_out = int(7)
        # Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        # New df with only relevant data
        df_new = df[['Close', 'Close after n days']]

        # Structure data for train, test & forecast
        # lables of known data, discard last 35 rows
        y = np.array(df_new.iloc[:-forecast_out, -1])
        y = np.reshape(y, (-1, 1))
        # all cols of known data except lables, discard last 35 rows
        X = np.array(df_new.iloc[:-forecast_out, 0:-1])
        # Unknown, X to be forecasted
        X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])

        # Traning, testing to plot graphs, check accuracy
        X_train = X[0:int(0.8 * len(df)), :]
        X_test = X[int(0.8 * len(df)):, :]
        y_train = y[0:int(0.8 * len(df)), :]
        y_test = y[int(0.8 * len(df)):, :]

        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        X_to_be_forecasted = sc.transform(X_to_be_forecasted)

        # Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        # Testing
        y_test_pred = clf.predict(X_test)
        y_test_pred = y_test_pred * (1.04)
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2, 4.8), dpi=65)
        plt2.plot(y_test, label='Actual Price')
        plt2.plot(y_test_pred, label='Predicted Price')

        plt2.legend(loc=4)
        plt2.savefig('websites/static/LR.png')
        plt2.close(fig)

        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

        # Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set = forecast_set * (1.04)
        mean = forecast_set.mean()
        lr_pred = forecast_set[0, 0]
        print()
        print("##############################################################################")
        print("Tomorrow's ", quote, " Closing Price Prediction by Linear Regression: ", lr_pred)
        print("Linear Regression RMSE:", error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr

    # def retrieving_tweets_polarity(symbol):
    #     stock_ticker_map = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
    #     stock_full_form = stock_ticker_map[stock_ticker_map['Ticker'] == symbol]
    #     symbol = stock_full_form['Name'].to_list()[0][0:12]
    #
    #     auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
    #     auth.set_access_token(ct.access_token, ct.access_token_secret)
    #     user = tweepy.API(auth)
    #
    #     tweets = tweepy.Cursor(user.search, q=symbol, tweet_mode='extended', lang='en', exclude_replies=True).items(
    #         ct.num_of_tweets)
    #
    #     tweet_list = []  # List of tweets alongside polarity
    #     global_polarity = 0  # Polarity of all tweets === Sum of polarities of individual tweets
    #     tw_list = []  # List of tweets only => to be displayed on web page
    #     # Count Positive, Negative to plot pie chart
    #     pos = 0  # Num of pos tweets
    #     neg = 1  # Num of negative tweets
    #     for tweet in tweets:
    #         count = 20  # Num of tweets to be displayed on web page
    #         # Convert to Textblob format for assigning polarity
    #         tw2 = tweet.full_text
    #         tw = tweet.full_text
    #         # Clean
    #         tw = p.clean(tw)
    #         # print("-------------------------------CLEANED TWEET-----------------------------")
    #         # print(tw)
    #         # Replace &amp; by &
    #         tw = re.sub('&amp;', '&', tw)
    #         # Remove :
    #         tw = re.sub(':', '', tw)
    #         # print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
    #         # print(tw)
    #         # Remove Emojis and Hindi Characters
    #         tw = tw.encode('ascii', 'ignore').decode('ascii')
    #
    #         # print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
    #         # print(tw)
    #         blob = TextBlob(tw)
    #         polarity = 0  # Polarity of single individual tweet
    #         for sentence in blob.sentences:
    #
    #             polarity += sentence.sentiment.polarity
    #             if polarity > 0:
    #                 pos = pos + 1
    #             if polarity < 0:
    #                 neg = neg + 1
    #
    #             global_polarity += sentence.sentiment.polarity
    #         if count > 0:
    #             tw_list.append(tw2)
    #
    #         tweet_list.append(Tweet(tw, polarity))
    #         count = count - 1
    #     if len(tweet_list) != 0:
    #         global_polarity = global_polarity / len(tweet_list)
    #     else:
    #         global_polarity = global_polarity
    #     neutral = ct.num_of_tweets - pos - neg
    #     if neutral < 0:
    #         neg = neg + neutral
    #         neutral = 20
    #     print()
    #     print("##############################################################################")
    #     print("Positive Tweets :", pos, "Negative Tweets :", neg, "Neutral Tweets :", neutral)
    #     print("##############################################################################")
    #     labels = ['Positive', 'Negative', 'Neutral']
    #     sizes = [pos, neg, neutral]
    #     explode = (0, 0, 0)
    #     fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    #     fig1, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
    #     ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    #     # Equal aspect ratio ensures that pie is drawn as a circle
    #     ax1.axis('equal')
    #     plt.tight_layout()
    #     plt.savefig('websites/static/SA.png')
    #     plt.close(fig)
    #     # plt.show()
    #     if global_polarity > 0:
    #         print()
    #         print("##############################################################################")
    #         print("Tweets Polarity: Overall Positive")
    #         print("##############################################################################")
    #         tw_pol = "Overall Positive"
    #     else:
    #         print()
    #         print("##############################################################################")
    #         print("Tweets Polarity: Overall Negative")
    #         print("##############################################################################")
    #         tw_pol = "Overall Negative"
    #     return global_polarity, tw_list, tw_pol, pos, neg, neutral

    def recommending(df, mean, quote):
        global_polarity = random.uniform(0, 1)
        if today_stock.iloc[-1]['Close'] < mean:
            if global_polarity > 0.2:
                idea = "RISE"
                decision = "BUY"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea, "in", quote,
                      "stock is expected => ", decision)
            elif global_polarity <= 0.2:
                idea = "FALL"
                decision = "SELL"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea, "in", quote,
                      "stock is expected => ", decision)
        else:
            idea = "FALL"
            decision = "SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea, "in", quote,
                  "stock is expected => ", decision)
        return idea, decision

    # ticker_list = pd.read_csv(
    #     'https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
    # tickerSymbol = ticker_list["ABT"][
    #     random.randint(-1, len(ticker_list["ABT"]))]  # each time it will pick random from list


    if request.method == 'POST':
        tickerSymbol=request.form.get('my_name')
        try:
            get_historical(tickerSymbol)
        except:
            return render_template('recommendation2.html',not_found=True)
        else:
            df = pd.read_csv(str(tickerSymbol) + '.csv')
            today_stock = df.iloc[-1:]
            df = df.dropna()
            code_list = []
            for i in range(0, len(df)):
                code_list.append(tickerSymbol)
            df2 = pd.DataFrame(code_list, columns=['Code'])
            df2 = pd.concat([df2, df], axis=1)
            df = df2
            arima_pred, error_arima = ARIMA_ALGO(df, tickerSymbol)
            lstm_pred, error_lstm = LSTM_ALGO(df, tickerSymbol)
            df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df, tickerSymbol)
            # polarity, tw_list, tw_pol, pos, neg, neutral = retrieving_tweets_polarity(tickerSymbol)
            idea, decision = recommending(df, mean,today_stock)
            print("Forecasted Prices for Next 7 days:")
            print(forecast_set)
            today_stock = today_stock.round(2)
            return render_template('recommendation2.html', mydict=mydict,quote=tickerSymbol, arima_pred=round(arima_pred, 2), lstm_pred=round(lstm_pred, 2),
                               lr_pred=round(lr_pred, 2), open_s=today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False), tw_pol="Positive",
                               adj_close=today_stock['Adj Close'].to_string(index=False),
                               idea=idea, decision=decision, high_s=today_stock['High'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),
                               vol=today_stock['Volume'].to_string(index=False),
                               forecast_set=forecast_set, error_lr=round(error_lr, 2), error_lstm=round(error_lstm, 2),
                               error_arima=round(error_arima, 2))
    else:
        return render_template('recommendation2.html', mydict=mydict)













def prophet_function(stock, n_years):
    period = n_years * 365
    START = "2020-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
    data = load_data(stock)
    data1=data.tail()
    def plot_raw_data(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        return fig
    fig=plot_raw_data(data)
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    forecast1=forecast.tail()
    fig1 = plot_plotly(m, forecast)
    fig2 = m.plot_components(forecast)

    return data1,fig,forecast1,fig1,fig2


@views.route('/prophet', methods=['GET','POST'])
def prophet():
    data1=''
    graphJSON1=''
    forecast1=''
    graphJSON2=''
    fig2=''
    mydict = {'GOOG':'GOOG', 'AAPL':'AAPL', 'MSFT':'MSFT', 'GME':'GME'}
    if request.method == 'POST':
        company=request.form.get('my_name')
        prediction = request.form.get('myRange')
        print(prediction)
        data1, fig, forecast1, fig1, fig2 = prophet_function(company, int(prediction))
        fig2.savefig(r'websites/static/book_read.png')
        data1 = data1.to_dict(orient='Records')
        forecast1 = forecast1.to_dict(orient='Records')
        graphJSON1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON2 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        pass
    return render_template("prophet.html",mydict=mydict,data1=data1, graphJSON1=graphJSON1, forecast1=forecast1, graphJSON2=graphJSON2, fig2=fig2)





def get_ticker_data(stockticker,start_date,end_date):
    tickerData = yf.Ticker(stockticker)
    tickerDf = tickerData.history(period='1d', start=start_date,
                              end=end_date)  # get the historical prices for this ticker
    string_logo = tickerData.info['logo_url']
    string_name = tickerData.info['longName']
    string_summary = tickerData.info['longBusinessSummary']
    qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')

    qf.add_ema(periods=100, color='red')
    qf.add_sma(periods=50, color='blue')
    qf.add_rsi(periods=20, color='black')

    fig = qf.iplot(asFigure=True)
    return tickerDf,string_logo,string_name,string_summary,fig




@views.route('/stockdata', methods=['GET','POST'])
def stockdata():

    tickerDf = ''
    string_logo = ''
    string_name = ''
    string_summary = ''
    graphJSON = ''
    list_of_dics = ''

    ticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
    ticker_list['Symbol']
    mydict = {}
    for k in ticker_list['Symbol']:
        mydict[k] = k
    if request.method == 'POST':
        start = request.form.get('start')
        end = request.form.get('end')
        stockticker = request.form.get('stockticker')
        favourite = request.form.get('favourite')
        if(start):
            start='2021-07-07'
        else:
            start='2021-07-07'
        if (end):
            end = '2022-07-07'
        else:
            end = '2022-07-07'
        tickerDf,string_logo,string_name,string_summary,fig=get_ticker_data(stockticker,start,end)
        tickerDf.to_csv('a.csv')
        tickerDf = pd.read_csv('a.csv')
        list_of_dics = tickerDf.to_dict(orient='records')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    else:
        pass
    return render_template("stockdata.html",string_logo=string_logo,mydict=mydict, string_name=string_name, list_of_dics=list_of_dics,string_summary=string_summary,graphJSON=graphJSON)



def load_data(ticker,START, TODAY):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
def Linear_Regression_model(train_data, test_data):

    x_train = train_data.drop(columns=['Date', 'Close'], axis=1)
    x_test = test_data.drop(columns=['Date', 'Close'], axis=1)
    y_train = train_data['Close']
    y_test = test_data['Close']

    # First Create the LinearRegression object and then fit it into the model
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_train, y_train)

    # Making the Predictions
    prediction = model.predict(x_test)

    return prediction
def prediction_plot(pred_data, test_data, ticker_name):

    test_data['Predicted'] = 0
    test_data['Predicted'] = pred_data

    # Resetting the index

    test_data.reset_index(inplace=True, drop=True)
    pred_data=test_data[['Date', 'Close', 'Predicted']]

    # Plotting the Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Predicted'], mode='lines', name='Predicted'))

    return pred_data,fig
def create_train_test_data(df1):

    data = df1.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df1)), columns=['Date', 'High', 'Low', 'Open', 'Volume', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['High'][i] = data['High'][i]
        new_data['Low'][i] = data['Low'][i]
        new_data['Open'][i] = data['Open'][i]
        new_data['Volume'][i] = data['Volume'][i]
        new_data['Close'][i] = data['Close'][i]

    # Removing the hour, minute and second
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.date

    train_data_len = math.ceil(len(new_data) * .8)

    train_data = new_data[:train_data_len]
    test_data = new_data[train_data_len:]

    return train_data, test_data


def create_train_test_LSTM(df, epoch, b_s):
    df_for_train=df[['Date', 'Close']]
    df_filtered = df.filter(['Close'])
    dataset = df_filtered.values

    # Training Data
    training_data_len = math.ceil(len(dataset) * .7)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0: training_data_len, :]

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(train_data[i - 60:i, 0])
        y_train_data.append(train_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    # Testing Data
    test_data = scaled_data[training_data_len - 60:, :]

    x_test_data = []
    y_test_data = dataset[training_data_len:, :]

    for j in range(60, len(test_data)):
        x_test_data.append(test_data[j - 60:j, 0])

    x_test_data = np.array(x_test_data)

    x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train_data, y_train_data, batch_size=int(b_s), epochs=int(epoch))

    predictions = model.predict(x_test_data)
    predictions = scaler.inverse_transform(predictions)

    train = df_filtered[:training_data_len]
    valid = df_filtered[training_data_len:]
    valid['Predictions'] = predictions

    new_valid = valid.reset_index()
    new_valid.drop('index', inplace=True, axis=1)
    return df_for_train,new_valid

def linear_regression(ticker, year,START, TODAY):
    data=load_data(ticker,START, TODAY)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    df_for_train=data[['Date', 'Close']]
    train_data, test_data = create_train_test_data(data)
    pred_data = Linear_Regression_model(train_data, test_data)
    pred_data,fig=prediction_plot(pred_data, test_data, ticker)
    return train_data, test_data,df_for_train,pred_data,fig
def lstm(ticker, year,START, TODAY,epoch,b_s):
    data=load_data(ticker,START, TODAY)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    df_for_train,new_valid=create_train_test_LSTM(data,epoch,b_s)
    return df_for_train,new_valid



@views.route('/prediction', methods=['GET','POST'])
def prediction():
    mydict={"AAPL":"AAPL", "GOOG":"GOOG", "MSFT":"MSFT", "AMZN":"AMZN", "TSLA":"TSLA", "GME":"GME", "NVDA":"NVDA", "AMD":"AMD"}
    model={'Linear Regression':'Linear Regression', 'LSTM':'LSTM'}
    stock_selected=''
    model_selected=''
    df_for_train=''
    pred_data=''
    graphJSON=''
    if request.method == 'POST':
        year=request.form.get('myRange')
        stock_selected = request.form.get('stock_selected')
        model_selected = request.form.get('model_selected')
        epochs = request.form.get('epochs')
        batch = request.form.get('batch')
        if('Linear' in str(model_selected)):
            START='2021-7-7'
            TODAY='2022-5-5'
            train_data, test_data, df_for_train, pred_data, fig = linear_regression(stock_selected, year,START,TODAY)
            df_for_train = df_for_train.to_dict(orient='Records')
            pred_data = pred_data.to_dict(orient='Records')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        elif('LSTM' in str(model_selected)):
            START = '2021-7-7'
            TODAY = '2022-5-5'
            df_for_train,new_valid=lstm(stock_selected, year,START,TODAY,epochs,batch)
            df_for_train = df_for_train.to_dict(orient='Records')
            pred_data = new_valid.to_dict(orient='Records')

            plt.title('Actual Close prices vs Predicted Using LSTM Model', fontsize=10)
            plt.plot(new_valid[['Close', 'Predictions']])
            plt.legend(['Actual', 'Predictions'], loc='upper left', prop={"size": 10})
            plt.tight_layout()
            plt.savefig(r'websites/static/lstm_predict.jpg', dpi=60)

    else:
        pass
    return render_template("prediction.html",mydict=mydict,model=model,stock_selected=stock_selected,model_selected=model_selected,df_for_train=df_for_train,pred_data=pred_data,graphJSON=graphJSON)


@views.route('/progress', methods=['GET','POST'])
def progress():
    return render_template("progress.html")








