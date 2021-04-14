import os
import re

import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd

from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')


from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import streamlit as st


pairs = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CAD', 'CADCHF', 
        'CADJPY', 'CHF', 'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
        'EURJPY', 'EURNZD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 
        'GBPNZD', 'GBPUSD', 'JPY', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD']

def get_data(pair):
        ''' Retrieves (from a github repo) and prepares the data.
        '''
        url = f'https://raw.githubusercontent.com/African-Quant/WQU_MScFE_Capstone_Grp9/master/Datasets/{pair}%3DX.csv'
        raw = pd.read_csv(url)
        raw = pd.DataFrame(raw).drop(['Adj Close', 'Volume'], axis=1)
        raw.iloc[:,0] = pd.to_datetime(raw.iloc[:,0])
        raw.set_index('Date', inplace=True)
        return raw


# ATR
def eATR(df1,n=14):
    """This calculates the exponential Average True Range of of a dataframe of the open,
    high, low, and close data of an instrument"""

    df = df1[['Open',	'High',	'Low',	'Close']].copy()
    # True Range
    df['TR'] = 0
    for i in range(len(df)):
        try:
            df.iloc[i, 4] = max(df.iat[i,1] - df.iat[i,2],
                         abs(df.iat[i,1] - df.iat[i-1,3]),
                         abs(df.iat[i,2] - df.iat[i-1,3]))
        except ValueError:
            pass

    # eATR
    df['eATR'] = df['TR'].ewm(span=n, adjust=False).mean()
           
    return df['eATR']

def ssl(df1):
    """This function adds the ssl indicator as features to a dataframe
    """
    df = df1.copy()
    df['smaHigh'] = df['High'].rolling(window=10).mean()
    df['smaLow'] = df['Low'].rolling(window=10).mean()
    df['hlv'] = 0
    df['hlv'] = np.where(df['Close'] > df['smaHigh'],1,np.where(df['Close'] < df['smaLow'],-1,df['hlv'].shift(1)))
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    df['sslPosition'] = np.where(df['Close'] > df['sslUp'], 1,
                               np.where(df['Close'] < df['sslDown'], -1, 0))
    return df[['sslDown', 'sslUp', 'sslPosition']]

# Waddah Attar
def WAE(df1):
    """This function creates adds the indicator Waddah Attar features to a dataframe
    """
    df = df1.copy()

  # EMA
    long_ema = df.loc[:,'Close'].ewm(span=40, adjust=False).mean()
    short_ema = df.loc[:,'Close'].ewm(span=20, adjust=False).mean()

      # MACD
    MACD = short_ema - long_ema
  
  # bBands
    sma20 = df.loc[:,'Close'].rolling(window=20).mean()  # 20 SMA
    
    stddev = df.loc[:,'Close'].rolling(window=20).std() # 20 STDdev
    lower_band = sma20 - (2 * stddev)
    upper_band = sma20 + (2 * stddev)

    #Waddah Attar
    t1 = (MACD - MACD.shift(1))* 150
    #t2 = MACD.shift(2) - MACD.shift(3)
    df['e1'] = upper_band - lower_band
    df['e2'] = -1 *df['e1']
        #e2 = upper_band.shift(1) - lower_band.shift(1)

    df['trendUp'] = np.where(t1 > 0, t1, 0)
    df['trendDown'] =  np.where(t1 < 0, t1, 0)

    df['waePosition'] = np.where(df['trendUp'] > 0, 1,
                               np.where(df['trendDown'] < 0, -1, 0))
  
  
    return df[['e1','e2','trendUp', 'trendDown', 'waePosition']]

def lag_feat(data1):
    """This function adds lag returns as features to a dataframe
    """
    data = data1.copy()
    lags = 8
    cols = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        data[col] = data['ret'].shift(lag)
        cols.append(col)
    return data[cols]

def datepart_feat(df0, colname = 'Date'):
    """This function adds some common pandas date parts like 'year',
        'month' etc as features to a dataframe
    """
    df = df0.copy()
    df.reset_index(inplace=True)
    df1 = df.loc[:,colname]
    nu_feats = ['Day', 'Dayofweek', 'Dayofyear']
    
    targ_pre = re.sub('[Dd]ate$', '', colname)
    for n in nu_feats:
        df[targ_pre+n] = getattr(df1.dt,n.lower())

    df[targ_pre+'week'] = df1.dt.isocalendar().week
    df['week'] = np.int64(df['week'])
    df[targ_pre+'Elapsed'] = df1.astype(np.int64) // 10**9
    nu_feats.extend(['week', 'Elapsed'])
    df.set_index(colname, inplace=True)
    return df[nu_feats]

def gen_feat(pair):
    df0 = get_data(pair)
    df0['ret'] = df0['Close'].pct_change()
    df0['dir'] = np.sign(df0['ret'])
    eATR_ = eATR(df0).shift(1)
    wae = WAE(df0).shift(1)
    ssl1 = ssl(df0).shift(1)
    datepart = datepart_feat(df0)
    lags = lag_feat(df0)
    return pd.concat([df0,  eATR_, wae, ssl1, datepart, lags], axis=1).dropna()

# random forest
def rfc(xs, y, n_estimators=40, max_samples=100,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)

def rfc_deploy():
  """This function trains a Random Forest classifier and outputs the 
  out-of-sample performance from the validation and test sets
  """
  df = pd.DataFrame() 
  
  for pair in pairs:
    # retrieving the data and preparing the features
    dataset = gen_feat(pair)
    dataset.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)

    # selecting the features to train on
    cols = list(dataset.columns)
    feats = cols[2:]

    #splitting into training, validation and test sets
    df_train = dataset.iloc[:-100,:]
    train = df_train.copy()
    df_test = dataset.iloc[-100:,:]
    test = df_test.copy()
    train_f = train.iloc[:-100,:]
    valid = train.iloc[-100:,:]

    #training the algorithm
    m = rfc(train_f[feats], train_f['dir'])

    # test sets
    test_pred = m.predict(test[feats])
    test_proba = m.predict_proba(test[feats])

    df1 = pd.DataFrame(test_pred,columns=['prediction'], index=test.index)

    proba_short = []
    proba_long = []
    for x in range(len(test_proba)):
      proba_short.append(test_proba[x][0])
      proba_long.append(test_proba[x][-1])

    proba = {'proba_short': proba_short,
        'proba_long': proba_long}

    df2 = pd.DataFrame(proba, index=test.index)

    df1['probability'] = np.where(df1['prediction'] == 1, df2['proba_long'],
                              np.where(df1['prediction'] == -1, df2['proba_short'], 0))

    df1['signal'] = np.where((df1['probability'] >= .7) & (df1['prediction'] == 1), 'Go Long',
                          np.where((df1['probability'] >= 0.7) & (df1['prediction'] == -1), 'Go Short', 'Stand Aside'))
    
    
    df1.reset_index(inplace=True)
    
    df1['pair'] = pair

    df1.set_index('pair', inplace=True)

    entry_sig = df1[['probability', 'signal']].iloc[-1:]
    
    

    # Merge
    df = pd.concat([df, entry_sig], axis=0)
    
  #output
  return df

# Light GBM
def lgb(xs, y, learning_rate=0.15, boosting_type='gbdt',
        objective='binary', n_estimators=50,
        metric=['auc', 'binary_logloss'],
        num_leaves=100, max_depth= 1,
        **kwargs):
    return LGBMClassifier().fit(xs, y)

def lgb_deploy():
  """This function trains a Light Gradient Boosting Method and outputs the 
  out-of-sample performance from the validation and test sets
  """
  df = pd.DataFrame() 
  
  for pair in pairs:
    # retrieving the data and preparing the features
    dataset = gen_feat(pair)
    dataset.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)

    # selecting the features to train on
    cols = list(dataset.columns)
    feats = cols[2:]

    #splitting into training, validation and test sets
    df_train = dataset.iloc[:-1000,:]
    train = df_train.copy()
    df_test = dataset.iloc[-1000:,:]
    test = df_test.copy()
    train_f = train.iloc[:-1000,:]
    valid = train.iloc[-1000:,:]

    #training the algorithm
    m = lgb(train_f[feats], train_f['dir']);

    # test sets
    test_pred = m.predict(test[feats])
    test_proba = m.predict_proba(test[feats])

    df1 = pd.DataFrame(test_pred,columns=['prediction'], index=test.index)

    proba_short = []
    proba_long = []
    for x in range(len(test_proba)):
      proba_short.append(test_proba[x][0])
      proba_long.append(test_proba[x][-1])

    proba = {'proba_short': proba_short,
        'proba_long': proba_long}

    df2 = pd.DataFrame(proba, index=test.index)

    df1['probability'] = np.where(df1['prediction'] == 1, df2['proba_long'],
                              np.where(df1['prediction'] == -1, df2['proba_short'], 0))

    df1['signal'] = np.where((df1['probability'] >= .7) & (df1['prediction'] == 1), 'Go Long',
                          np.where((df1['probability'] >= 0.7) & (df1['prediction'] == -1), 'Go Short', 'Stand Aside'))
    
    
    df1.reset_index(inplace=True)
    
    df1['pair'] = pair

    df1.set_index('pair', inplace=True)

    entry_sig = df1[['probability', 'signal']].iloc[-1:]
    
    

    # Merge
    df = pd.concat([df, entry_sig], axis=0)
    
  #output
  return df

  # eXtreme Gradient Boosting
def xgb(xs, y):
  return XGBClassifier().fit(xs, y)

def xgb_deploy():
  """This function trains a eXtreme Gradient Boosting Method and outputs the 
  out-of-sample performance from the validation and test sets
  """
  df = pd.DataFrame() 
  
  for pair in pairs:
    # retrieving the data and preparing the features
    dataset = gen_feat(pair)
    dataset.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)

    # selecting the features to train on
    cols = list(dataset.columns)
    feats = cols[2:]

    #splitting into training, validation and test sets
    df_train = dataset.iloc[:-1000,:]
    train = df_train.copy()
    df_test = dataset.iloc[-1000:,:]
    test = df_test.copy()
    train_f = train.iloc[:-1000,:]
    valid = train.iloc[-1000:,:]

    #training the algorithm
    m = xgb(train_f[feats], train_f['dir']);

    
    # test sets
    test_pred = m.predict(test[feats])
    test_proba = m.predict_proba(test[feats])

    df1 = pd.DataFrame(test_pred,columns=['prediction'], index=test.index)

    proba_short = []
    proba_long = []
    for x in range(len(test_proba)):
      proba_short.append(test_proba[x][0])
      proba_long.append(test_proba[x][-1])

    proba = {'proba_short': proba_short,
        'proba_long': proba_long}

    df2 = pd.DataFrame(proba, index=test.index)

    df1['probability'] = np.where(df1['prediction'] == 1, df2['proba_long'],
                              np.where(df1['prediction'] == -1, df2['proba_short'], 0))

    df1['signal'] = np.where((df1['probability'] >= .7) & (df1['prediction'] == 1), 'Go Long',
                          np.where((df1['probability'] >= 0.7) & (df1['prediction'] == -1), 'Go Short', 'Stand Aside'))
    
    
    df1.reset_index(inplace=True)
    
    df1['pair'] = pair

    df1.set_index('pair', inplace=True)

    entry_sig = df1[['probability', 'signal']].iloc[-1:]
    
    

    # Merge
    df = pd.concat([df, entry_sig], axis=0)
    
  #output
  return df

df = pd.concat([xgb_deploy(), lgb_deploy()], axis=1)

st.subheader(f"Today's Signals: {date.today()}")	
st.write(df)