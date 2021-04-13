import os
import re
import tpqoa

from fastbook import *
  
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
os.environ['PYTHONHASHSEED'] = '0'

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from deploy import *

pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF', 
        'AUD_USD', 'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY',
        'GBP_JPY', 'CHF_JPY', 'GBP_CHF', 'EUR_AUD', 'EUR_CAD', 
        'AUD_CAD', 'AUD_JPY', 'CAD_JPY', 'NZD_JPY', 'GBP_CAD', 
        'GBP_NZD', 'GBP_AUD', 'AUD_NZD',  'AUD_CHF', 'EUR_NZD',
        'NZD_CHF', 'CAD_CHF', 'NZD_CAD',  'EUR_CHF']

def get_data(instr, gran = 'D', td=1000):
    start = f"{date.today() - timedelta(td) }"      
    end = f"{date.today() - timedelta(1)}"        
    granularity = gran          
    price = 'M'
    data = api.get_history(instr, start, end, granularity, price)
    data.drop(['complete'], axis=1, inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns = {'time':'Date','o':'Open','c': 'Close', 'h':'High', 'l': 'Low'}, inplace = True)
    data.set_index('Date', inplace=True)
    return data


df = pd.concat([xgb_deploy(), lgb_deploy(), rfc_deploy()], axis=1)

st.write(df)
