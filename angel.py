from tensorflow.keras.models import load_model
import yfinance,numpy
import numpy as np
import pandas
import torch
from os import listdir,mkdir
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM

yfinance.pdr_override()
X_train,y_train,t=[],[],"IOB"
data_devil=yfinance.download(f"{t}.NS")["Close"]
print(data_devil)

lstm=load_model("D_PREDICTER/IOB/sub/1091_727_727.keras")
lstm.summary()

ang=numpy.array([numpy.array(data_devil[len(data_devil)-4::]).reshape(1, 4)])
print(ang)
print(lstm.predict(ang).reshape(-1))
#y_pred = 
    
