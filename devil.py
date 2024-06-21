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
print(listdir("D_PREDICTER"))
if t not in listdir("D_PREDICTER"):
    mkdir(f"D_PREDICTER/{t}")
    mkdir(f"D_PREDICTER/{t}/sub")
print(data_devil)
for i in range(0,len(data_devil),5):
    if (i+5)-1<len(data_devil):
        X_train.append(data_devil[i:i+4])
        y_train.append(data_devil[(i+4)])
X_train,y_train=np.array(X_train),np.array(y_train).astype(np.float32)
X_train=X_train.reshape(X_train.shape[0], 1, X_train.shape[1]).astype(np.float32)
high_devil_,rec_devil_=0,0

def checker_devil(devil_in,angel_in,t=0,f=0):
    global rec_devil_
    for i in zip(devil_in,angel_in):
        if i[0]==True and i[1]==True:
            t+=1
        else:
            f+=1
    rec_devil_=t
    if t>=len(devil_in)-(len(devil_in)/100):
        return True
    else:
        return False


lstm=Sequential()
lstm.add(LSTM(32, input_shape=(X_train.shape[1],4), activation="relu", return_sequences=False))
lstm.add(Dense(1))
lstm.add(Dense(1))
lstm.compile(loss="mean_squared_error", optimizer="adam")

epoch=0

while True:
    lstm.fit(X_train, y_train, epochs=1, batch_size=8, verbose=0, shuffle=False)
    y_pred = lstm.predict(X_train).reshape(-1)
    epoch+=1
    if epoch%100==0:
        print(epoch)
        print(y_pred)
    if checker_devil(((y_pred.reshape((-1,))>(y_train-1))),((y_pred.reshape((-1,))<(y_train+1)))):
        print(y_pred)
        break
    else:
        if rec_devil_>high_devil_:
            print(f"{len(y_train)}-{high_devil_}-{rec_devil_}")
            high_devil_=rec_devil_
            lstm.save(f"D_PREDICTER/{t}/sub/{len(y_train)}_{high_devil_}_{rec_devil_}.keras")
            lstm.save(f"D_PREDICTER/{t}/high.keras")
        continue

print(y_train)
