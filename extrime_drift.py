import yfinance,numpy
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

yfinance.pdr_override()
X_train,y_train=[],[]
data_devil=numpy.array(yfinance.download("RELIANCE.NS")["Close"])
print(len(data_devil))
for i in range(0,len(data_devil),5):
	if (i+5)-1<len(data_devil):
		X_train.append(data_devil[i:i+4])
		y_train.append(data_devil[(i+4)])
X_train,y_train=np.array(X_train),torch.from_numpy(np.array(y_train).astype(np.float32))
X_train=torch.from_numpy(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]).astype(np.float32))
def checker_devil(devil_in,t=0,f=0):
	for i in devil_in:
		if i==True:
			t+=1
		else:
			f+=1
	if t>=len(devil_in)-(len(devil_in)/100):
		return True
	else:
		return False

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=200, num_layers=5)
        self.linear = nn.Linear(200, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = LSTM()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
epoch=0

while True:
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(torch.reshape(y_pred,(-1,)),torch.reshape(y_train,(-1,)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch)
    epoch+=1
    if epoch%100==0:
    	print(y_pred)
    if checker_devil((torch.reshape(y_pred,(-1,))>(y_train-3)).numpy()):
    	if checker_devil((torch.reshape(y_pred,(-1,))<=(y_train)).numpy()):
    		print(y_pred)
    		break
    	else:
    		continue
    else:
    	continue
print(y_train)