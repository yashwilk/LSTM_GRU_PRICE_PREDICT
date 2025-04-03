import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv('AMZN.csv')

data=data[['Date','Close']]
device='cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

data['Date']=pd.to_datetime(data['Date'])

plt.plot(data['Date'],data['Close'])
plt.show()


def prepare_dataframe_for_lstm(df,n):
    df=dc(df)
    df['Date']=pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    for i in range(1,n+1):
        df[f'Close(t-{i})']=df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

lookback=7
shifted_df=prepare_dataframe_for_lstm(data,lookback)
print(shifted_df)

shifted_df_as_np=shifted_df.to_numpy()

scaler=MinMaxScaler(feature_range=(-1,1))
shifted_df_as_np=scaler.fit_transform(shifted_df_as_np)
print(shifted_df_as_np)

X=shifted_df_as_np[:,1:]
y=shifted_df_as_np[:,0]
print(X.shape,y.shape)

X=dc(np.flip(X,axis=1))

split_index=int(len(X)*0.95)

X_train=X[:split_index]
X_test=X[split_index:]

y_train=y[:split_index]
y_test=y[split_index:]

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

X_train=X_train.reshape((-1,lookback,1))
X_test=X_test.reshape((-1,lookback,1))
y_train=y_train.reshape((-1,1))
y_test=y_test.reshape((-1,1))

X_train=torch.tensor(X_train).float()
X_test=torch.tensor(X_test).float()
y_train=torch.tensor(y_train).float()
y_test=torch.tensor(y_test).float()


from torch.utils.data import Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
       return self.X[i],self.y[i]
    
train_dataset = TimeSeriesDataset(X_train, y_train)  # Use y_train as target
test_dataset = TimeSeriesDataset(X_test, y_test)  # Correct, use y_test as target



from torch.utils.data import DataLoader
batch_size=16
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

for _,batch in enumerate(train_loader):
    x_batch,y_batch=batch[0].to(device),batch[1].to(device)
    print(x_batch.shape,y_batch.shape)
    break


class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_stacked_layer):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_stacked_layer=num_stacked_layer
        self.lstm=nn.LSTM(input_size,hidden_size,num_stacked_layer,batch_first=True)
        self.fc=nn.Linear(hidden_size,1)

    def forward(self,x):
        h0 = torch.zeros(self.num_stacked_layer, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layer, x.size(0), self.hidden_size).to(device)

        out,_=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        return out
    
lstm_model=LSTM(1,4,1)
lstm_model.to(device)



class GRU(nn.Module):
    def __init__(self, input_size,hidden_size,num_stacked_layer):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_stacked_layer=num_stacked_layer
        self.gru=nn.GRU(input_size,hidden_size,num_stacked_layer,batch_first=True)
        self.fc=nn.Linear(hidden_size,1)

    def forward(self,x):
        h0 = torch.zeros(self.num_stacked_layer, x.size(0), self.hidden_size).to(device)
        out,_=self.gru(x,h0)
        out=self.fc(out[:,-1,:])
        return out
    
gru_model=GRU(1,4,1)
gru_model.to(device)






def train_one_epoch(model,optimizer,train_loader):
    model.train(True)
    print(f'Epoch: {epoch+1}')
    running_loss=0

    for batch_index,batch in enumerate(train_loader):
        x_batch,y_batch=batch[0].to(device),batch[1].to(device)

        output=model(x_batch)
        loss=loss_function(output,y_batch)
        running_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100==99:
            avg_loss_across_batches=running_loss/100
            print('Batch{0},loss:{1:3f}'.format(batch_index+1,avg_loss_across_batches))
    print()

def validate_one_epoch(model,test_loader):
    model.eval()
    running_loss=0

    for batch_index,batch in enumerate(test_loader):
        x_batch,y_batch=batch[0].to(device),batch[1].to(device)

        with torch.no_grad():
            output=model(x_batch)
            loss=loss_function(output,y_batch)
            running_loss+=loss.item()
 

       
        avg_loss_across_batches=running_loss/len(test_loader)
        print('Vloss:{0:3f}'.format(avg_loss_across_batches))
    print()


learning_rate=0.001
num_epochs=10
loss_function=nn.MSELoss()
lstm_optimizer=torch.optim.Adam(lstm_model.parameters(),lr=learning_rate)
gru_optimizer=torch.optim.Adam(gru_model.parameters(),lr=learning_rate)



for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    
    # Train and validate LSTM model
    print("Training LSTM Model")
    train_one_epoch(lstm_model, lstm_optimizer, train_loader)
    lstm_val_loss = validate_one_epoch(lstm_model, test_loader)
    
    # Train and validate GRU model
    print("Training GRU Model")
    train_one_epoch(gru_model, gru_optimizer, train_loader)
    gru_val_loss = validate_one_epoch(gru_model, test_loader)

    
    print('-' * 50)

with torch.no_grad():
    lstm_predicted = lstm_model(X_train.to(device)).to('cpu').numpy()
    gru_predicted = gru_model(X_train.to(device)).to('cpu').numpy()

    # Plot the results
    plt.plot(y_train.numpy(), label='Actual Close')
    plt.plot(lstm_predicted, label='LSTM Predicted Close')
    plt.plot(gru_predicted, label='GRU Predicted Close')
    plt.plot(y_train,label='Actual Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()