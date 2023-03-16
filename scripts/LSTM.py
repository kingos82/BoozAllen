# Loading the libraries data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import sleep
from os import listdir
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation as FA
import os
import sys
import joblib
import pickle

sys.path.append(os.path.abspath(os.path.join('.','./CMAPSSData/')))


from utils import rename_col, add_rul, \
        minmax_dic, minmax_scl,smooth, \
            smoothing, drop_org, \
            LSTMRegressor, device, learning_rate, n_hidden_units


def validation():
    model.eval()
    X, y = next(iter(valloader))
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    
    with torch.no_grad():
        y_pred = model(X)
        val_loss = loss_fn(y_pred, y).item()
        
    return val_loss


def test_model(model, testloader, device):
    loss_L1 = nn.L1Loss()
    model.eval()
    X, y = next(iter(testloader))
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    
    with torch.no_grad():
        y_pred = model(X)
        test_loss_MSE = loss_fn(y_pred, y).item()
        test_loss_L1 = loss_L1(y_pred, y).item()
        
    return test_loss_MSE, test_loss_L1,  y_pred, y

class data(Dataset):
    
    def __init__(self, list_indices, df_train):
        
        self.indices = list_indices
        self.df_train = df_train
        
    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        
        ind = self.indices[idx]
        X_ = self.df_train.iloc[ind : ind + 20, :].drop(['time','unit','rul'], axis = 1).copy().to_numpy()
        y_ = self.df_train.iloc[ind + 19]['rul']
        
        return X_, y_
    

class test(Dataset):
    
    def __init__(self, units, df_test):
        
        self.units = units
        self.df_test = df_test
        
    def __len__(self):
        
        return len(self.units)
    
    def __getitem__(self, idx):
        
        n = self.units[idx]
        U = self.df_test[self.df_test['unit'] == n].copy()
        X_ = U.reset_index().iloc[-20:,:].drop(['index','unit','time','rul'], axis = 1).copy().to_numpy()
        y_ = U['rul'].min()
        
        return X_, y_
    

if __name__=="__main__":

    df_train = pd.read_csv("./CMAPSSData/train_FD001.txt", header=None, sep = ' ')
    df_test = pd.read_csv("./CMAPSSData/test_FD001.txt", header=None, sep = ' ')
    rul_test = pd.read_csv("./CMAPSSData/RUL_FD001.txt", header=None)


    ## Refactor data wrangling commands
    df_train=rename_col(df_train)
    df_test=rename_col(df_test)

    df_train=add_rul(df_train, 'train')
    df_test=add_rul(df_test, 'test')


    #Drop os3, s1, s5, s6, s10, s16, s18, s19 from both train and test

    drop_cols1 = ['os3','s1','s5','s6','s10','s16','s18','s19']
    df_train = df_train.drop(drop_cols1, axis = 1)
    df_test = df_test.drop(drop_cols1, axis = 1)

    #minmax scale the sensor values
    minmax_dict=minmax_dic(df_train)
    df_train=minmax_scl(df_train, minmax_dict)
    df_test=minmax_scl(df_test, minmax_dict)

    #smoothing the training & test data
    df_train=smoothing(df_train)
    df_test=smoothing(df_test)

    #drop original data
    df_train=drop_org(df_train)
    df_test=drop_org(df_test)

    n_features = len([c for c in df_train.columns if 's' in c])
    window = 20

    #splitting train and validation datasets
    np.random.seed(5)
    units = np.arange(1,101)
    train_units = list(np.random.choice(units, 80, replace = False))
    val_units = list(set(units) - set(train_units))
    print('validation units:', val_units)

    # Preparing Training, Validation and Test Dataloaders
    train_data = df_train[df_train['unit'].isin(train_units)].copy()
    val_data = df_train[df_train['unit'].isin(val_units)].copy()

    train_indices = list(train_data[(train_data['rul'] >= (window - 1)) & (train_data['time'] > 10)].index)
    val_indices = list(val_data[(val_data['rul'] >= (window - 1)) & (val_data['time'] > 10)].index)

    train = data(train_indices, df_train)
    val = data(val_indices, df_train)
    test = test(units, df_test)

    torch.manual_seed(5)
    trainloader = DataLoader(train, batch_size = 64, shuffle = True)
    valloader = DataLoader(val, batch_size = len(val_indices), shuffle = True)
    testloader = DataLoader(test, batch_size = 100)

    

    torch.manual_seed(15)

    model = LSTMRegressor(n_features, n_hidden_units).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                                
    ks = [key for key in model.state_dict().keys() if 'linear' in key and '.weight' in key]

    for k in ks:
        nn.init.kaiming_uniform_(model.state_dict()[k])
        
    bs = [key for key in model.state_dict().keys() if 'linear' in key and '.bias' in key]

    for b in bs:
        nn.init.constant_(model.state_dict()[b], 0)

    #Training using ADAM optimizer for 100 epochs with learning rate =0.001
    T = []
    V = []

    epochs = 100

    model.train()

    for i in tqdm(range(epochs)):
        
        L = 0
        
        for batch, (X,y) in enumerate(trainloader):
            
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            L += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss = validation()
        
        T.append(L/len(trainloader))
        V.append(val_loss)
        
        if (i+1) % 10 == 0:
            sleep(0.5)
            print(f'epoch:{i+1}, avg_train_loss:{L/len(trainloader)}, val_loss:{val_loss}')
            
            model.train()



    #fout_name = "model230310.joblib"


    #joblib.dump(model, open("jbl230312.joblib", "wb"))
    #model=joblib.load(open("jbl230312.joblib", "rb"))

    #pickle.dump(model, open("pkl230312a.pkl", "wb"))
    #model=pickle.load(open("pkl230312a.pkl", "rb"))

    #joblib.dump(model, 'pkl230312b.pkl.pkl')
    #model = joblib.load('pkl230312b.pkl.pkl')


    
    
    #joblib.dump(model, fout_name)


    model_path = "model2140_1.pt"
    torch.save(model.state_dict(), model_path)
    mse, l1, y_pred, y = test_model(model, testloader, device)

    print(f'Test MSE:{round(mse,2)}, L1:{round(l1,2)}')





    ## Set up functions for building model

    ## Train the model

    ## Save the model