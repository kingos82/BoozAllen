import os
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec

filename = '/home/kingos82/Fourthbrain/BoozAllen/data/N-CMAPSS_DS01-005.h5'

def read_h5f(filename, parm_name):
    #Time tracking, Operation time (min):  0.003
    t = time.process_time()  

    # Load data
    with h5py.File(filename, 'r') as hdf:
            # Development set
            X_dev = np.array(hdf.get(f'{parm_name}_dev'))             # W

            # Test set
            X_test = np.array(hdf.get(f'{parm_name}_test'))           # W
            
                # Varnams
            X_var = np.array(hdf.get(f'{parm_name}_var'))

            if parm_name!="Y":
                # from np.array to list dtype U4/U5
                X_var = list(np.array(X_var, dtype='U20'))
                            
    X = np.concatenate((X_dev, X_test), axis=0)  
    
    return X, X_var


def create_df(X, X_var):
    if Y!="Y":
        df_X = DataFrame(data=X, columns=["target"])
    else:
        df_X = DataFrame(data=X, columns=X_var)
    df_X["Index"]=[i for i in range(df_X.shape[0])]
    return df_X


def mc(df_X, dfY):
    X0=pd.merge(df_X, dfY, on="Index")
    X1=X0.corr()
    return X1


W, W_var=read_h5f(filename, "W")
X_s, X_s_var=read_h5f(filename, "X_s")
X_v, X_v_var=read_h5f(filename, "X_v")
T, T_var=read_h5f(filename, "T")
A, A_var=read_h5f(filename, "A")
Y, Y_var=read_h5f(filename, "Y")

#def create_df(np_var):
df_W=create_df(W, W_var)
df_X_s=create_df(X_s, X_s_var)
df_X_v=create_df(X_v, X_v_var)
df_T=create_df(T, T_var)
df_A=create_df(A, A_var)
df_Y=create_df(Y, Y_var)

W1=mc(df_W, df_Y)
X_s1=mc(df_X_s, df_Y)
X_v1=mc(df_X_v, df_Y)
T1=mc(df_T, df_Y)
A1=mc(df_A, df_Y)

sns.heatmap(A1, annot=True)
