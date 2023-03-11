import numpy as np
import pandas as pd

def rename_col(df):
    '''
    this function creates new columns and assigns them to an exisiting data frame
    Let's attach column names: 3 operational setting columns (os + number), and 21 sensor columns (s + number). 
    Let's drop last 2 columns with NaNs
    '''
    col_names = []
    col_names.append('unit')
    col_names.append('time')
    for i in range(1,4):
        col_names.append('os'+str(i))
    for i in range(1,22):
        col_names.append('s'+str(i))
    df = df.iloc[:,:-2].copy()
    df.columns = col_names
    return df

def add_rul(df, test_or_train):
    '''
    attaching remaining useful lifetime to the dataset
    '''
    rul_list = []

    df = df.copy()

    for n in np.arange(1,101):

        time_list = np.array(df[df['unit'] == n]['time'])
        length = len(time_list)
        if test_or_train=='test':
            #print(df.iloc[n-1].tolist()[0], n)
            rul_val = df.iloc[n-1].tolist()[0]
            rul = list(length - time_list + rul_val)
        elif test_or_train=='train':
            rul = list(length - time_list)
        else:
            print('test_or_train must be "test" or "train"')
            return
        rul_list += rul

    df['rul'] = rul_list

    return df


def minmax_dic(df):
    minmax_dict={}
    for c in df.columns:
        if 's' in c:
            minmax_dict[c+'min'] = df[c].min()
            minmax_dict[c+'max']=  df[c].max()
    return minmax_dict

def minmax_scl(df, dict):
    '''
    minmax scale
    '''
    df = df.copy()
    for c in df.columns:
        if 's' in c:
            df[c] = (df[c] - dict[c+'min']) / (dict[c+'max'] - dict[c+'min'])

    return df


def smooth(s, b = 0.98):
    '''
    Smoothing Function: Exponentially Weighted Averages
    '''

    v = np.zeros(len(s)+1) #v_0 is already 0.
    bc = np.zeros(len(s)+1)
    for i in range(1, len(v)): #v_t = 0.95
        v[i] = (b * v[i-1] + (1-b) * s[i-1]) 
        bc[i] = 1 - b**i

    sm = v[1:] / bc[1:]
    return sm


def smoothing(df):
    '''
    Smoothing each time series for each engine in both training and test sets
    '''
    df = df.copy()
    for c in df.columns:
        sm_list = []
        if 's' in c:
            for n in np.arange(1,101):
                s = np.array(df[df['unit'] == n][c].copy())
                sm = list(smooth(s, 0.98))
                sm_list += sm
            df[c+'_smoothed'] = sm_list
    return df

def drop_org(df):
    '''
    drop original column leaving smooth data
    '''
    df = df.copy()
    for c in df.columns:
        if ('s' in c) and ('smoothed' not in c):
            df[c] = df[c+'_smoothed']
            df.drop(c+'_smoothed', axis = 1, inplace = True)
    return df