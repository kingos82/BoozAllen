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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


filename = '/home/kingos82/Fourthbrain/BoozAllen/data/N-CMAPSS_DS01-005.h5'
'''
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
'''

def read_h5f(filename):
    #Time tracking, Operation time (min):  0.003
    t = time.process_time()  

    # Load data
    with h5py.File(filename, 'r') as hdf:
            
            # Development set
            W_dev = np.array(hdf.get('W_dev'))             # W
            # Test set
            W_test = np.array(hdf.get('W_test'))           # W    
            # Varnams
            W_var = np.array(hdf.get('W_var'))
            W_var = list(np.array(W_var, dtype='U20'))
            W = np.concatenate((W_dev, W_test), axis=0)  

            # Development set
            X_s_dev = np.array(hdf.get('X_s_dev'))             # W
            # Test set
            X_s_test = np.array(hdf.get('X_s_test'))           # W
            # Varnams
            X_s_var = np.array(hdf.get('X_s_var'))
            X_s_var = list(np.array(X_s_var, dtype='U20'))
            X_s = np.concatenate((X_s_dev, X_s_test), axis=0)  

            # Development set
            X_v_dev = np.array(hdf.get('X_v_dev'))             # W
            # Test set
            X_v_test = np.array(hdf.get('X_v_test'))           # W
            #Varnams
            X_v_var = np.array(hdf.get('X_v_var'))
            X_v_var = list(np.array(X_v_var, dtype='U20'))
            X_v = np.concatenate((X_v_dev, X_v_test), axis=0)  

            # Development set
            T_dev = np.array(hdf.get('T_dev'))             # W
            # Test set
            T_test = np.array(hdf.get('T_test'))           # W
                # Varnams
            T_var = np.array(hdf.get('T_var'))
            T_var = list(np.array(T_var, dtype='U20'))
            T = np.concatenate((T_dev, T_test), axis=0)  

            # Development set
            A_dev = np.array(hdf.get('A_dev'))             # W
            # Test set
            A_test = np.array(hdf.get('A_test'))           # W
            # Varnams
            A_var = np.array(hdf.get('A_var'))
            A_var = list(np.array(A_var, dtype='U20'))
            A = np.concatenate((A_dev, A_test), axis=0)  

            # Development set
            Y_dev = np.array(hdf.get('Y_dev'))             # W
            # Test set
            Y_test = np.array(hdf.get('Y_test'))           # W
            # Varnams
            Y_var = np.array(hdf.get('Y_var'))
            Y = np.concatenate((Y_dev, Y_test), axis=0) 
   
    return W, W_var, X_s, X_s_var, X_v, X_v_var, T, T_var, A, A_var, Y, Y_var


def create_df(X, X_var, param):
    if param=="Y":
        df_X = DataFrame(data=X, columns=["target"])
    else:
        df_X = DataFrame(data=X, columns=X_var)
    df_X["Index"]=[i for i in range(df_X.shape[0])]
    return df_X


def mc(df_X, dfY):
    X0=pd.merge(df_X, dfY, on="Index")
    #X1=X0.corr()
    return X0


def addhs(df_X, A0):
    df_Ahs = DataFrame(data=A0['hs'], columns=['hs'])
    df_Ahs["Index"]=[i for i in range(df_Ahs.shape[0])]
    X=pd.merge(df_X, df_Ahs, on='Index')
    return X

'''
W, W_var=read_h5f(filename, "W")
X_s, X_s_var=read_h5f(filename, "X_s")
X_v, X_v_var=read_h5f(filename, "X_v")
T, T_var=read_h5f(filename, "T")
A, A_var=read_h5f(filename, "A")
Y, Y_var=read_h5f(filename, "Y")

'''

W, W_var, X_s, X_s_var, X_v, X_v_var, T, T_var, A, A_var, Y, Y_var=read_h5f(filename)


#def create_df(np_var):
df_W=create_df(W, W_var, 'W')
df_X_s=create_df(X_s, X_s_var, 'X_s')
df_X_v=create_df(X_v, X_v_var, 'X_v')
df_T=create_df(T, T_var, 'T')
df_A=create_df(A, A_var, 'A')
df_Y=create_df(Y, Y_var, 'Y')

W1=mc(df_W, df_Y)
X_s1=mc(df_X_s, df_Y)
X_v1=mc(df_X_v, df_Y)
T1=mc(df_T, df_Y)
A1=mc(df_A, df_Y)

W2=addhs(W1, df_A)
X_s2=addhs(X_s1, df_A)
X_v2=addhs(X_v1, df_A)
T2=addhs(T1, df_A)



W2hs=sns.heatmap(W2.corr(), annot=True)
figW = W2hs.get_figure()
figW.savefig("W-heatmap.png")
plt.clf()

X_s2hs=sns.heatmap(X_s2.corr(), annot=True)
figX_s = X_s2hs.get_figure()
figX_s.savefig("X_s-heatmap.png")
plt.clf()

X_v2hs=sns.heatmap(X_v2.corr(), annot=True)
figX_v = X_v2hs.get_figure()
figX_v.savefig("X_v-heatmap.png")
plt.clf()

T2hs=sns.heatmap(T2.corr(), annot=True)
figT = T2hs.get_figure()
figT.savefig("T-heatmap.png")
plt.clf()

A2=sns.heatmap(A1.corr(), annot=True)
figA = A2.get_figure()
figA.savefig("A-heatmap.png")
plt.clf()

'''
figW = W2hs.get_figure()
plt.savefig("W-heatmap.png")

figX_s = X_s2hs.get_figure()
plt.savefig("X_s-heatmap.png")

figX_v = X_v2hs.get_figure()
plt.savefig("X_v-heatmap.png")

figT = T2hs.get_figure()
plt.savefig("T-heatmap.png")

figA = A2.get_figure()
plt.savefig("A-heatmap.png")
'''

#logistic regression for relationship between hs and target

y=df_A['hs']
x=df_Y[["target"]]
numeric_features=['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#categorical_variables[]
numeric_transformer=Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
preprocessor=ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])
model_classifier=Pipeline(steps=[("preprocessor", preprocessor),("LogisticRegression", LogisticRegression(random_state=42))])
model_classifier.fit(x_train, y_train) #add y_train.values.ravel() 
y_pred=model_classifier.predict(x_test)
accuracy=metrics.accuracy_score(y_test, y_pred)
balanced_accuracy=metrics.balanced_accuracy_score(y_test, y_pred)

score_train=model_classifier.score(x_train, y_train)
score_test=model_classifier.score(x_test, y_test)
report=metrics.classification_report(y_test, y_pred)

print("balanced accuracy", balanced_accuracy,"score_train", score_train,"score_test", score_test)
print("report")
print(report)

model_name="LogisticRegression"
fig, ax = plt.subplots(figsize=(10, 5))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
_ = ax.set_title(
f"Confusion Matrix for {model_name}"
)

plt.savefig("hs_target_con_mat")
plt.clf()


probs=model_classifier.predict_proba(x_test)
preds=probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc=metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc}')
plt.plot([0,1], [0,1], "r--")
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.savefig("auc_curve")
plt.clf()

#if the correlation number improves if merging all the files into one.
