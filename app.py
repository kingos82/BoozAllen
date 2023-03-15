import plotly.express as px
from scripts.utils import rename_col, add_rul, minmax_dic, minmax_scl,smooth, smoothing, drop_org
import dash
import joblib
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

################ DATA PREPROCESSING ################

#model=joblib.load('/home/kingos82/Fourthbrain/BoozAllen/model2140.joblib')


df_test = pd.read_csv("/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/test_FD001.txt", header=None, sep = ' ')
df_train = pd.read_csv("/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/train_FD001.txt", header=None, sep = ' ')



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

#y = model.predict(df_test.iloc[0:1,2:])
############# APP LAYOUT #############

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Intialize app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Engine prediction")
server = app.server
app.layout=html.Div(
    children=[html.H2('RUL',),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Engine 1', 'value': '1'},
            {'label': 'Engine 2', 'value': '2'},
            {'label': 'Engine 3', 'value': '3'},
            {'label': 'Engine 4', 'value': '4'},
            {'label': 'Engine 5', 'value': '5'}]),
    dcc.Graph(id='graph')

    ]
)

@app.callback(
    Output('graph', 'figure'),
    Input('dropdown', 'value')
)
def generate_plot(value):
    fig = px.line(x=df_test['time'], y=df_test['rul'])
    return fig
if __name__ == '__main__':

    app.run_server(debug=True)