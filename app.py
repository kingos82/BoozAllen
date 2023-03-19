import plotly.express as px
from scripts.utils import rename_col, add_rul, minmax_dic, minmax_scl,smooth, smoothing, drop_org
import dash
import joblib
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import torch 
from scripts.utils import LSTMRegressor, device, learning_rate, n_hidden_units
import plotly.graph_objects as go
import numpy as np
################ DATA PREPROCESSING ################




def plt_rlu(y, y_pred):
    fig = go.Figure()

    fig.add_trace(
    go.Scatter(
        x =np.arange(len(y_pred)),
        y = y_pred,
        mode = 'lines', # Change the mode in this section!
        name='prediction'
        )
    )

    fig.add_trace(
    go.Scatter(
        x =np.arange(len(y)),
        y = y,
        mode = 'lines', # Change the mode in this section!
        name='True'
        )
    )
    return fig


def choose_model(input):
    if input =="model1":
        model_path="/home/kingos82/Fourthbrain/BoozAllen/model2140_1.pt"
        train_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/train_FD001.txt"
    elif input=='model2':
        model_path="/home/kingos82/Fourthbrain/BoozAllen/model2140_2.pt"
        train_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/train_FD002.txt"
    elif input=='model3':
        model_path="/home/kingos82/Fourthbrain/BoozAllen/model2140_3.pt"
        train_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/train_FD003.txt"
    elif input=='model4':
        model_path="/home/kingos82/Fourthbrain/BoozAllen/model2140_4.pt"
        train_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/train_FD004.txt"
    return model_path, train_path



def choose_test(input):
    if input =="test1":
        test_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/test_FD001.txt"
        RLU_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/RUL_FD001.txt"
    elif input=='test2':
        test_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/test_FD002.txt"
        RLU_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/RUL_FD002.txt"
    elif input=='test3':
        test_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/test_FD003.txt"
        RLU_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/RUL_FD003.txt"
    elif input=='test4':
        test_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/test_FD004.txt"
        RLU_path="/home/kingos82/Fourthbrain/BoozAllen/CMAPSSData/RUL_FD004.txt"
    return test_path, RLU_path






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

# Instantiate the model
n_features = len([c for c in df_train.columns if 's' in c])
loaded_model = LSTMRegressor(n_features, n_hidden_units)

# Load the saved state_dict
model_path = "model2140_1.pt"
loaded_model.load_state_dict(torch.load(model_path))

############# APP LAYOUT #############

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Define the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Remaining Useful Life Prediction", className="text-center"))
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select a unit number:"),
            #dcc.Input(id="unit-input", type="number", value=1, min=1, max=len(df_test["unit"].unique())),
            #html.Br(),
            dbc.Dropdown(
                id='y',
                options=['model1', 'model2', 'model3', 'model4'],
                value='model')
                ],className="six_columns"),
        dbc.Col([
            html.Label("select test data:"),
            dcc.Dropdown(
                id='y_pred',
                options=['test1', 'test2', 'test3', 'test4'],
                value='test data'),
                ], className="six columns"),
            ], className='row'),
        dbc.Col([
            dcc.Graph(id="rul-graph")
            ], className="four columns")
    , className="row",
])

@app.callback(
        Output("rul_graph", "figure"),
        Input('y', 'value'),
        Input("y_pred", 'value'))
def update_figure(y, y_pred):
    fig = plt_rlu(y, y_pred)
    return fig


if __name__ == '__main__':

    app.run_server(debug=True)