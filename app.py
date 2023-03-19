import plotly.express as px
from scripts.utils import rename_col, add_rul, minmax_dic,\
      minmax_scl,smooth, smoothing, drop_org, \
        LSTMRegressor, n_hidden_units, test, test_model,\
              device
import dash
from pathlib import Path
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import torch 
from torch.utils.data import DataLoader
from scripts.utils import LSTMRegressor, device, learning_rate, n_hidden_units
import plotly.graph_objects as go
import numpy as np
################ DATA PREPROCESSING ################
def choose_model(input):
    if input =="model1":
        model_path="./model2140_1.pt"
        train_path="./CMAPSSData/train_FD001.txt"
    elif input=='model2':
        model_path="./model2140_2.pt"
        train_path="./CMAPSSData/train_FD002.txt"
    elif input=='model3':
        model_path="./model2140_3.pt"
        train_path="./CMAPSSData/train_FD003.txt"
    elif input=='model4':
        model_path="./model2140_4.pt"
        train_path="./CMAPSSData/train_FD004.txt"
    return model_path, train_path



def choose_test(input):
    if input =="test1":
        test_path="./CMAPSSData/test_FD001.txt"
        RLU_path="./CMAPSSData/RUL_FD001.txt"
    elif input=='test2':
        test_path="./CMAPSSData/test_FD002.txt"
        RLU_path="./CMAPSSData/RUL_FD002.txt"
    elif input=='test3':
        test_path="./CMAPSSData/test_FD003.txt"
        RLU_path="./CMAPSSData/RUL_FD003.txt"
    elif input=='test4':
        test_path="./CMAPSSData/test_FD004.txt"
        RLU_path="./CMAPSSData/RUL_FD004.txt"
    return test_path, RLU_path


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

    # Update figure to have title
    fig.update_layout(
        title="RUL prediction",
        xaxis_title="Number of units",
        yaxis_title="RUL",
        font=dict(
            size=18,
            color="black"
        )
    )
    
    return fig

def wrangle_data(input_path): 

        
    df = pd.read_csv(Path(input_path), header=None, sep = ' ')
    
    ## Refactor data wrangling commands
    df=rename_col(df)
    df=add_rul(df, 'train')

    #Drop os3, s1, s5, s6, s10, s16, s18, s19 from both train and test
    drop_cols1 = ['os3','s1','s5','s6','s10','s16','s18','s19']
    df = df.drop(drop_cols1, axis = 1)

    #minmax scale the sensor values
    minmax_dict=minmax_dic(df)
    df=minmax_scl(df, minmax_dict)

    #smoothing the training & test data
    df=smoothing(df)

    #drop original data
    df=drop_org(df)

    return df
    
def get_y_true_and_pred(model_input, test_input):

    model_path, train_path = choose_model(model_input)
    test_path, RUL_path = choose_test(test_input)

    df_train = wrangle_data(train_path)
    df_test = wrangle_data(test_path)

    # Instantiate the model
    n_features = len([c for c in df_train.columns if 's' in c])
    loaded_model = LSTMRegressor(n_features, n_hidden_units)

    # Load the saved state_dict
    full_model_path = Path(model_path) 
    loaded_model.load_state_dict(torch.load(full_model_path))

    eng_num=df_test['unit'].max()+1
    units = np.arange(1,eng_num)

    test_data = test(units, df_test)

    torch.manual_seed(5)

    testloader = DataLoader(test_data, batch_size = 100)
    mse, l1, y_pred, y = test_model(loaded_model, testloader, device)

    df_RUL = pd.read_csv(Path(RUL_path), header=None, sep = ' ')
    y=df_RUL[0].to_list()

    return y, y_pred

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
            dcc.Dropdown(
                id='model-n',
                options=['model1', 'model2', 'model3', 'model4'],
                value='model1')
                ],className="six_columns"
                ),
        dbc.Col([
            html.Label("select test data:"),
            dcc.Dropdown(
                id='test-n',
                options=['test1', 'test2', 'test3', 'test4'],
                value='test1'),
                ], className="six columns"),
            ], className='row'),
        dbc.Col([
            dcc.Graph(id="rul-graph")
            ], className="four columns")

])

@app.callback(
        Output("rul-graph", "figure"),
        Input('model-n', 'value'),
        Input("test-n", 'value'))
def update_figure(model_n, test_n):
    print(model_n, test_n)
    y, y_pred = get_y_true_and_pred(model_n, test_n)
    fig = plt_rlu(y, y_pred)
    return fig


if __name__ == '__main__':

    app.run_server(debug=True)