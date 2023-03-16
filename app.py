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
################ DATA PREPROCESSING ################


# Read data
df_test = pd.read_csv("d:/Documents-folders/GitHub/BoozAllen/CMAPSSData/test_FD001.txt", header=None, sep = ' ')
df_train = pd.read_csv("d:/Documents-folders/GitHub/BoozAllen/CMAPSSData/train_FD001.txt", header=None, sep = ' ')


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
            dcc.Input(id="unit-input", type="number", value=1, min=1, max=len(df_test["unit"].unique())),
            html.Br(),
            dbc.Button("Predict RUL", id="predict-button", color="primary", className="mt-3")
        ], width=6, xs=12),
        dbc.Col([
            dcc.Graph(id="rul-graph")
        ], width=6, xs=12)
    ], className="mt-4")
])

# Define the callback function for the predict button
@app.callback(
    Output("rul-graph", "figure"),
    Input("predict-button", "n_clicks"),
    Input("unit-input", "value")
)
def predict_rul(n_clicks, unit):
    if n_clicks:
        # Get the data for the selected unit
        unit_data = df_test[df_test["unit"] == unit].copy()
        unit_data = unit_data.reset_index(drop=True)

        # Get the initial sensor readings for the unit
        x = unit_data.iloc[0, :-2].values.reshape(1, -1)
        x = torch.tensor(x, dtype=torch.float32).to(device)

        # Modify the input shape to add a batch size dimension
        x = x.unsqueeze(0)

        # Initialize the hidden state and cell state with batch size of 1
        hx = torch.zeros(1, 1, n_hidden_units).to(device)
        cx = torch.zeros(1, 1, n_hidden_units).to(device)

        # Predict the RUL for each timestep in the unit
        predictions = []
        for i in range(len(unit_data)):
            with torch.no_grad():
                y, (hx, cx) = loaded_model(x, hx, cx)
                y = y.cpu().numpy()[0][0]
                predictions.append(y)

            # Update the input sequence with the latest sensor readings
            x = torch.tensor(unit_data.iloc[i, :-2].values.reshape(1, -1), dtype=torch.float32).to(device)
            x = x.unsqueeze(0)

        # Add the predicted RUL to the unit data
        unit_data["rul_predicted"] = predictions

        # Create a line chart of the predicted RUL
        fig = px.line(unit_data, x="time_in_cycles", y="rul_predicted")

        return fig

    else:
        # Return an empty figure if the button has not been clicked yet
        return {}
if __name__ == '__main__':

    app.run_server(debug=True)