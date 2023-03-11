import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Intialize app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Engine prediction")
server = app.server
app.layout=html.Div(
    children=[html.H2('RUL')]
)

if __name__ == '__main__':

    app.run_server(debug=True)