import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Load CSV
data = pd.read_csv('data.csv')

# Convert 'shopping_mall', 'gender', and 'category' to dummy variables
data = pd.get_dummies(data, columns=['shopping_mall', 'gender', 'category'], drop_first=True)
y = data['price']
# Select features for the model
dummy_columns = [col for col in data.columns if 'shopping_mall_' in col or 'gender_' in col or 'category_' in col]
features = dummy_columns
X = data[features]


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Gradient Boosting Regressor
reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
reg.fit(X_scaled, y)

# Dropdown options
dropdown_options = [{'label': feature, 'value': feature} for feature in features] + [{'label': 'All Features', 'value': 'All'}]

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='feature-dropdown',
        options=dropdown_options,
        value='gender_Male',
        multi=True
    ),
    dcc.Graph(id='feature-graph'),
    html.Div(id='model-metrics')
])

@app.callback(
    [Output('feature-graph', 'figure'),
     Output('model-metrics', 'children')],
    [Input('feature-dropdown', 'value')]
)
def update_graph(selected_features):
    if 'All' in selected_features:
        X_selected = X.copy()
    else:
        X_selected = X.copy()
        for column in X.columns:
            if column not in selected_features:
                X_selected[column] = X[column].mean()

    predictions = reg.predict(scaler.transform(X_selected))

    mask = X[[selected_features]].eq(1).all(axis=1) if isinstance(selected_features, str) else X[selected_features].eq(1).all(axis=1)

    filtered_y = y[mask]
    filtered_predictions = predictions[mask]
    if len(filtered_y) == 0:
        return go.Figure(), "No data available for selected features"

    df_viz = pd.DataFrame({
        'Actual Price': filtered_y,
        'Predicted Price': filtered_predictions
    })

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_viz['Actual Price'], y=df_viz['Predicted Price'], mode='lines+markers', name='Predicted Price'))
    fig.add_trace(go.Scatter(x=df_viz['Actual Price'], y=df_viz['Actual Price'], mode='lines', name='Actual Price'))

    mae = mean_absolute_error(df_viz['Actual Price'], df_viz['Predicted Price'])
    mse = mean_squared_error(df_viz['Actual Price'], df_viz['Predicted Price'])
    r2 = r2_score(df_viz['Actual Price'], df_viz['Predicted Price'])

    metrics_text = [
        html.P(f"Min Predicted Value: {predictions.min()}"),
        html.P(f"Max Predicted Value: {predictions.max()}"),
        html.P(f"Standard Deviation of Predictions: {predictions.std()}"),
        html.P(f"Min Actual Value: {df_viz['Actual Price'].min()}"),
        html.P(f"Max Actual Value: {df_viz['Actual Price'].max()}"),
        html.P(f"Standard Deviation of Actual Values: {df_viz['Actual Price'].std()}"),
        html.P(f"Mean Absolute Error: {mae}"),
        html.P(f"Mean Squared Error: {mse}"),
        html.P(f"R^2 Score: {r2}")
    ]

    return fig, metrics_text

app.run_server(debug=True)
