import pandas as pd
import xgboost as xgb
import joblib
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('customer_shopping_data.csv')

# Select relevant features
selected_features = ['age', 'price', 'quantity', 'gender']

X = df[selected_features]
# Create dummy variables for gender
X = pd.get_dummies(X, columns=['gender'], drop_first=True)

# Define target variable
y = df['quantity']

# Create and train the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, max_depth=10, min_child_weight=1, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'quantity_prediction_model.pkl')

# Calculate predicted quantities
predicted_quantities = model.predict(X)

# Create a dashboard to showcase the sales prediction
app = dash.Dash(__name__)
print(app)
app.layout = html.Div([
    html.H1('Sales Prediction Dashboard'),
    html.P("This dashboard demonstrates the power of Machine Learning to predict sales quantities."),
    dcc.Graph(
        id='line-plot',
        figure=px.line(x=y, y=predicted_quantities, labels={'x': 'Actual Quantity', 'y': 'Predicted Quantity'},
                       title='Sales Prediction')
    ),
    html.Div([
        html.Label('Select a feature to visualize:'),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': col, 'value': col} for col in selected_features],
            value='age'
        )
    ]),
    dcc.Graph(id='feature-histogram'),
    dcc.Graph(id='scatter-plot',
              figure=px.scatter(x=y, y=predicted_quantities, labels={'x': 'Actual Quantity', 'y': 'Predicted Quantity'},
                                title='Scatter Plot')),
    dcc.Markdown('''
        **Additional Information**

        This dashboard uses Machine Learning to predict sales quantities     ''')
])


@app.callback(Output('feature-histogram', 'figure'), Input('feature-dropdown', 'value'))
def update_histogram(selected_feature):
    feature_data = df[selected_feature]
    histogram = px.histogram(feature_data, x=selected_feature, title=f'Histogram of {selected_feature}')
    return histogram


if __name__ == '__main__':
    app.run_server(debug=True)
