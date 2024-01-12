import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import xgboost as xgb 

app = Flask(__name__)

#CSV data
df = pd.read_csv('Forecast-Energy-Consumption/data/PJME_hourly.csv') 
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

#time series features
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)

# Train/test split
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

# Train XGBoost model
FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

reg = xgb.XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    objective='reg:squarederror',
    max_depth=3,
    learning_rate=0.01
)

reg.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=100
)

# Save the trained model
model_filename = 'xgboost_model.joblib'
joblib.dump(reg, model_filename)

#trained model
loaded_model = joblib.load(model_filename)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        date_str = request.form['date']
        hour = int(request.form['hour'])

        # Create a DataFrame with required features
        input_data = pd.DataFrame({
            'hour': [hour],
            'dayofweek': pd.to_datetime(date_str).dayofweek,
            'quarter': pd.to_datetime(date_str).quarter,
            'month': pd.to_datetime(date_str).month,
            'year': pd.to_datetime(date_str).year
        })

        # Predict energy consumption
        predicted_value = loaded_model.predict(input_data)[0]
    except Exception as e:
        print(f"Error predicting: {e}")
        predicted_value = None

    return render_template('index.html', prediction=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)