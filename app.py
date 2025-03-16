import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb  # type: ignore
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the dataset (Replace 'train.csv' with the actual file path)
data = pd.read_csv('train.csv')

# Preprocess the data
# Encoding categorical columns
label_encoders = {}
categorical_columns = ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and Target
X = data.drop(['ID', 'result', 'Class/ASD'], axis=1)
y = data['Class/ASD']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'asd_xgboost_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        user_input = np.array([[
            int(request.form['A1_Score']), int(request.form['A2_Score']),
            int(request.form['A3_Score']), int(request.form['A4_Score']),
            int(request.form['A5_Score']), int(request.form['A6_Score']),
            int(request.form['A7_Score']), int(request.form['A8_Score']),
            int(request.form['A9_Score']), int(request.form['A10_Score']),
            float(request.form['age']), int(request.form['gender']),
            int(request.form['ethnicity']), int(request.form['jaundice']),
            int(request.form['austim']), int(request.form['contry_of_res']),
            int(request.form['used_app_before']), int(request.form['age_desc']),
            int(request.form['relation'])
        ]])

        # Load the trained model and make prediction
        model = joblib.load('asd_xgboost_model.pkl')
        prediction = model.predict(user_input)

        result = "Positive (Autism Spectrum Disorder)" if prediction == 1 else "Negative (No Autism Spectrum Disorder)"
        return redirect(url_for('result', prediction=result))
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
