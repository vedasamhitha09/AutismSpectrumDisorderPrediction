import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the dataset (Replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('train.csv')

# Preprocess the data
# Encoding categorical columns
label_encoders = {}
categorical_columns = ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation']

# Applying LabelEncoder to all categorical text columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and Target
X = data.drop(['ID', 'result', 'Class/ASD'], axis=1)
y = data['Class/ASD']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model for later use
joblib.dump(model, 'asd_random_forest_model.pkl')

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
        a1_score = int(request.form['A1_Score'])
        a2_score = int(request.form['A2_Score'])
        a3_score = int(request.form['A3_Score'])
        a4_score = int(request.form['A4_Score'])
        a5_score = int(request.form['A5_Score'])
        a6_score = int(request.form['A6_Score'])
        a7_score = int(request.form['A7_Score'])
        a8_score = int(request.form['A8_Score'])
        a9_score = int(request.form['A9_Score'])
        a10_score = int(request.form['A10_Score'])
        age = float(request.form['age'])
        gender = int(request.form['gender'])  # 1 for Male, 0 for Female
        ethnicity = int(request.form['ethnicity'])
        jaundice = int(request.form['jaundice'])
        austim = int(request.form['austim'])
        country_of_res = int(request.form['contry_of_res'])
        used_app_before = int(request.form['used_app_before'])
        age_desc = int(request.form['age_desc'])
        relation = int(request.form['relation'])

        # Create input array for the model
        user_input = np.array([[a1_score, a2_score, a3_score, a4_score, a5_score, a6_score, a7_score, a8_score, a9_score, a10_score,
                                age, gender, ethnicity, jaundice, austim, country_of_res, used_app_before, age_desc, relation]])

        # Load the trained model and make prediction
        model = joblib.load('asd_random_forest_model.pkl')
        prediction = model.predict(user_input)

        if prediction == 1:
            result = "Positive (Autism Spectrum Disorder)"
        else:
            result = "Negative (No Autism Spectrum Disorder)"

        return redirect(url_for('result', prediction=result))

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
