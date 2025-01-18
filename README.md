# Autism Spectrum Disorder (ASD) Prediction using Machine Learning

This project aims to predict the likelihood of an individual being diagnosed with **Autism Spectrum Disorder (ASD)** based on various factors such as medical scores, age, gender, ethnicity, and other relevant data. The model uses **Random Forest** algorithm to analyze and predict the result. The system is implemented using **Flask** for creating a web interface where users can input data and receive predictions.

## Features
- **Flask Web Application**: A simple and interactive web interface for inputting the necessary features and receiving predictions.
- **Random Forest Classifier**: A machine learning model that predicts the presence of ASD based on input data.
- **Prediction Results Page**: Displays the result on a new page after clicking the "Predict" button.
- **Model Persistence**: The trained machine learning model is saved using **Joblib** to ensure the same model is used for predictions across sessions.

## Technologies Used
- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Data Preprocessing**: Pandas, LabelEncoder
- **Web Development**: HTML, CSS
- **Model Persistence**: Joblib (for saving and loading the trained model)

