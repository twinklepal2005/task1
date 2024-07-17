import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import importlib.metadata

# Sample data to write to CSV
data = [
    {'age': 25, 'gender': 'male', 'swimming_ability': 'beginner', 'socioeconomic_status': 'low', 'fitness_level': 'moderate', 'is_safe': 1},
    {'age': 30, 'gender': 'female', 'swimming_ability': 'advanced', 'socioeconomic_status': 'high', 'fitness_level': 'high', 'is_safe': 0},
    {'age': 20, 'gender': 'female', 'swimming_ability': 'none', 'socioeconomic_status': 'medium', 'fitness_level': 'low', 'is_safe': 1},
    {'age': 35, 'gender': 'male', 'swimming_ability': 'intermediate', 'socioeconomic_status': 'high', 'fitness_level': 'high', 'is_safe': 0},
    {'age': 28, 'gender': 'female', 'swimming_ability': 'beginner', 'socioeconomic_status': 'low', 'fitness_level': 'low', 'is_safe': 1},
    {'age': 32, 'gender': 'male', 'swimming_ability': 'advanced', 'socioeconomic_status': 'medium', 'fitness_level': 'high', 'is_safe': 0}
]

# Filepath to save the CSV file
filepath = 'safety_data.csv'

# Function to create and populate CSV file
def create_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['age', 'gender', 'swimming_ability', 'socioeconomic_status', 'fitness_level', 'is_safe']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Create and populate the CSV file
create_csv(filepath, data)

print(f'CSV file "{filepath}" created successfully.')

# Step 1: Data Preprocessing
def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Convert categorical variables to numerical
    data['gender'] = data['gender'].map({'male': 0, 'female': 1})
    data['swimming_ability'] = data['swimming_ability'].map({'none': 0, 'beginner': 1, 'intermediate': 2, 'advanced': 3})
    data['socioeconomic_status'] = data['socioeconomic_status'].map({'low': 0, 'medium': 1, 'high': 2})
    data['fitness_level'] = data['fitness_level'].map({'low': 0, 'moderate': 1, 'high': 2})

    # Define features and target
    X = data[['age', 'gender', 'swimming_ability', 'socioeconomic_status', 'fitness_level']]
    y = data['is_safe']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Step 2: Model Training
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 3: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('safety_data.csv')

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

# Step 4: Create Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract and preprocess input data
    age = data['age']
    gender = 0 if data['gender'] == 'male' else 1
    swimming_ability = {'none': 0, 'beginner': 1, 'intermediate': 2, 'advanced': 3}[data['swimming_ability']]
    socioeconomic_status = {'low': 0, 'medium': 1, 'high': 2}[data['socioeconomic_status']]
    fitness_level = {'low': 0, 'moderate': 1, 'high': 2}[data['fitness_level']]

    input_data = np.array([[age, gender, swimming_ability, socioeconomic_status, fitness_level]])
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    result = 'safe' if prediction[0] == 1 else 'not safe'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    # Retrieve Flask version using importlib.metadata
    flask_version = importlib.metadata.version("flask")
    print(f'Flask version: {flask_version}')

    # Run the Flask application using the development server
    app.run(debug=True, host='0.0.0.0', port=5000)
