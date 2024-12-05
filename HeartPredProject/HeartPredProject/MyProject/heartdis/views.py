import joblib
import numpy as np
from django.shortcuts import render
from sklearn.preprocessing import StandardScaler
import os

# Load the trained model and scaler (use absolute path if necessary)
# model = joblib.load('heart_disease_model.joblib')
# scaler = joblib.load('scaler.joblib')


MODEL_PATH = os.path.join(os.path.dirname(__file__), "heart_disease_model.joblib")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

MODEL_PATH1=os.path.join(os.path.dirname(__file__),"scaler.joblib")
try:
    scaler = joblib.load(MODEL_PATH1)
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None








def index(request):
    if request.method == 'POST':
        # Extract input data from the form
        try:
            # Ensure the form fields are correctly named and cast them to the appropriate types
            age = int(request.POST.get('age', 0))
            sex = int(request.POST.get('sex', 0))
            chest_pain = int(request.POST.get('chest_pain', 0))
            blood_pressure = int(request.POST.get('blood_pressure', 0))
            cholesterol = int(request.POST.get('cholesterol', 0))
            fasting_blood_sugar = int(request.POST.get('fasting_blood_sugar', 0))
            resting_ecg = int(request.POST.get('resting_ecg', 0))
            max_heart_rate = int(request.POST.get('max_heart_rate', 0))
            exercise_angina = int(request.POST.get('exercise_angina', 0))
            old_peak = float(request.POST.get('old_peak', 0.0))
            slope = int(request.POST.get('slope', 0))
            ca = int(request.POST.get('ca', 0))
            thal = int(request.POST.get('thal', 0))

            # Prepare the new input for prediction
            new_input = np.array([[age, sex, chest_pain, blood_pressure, cholesterol, fasting_blood_sugar,
                                   resting_ecg, max_heart_rate, exercise_angina, old_peak, slope, ca, thal]])

            # Scale the input using the loaded scaler
            new_input_scaled = scaler.transform(new_input)

            # Make prediction using the model
            prediction = model.predict(new_input_scaled)

            # Set prediction result
            prediction_label = "Disease Present" if prediction[0] == 1 else "No Disease"

            # Return result to the user
            return render(request, 'heartdis/index.html', {'prediction': prediction_label})
        except ValueError:
            # Handle the case where conversion fails or invalid input is provided
            return render(request, 'heartdis/index.html', {'error': 'Invalid input values! Please ensure all fields are filled correctly.'})

    # If it's a GET request, just return the form page
    return render(request, 'heartdis/index.html')
