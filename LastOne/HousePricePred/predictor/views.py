import os
import numpy as np
import joblib
from django.shortcuts import render


MODEL_PATH = os.path.join(os.path.dirname(__file__), "XGBoost_Model (1).joblib")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# # Load the trained model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "XGBoost_Model.joblib")
# model = joblib.load(MODEL_PATH)

def home(request):
    if request.method == 'POST':
        try:
            # Get input data from the form
            med_inc = float(request.POST.get('MedInc'))
            house_age = float(request.POST.get('HouseAge'))
            ave_rooms = float(request.POST.get('AveRooms'))
            ave_bedrms = float(request.POST.get('AveBedrms'))
            population = float(request.POST.get('Population'))
            ave_occup = float(request.POST.get('AveOccup'))
            latitude = float(request.POST.get('Latitude'))
            longitude = float(request.POST.get('Longitude'))

            # Prepare the data for the model
            features = np.array([
                med_inc, house_age, ave_rooms, ave_bedrms,
                population, ave_occup, latitude, longitude
            ]).reshape(1, -1)

            # Get the predicted price
            predicted_price = model.predict(features)[0]
            predicted_price = round(predicted_price, 2)

            # Pass the predicted price to the template
            return render(request, 'predictor/home.html', {
                'predicted_price': predicted_price,
                'submitted': True
            })

        except Exception as e:
            # Handle errors
            return render(request, 'predictor/home.html', {
                'error': f"Invalid input: {e}",
                'submitted': True
            })

    return render(request, 'predictor/home.html')
