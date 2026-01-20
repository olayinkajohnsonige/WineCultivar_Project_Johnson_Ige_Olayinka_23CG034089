from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'model', 'wine_cultivar_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'scaler.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect 6 inputs
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    
    # Scale and Predict
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)[0]
    
    # Map 0,1,2 to Cultivar names
    cultivars = {0: "Cultivar 1 (High Quality)", 1: "Cultivar 2 (Standard)", 2: "Cultivar 3 (Mass Market)"}
    result = cultivars.get(prediction, "Unknown")
    
    return render_template('index.html', prediction_text=f'Predicted Origin: {result}')

if __name__ == "__main__":
    app.run(debug=True)