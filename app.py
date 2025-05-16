from flask import Flask, render_template, request
import joblib
import pandas as pd
import random

app = Flask(__name__)

# Load trained model
model = joblib.load("diagnosis_model.pkl")

# Full list of symptoms (must match dataset feature columns)
symptom_columns = [
    "fever", "cough", "runny_nose", "headache", "fatigue", "body_aches",
    "dizziness", "nausea", "stomach_pain", "shortness_of_breath", "skin_rash",
    "sore_throat", "sweating", "chills", "swelling", "loss_of_appetite",
    "blurred_vision", "frequent_urination", "joint_pain", "congestion"
]

def simulate_iot_data():
    heart_rate = random.randint(60, 110)
    spo2 = random.randint(90, 100)
    temperature = round(random.uniform(36.0, 39.0), 1)
    return heart_rate, spo2, temperature

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    vitals = {}
    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")

        # Convert selected symptoms to binary input
        features = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]
        df_input = pd.DataFrame([features], columns=symptom_columns)
        prediction = model.predict(df_input)[0]

        # Simulate vitals
        heart_rate, spo2, temperature = simulate_iot_data()
        vitals = {
            "Heart Rate": f"{heart_rate} bpm",
            "SpO2": f"{spo2}%",
            "Temperature": f"{temperature} °C"
        }

        result = {
            "diagnosis": prediction,
            "note": "This is an AI-generated prediction. Please consult a doctor for confirmation."
        }

    return render_template("index.html", result=result, vitals=vitals, symptom_columns=symptom_columns)

if __name__ == "__main__":
    app.run(debug=True)
