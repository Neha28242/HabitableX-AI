from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("habitable_model.pkl")


# Load dataset
df = pd.read_csv(r'C:\Users\Hp\OneDrive\Documents\pyhtonproject\PSCompPars_2026.03.18_01.08.08.csv', comment='#')


# Keep only needed columns
df = df[['pl_name','pl_eqt','pl_rade','pl_bmasse','pl_orbsmax']].dropna()

FEATURES = ['pl_eqt','pl_rade','pl_bmasse','pl_orbsmax']

# Earth reference
EARTH = {
    "temp": 255,
    "radius": 1,
    "mass": 1,
    "distance": 1
}

# ==============================
# HOME
# ==============================
@app.route('/')
def home():
    return render_template('index.html')


# ==============================
# PREDICT + CLOSEST PLANET
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temp'])
    radius = float(request.form['radius'])
    mass = float(request.form['mass'])
    distance = float(request.form['distance'])

    input_data = np.array([[temp, radius, mass, distance]])

    # Probability
    prob = model.predict_proba(input_data)[0][1]
    percentage = round(prob * 100, 2)

    # Find closest planet
    df['distance_score'] = (
        abs(df['pl_eqt'] - temp) +
        abs(df['pl_rade'] - radius) +
        abs(df['pl_bmasse'] - mass)
    )

    closest = df.sort_values('distance_score').iloc[0]

    return render_template(
        'index.html',
        prediction=percentage,
        closest_name=closest['pl_name']
    )


# ==============================
# PLANET SEARCH
# ==============================
@app.route('/planet', methods=['POST'])
def planet():
    name = request.form['planet_name']

    planet = df[df['pl_name'].str.lower() == name.lower()]

    if planet.empty:
        return render_template('index.html', error="Planet not found")

    planet = planet.iloc[0]

    # Comparison logic
    def compare(val, earth_val):
        if val > earth_val:
            return "Higher than Earth"
        elif val < earth_val:
            return "Lower than Earth"
        else:
            return "Similar to Earth"

    return render_template(
        'index.html',
        planet_name=planet['pl_name'],
        temp=planet['pl_eqt'],
        radius=planet['pl_rade'],
        mass=planet['pl_bmasse'],
        distance=planet['pl_orbsmax'],
        temp_comp=compare(planet['pl_eqt'], EARTH['temp']),
        radius_comp=compare(planet['pl_rade'], EARTH['radius']),
        mass_comp=compare(planet['pl_bmasse'], EARTH['mass'])
    )


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)