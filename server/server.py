import json
import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Dynamic paths to files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "model", "bangalore_home_prices_model.pickle"
)
COLUMNS_PATH = os.path.join(BASE_DIR, "..", "model", "columns.json")

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load data columns from JSON
with open(COLUMNS_PATH, "r") as f:
    data_columns = json.load(f)["data_columns"]
    # First 3 columns are sqft, bath, bhk; remaining columns are locations
    LOCATIONS = data_columns[3:]


@app.route("/")
def home():
    return render_template(
        "index.html",
        locations=LOCATIONS,
        sqft="",
        bhk="",
        bath="",
        selected_location="",
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        sqft = float(request.form["sqft"])
        bhk = int(request.form["bhk"])
        bath = int(request.form["bath"])
        location = request.form["location"].lower()

        # Build feature array initialized with zeros matching exact column count
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        # Locate location column index and set its one-hot encoding value to 1
        if location in data_columns:
            loc_index = data_columns.index(location)
            x[loc_index] = 1

        # Predict price (returns value in Lakh)
        prediction = model.predict([x])[0]
        output = f"{round(float(prediction), 2)} Lakh"

        return render_template(
            "index.html",
            locations=LOCATIONS,
            sqft=sqft,
            bhk=str(bhk),
            bath=str(bath),
            selected_location=location,
            prediction_text=output,
        )

    except Exception as e:
        return render_template(
            "index.html",
            locations=LOCATIONS,
            sqft="",
            bhk="",
            bath="",
            selected_location="",
            prediction_text=f"Error: {str(e)}",
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)