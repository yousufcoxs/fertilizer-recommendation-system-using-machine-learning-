from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

 
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

 
available_soils = [col.replace('Soil_', '') for col in model_columns if col.startswith('Soil_')]
available_crops = [col.replace('Crop_', '') for col in model_columns if col.startswith('Crop_')]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get form data
            temp = float(request.form["temp"])
            moist = float(request.form["moist"])
            rain = float(request.form["rain"])
            ph = float(request.form["ph"])
            n = float(request.form["n"])
            p = float(request.form["p"])
            k = float(request.form["k"])
            c = float(request.form["c"])
            soil = request.form["soil"]
            crop = request.form["crop"]

            # Prepare input data
            data = {col: 0 for col in model_columns}
            data.update({
                "Temperature": temp,
                "Moisture": moist,
                "Rainfall": rain,
                "PH": ph,
                "Nitrogen": n,
                "Phosphorous": p,
                "Potassium": k,
                "Carbon": c,
                f"Soil_{soil}": 1,
                f"Crop_{crop}": 1,
            })

            df = pd.DataFrame([data])
            scaled_input = scaler.transform(df)
            pred = model.predict(scaled_input)
            prediction = le.inverse_transform(pred)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", soils=available_soils, crops=available_crops, prediction=prediction)


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

@app.route("/fermar_portal")
def fermar_portal():
    return render_template("fermar_portal.html")

if __name__ == "__main__":
    app.run(debug=True)
