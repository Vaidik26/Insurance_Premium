from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load preprocessor and model
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
model = pickle.load(open("artifacts/model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = {
            "age": int(request.form["age"]),
            "sex": request.form["sex"],
            "bmi": float(request.form["bmi"]),
            "children": int(request.form["children"]),
            "smoker": request.form["smoker"],
            "region": request.form["region"],
        }

        input_df = pd.DataFrame([data])
        transformed_input = preprocessor.transform(input_df)
        prediction = model.predict(transformed_input)[0]
        predicted_price = round(prediction, 2)

        return render_template("predict.html", prediction=predicted_price)

    return render_template("predict.html", prediction=None)


if __name__ == "__main__":
    app.run(debug=True)
