from flask import Flask, request, jsonify, render_template
import pandas as pd
from keras.models import load_model

app = Flask(__name__)

model = load_model("./modelo.h5")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        uploaded_file = request.files["file"]
        df = pd.read_csv(uploaded_file)


        df = df.fillna(0)


        df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1})
        df["Location"] = df["Location"].astype("category").cat.codes
        df["WindGustDir"] = df["WindGustDir"].astype("category").cat.codes
        df["WindDir9am"] = df["WindDir9am"].astype("category").cat.codes
        df["WindDir3pm"] = df["WindDir3pm"].astype("category").cat.codes

        if "RainTomorrow" in df.columns:
            df = df.drop(columns=["RainTomorrow"])

        if df.shape[1] != model.input_shape[1]:
            return jsonify({
                "error": f"El modelo espera {model.input_shape[1]} columnas, pero el archivo cargado tiene {df.shape[1]} columnas."
            })

        predictions = model.predict(df)

        result = df.copy()
        result["Predictions"] = predictions

        result_csv = "/tmp/predictions.csv"
        result.to_csv(result_csv, index=False)

        return jsonify({"message": "Predicciones generadas correctamente.", "file_path": result_csv})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)