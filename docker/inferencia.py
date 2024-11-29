import pandas as pd
from tensorflow.keras.models import load_model

MODEL_PATH = "modelo.h5"
model = load_model(MODEL_PATH)


def predict_from_csv(input_csv: str, output_csv: str):
    """
    Carga datos desde un archivo CSV, realiza predicciones y guarda los resultados en otro archivo CSV.

    Args: - input_csv (str): Ruta al archivo CSV con los datos de entrada. - output_csv (str): Ruta donde se guardarán
    los resultados de predicción.
    """
    input_data = pd.read_csv(input_csv)

    predictions = model.predict(input_data)

    results = pd.DataFrame(predictions, columns=["Prediction"])

    results.to_csv(output_csv, index=False)
    print(f"Predicciones guardadas en {output_csv}")
