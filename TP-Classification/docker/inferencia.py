import joblib
import pandas as pd
import tensorflow as tf
import numpy as np

# Cargar el pipeline previamente guardado
pipeline = joblib.load("preprocess.pkl")

# Cargar el modelo de deep learning
model = tf.keras.models.load_model("modelo_nn.h5")

# Cargar nuevos datos para inferencia
X_new = pd.read_csv("weatherAUS.csv")

# Seleccionar 10 datos al azar para la inferencia
X_new = X_new.sample(n=10, random_state=42)
true_rain_value = X_new["RainTomorrow"]
X_new = X_new.drop(["Location", "RainTomorrow"], axis=1)

# Convertir Date de una columna que contiene los dias a contener solamente los meses
X_new["Date"] = pd.to_datetime(X_new["Date"])
X_new["Date"] = X_new["Date"].dt.month

# Codificacion de variables categoricas
cardinal_to_angle = {
    "N": 0,
    "NNE": 22.5,
    "NE": 45,
    "ENE": 67.5,
    "E": 90,
    "ESE": 112.5,
    "SE": 135,
    "SSE": 157.5,
    "S": 180,
    "SSW": 202.5,
    "SW": 225,
    "WSW": 247.5,
    "W": 270,
    "WNW": 292.5,
    "NW": 315,
    "NNW": 337.5,
}

# Reemplazar los puntos cardinales por sus respectivos angulos
X_new["WindGustDir"] = X_new["WindGustDir"].replace(cardinal_to_angle)
X_new["WindDir9am"] = X_new["WindDir9am"].replace(cardinal_to_angle)
X_new["WindDir3pm"] = X_new["WindDir3pm"].replace(cardinal_to_angle)

# Calcular las coordenadas seno y coseno para la variable 'WindGustDir'.
X_new["WindGustDir_Sen"] = np.sin(X_new["WindGustDir"])
X_new["WindGustDir_Cos"] = np.cos(X_new["WindGustDir"])

# Calcular las coordenadas seno y coseno para la variable 'WindDir9am'.
X_new["WindDir9am_Sen"] = np.sin(X_new["WindDir9am"])
X_new["WindDir9am_Cos"] = np.cos(X_new["WindDir9am"])

# Calcular las coordenadas seno y coseno para la variable 'WindDir3pm'.
X_new["WindDir3pm_Sen"] = np.sin(X_new["WindDir3pm"])
X_new["WindDir3pm_Cos"] = np.cos(X_new["WindDir3pm"])

# Eliminar las variables originales 'WindGustDir', 'WindDir9am' y 'WindDir3pm', ya que no serán necesarias para el
# entrenamiento del modelo.
X_new = X_new.drop(columns=["WindGustDir"])
X_new = X_new.drop(columns=["WindDir9am"])
X_new = X_new.drop(columns=["WindDir3pm"])

# Convertir las variables RainToday y RainTomorrow en variables de valor 0 para No llovio y 1 para Llovio.
X_new[["RainToday"]] = X_new[["RainToday"]].replace({"No": 0, "Yes": 1})
print(X_new.columns)

# Aplicar el pipeline al nuevo conjunto de datos
X_new_scaled = pipeline.transform(X_new)

# Convertir a DataFrame si se necesita mantener nombres de columnas
X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=X_new.columns)

# Realizar inferencia con el modelo de deep learning
y_pred = model.predict(X_new_scaled)

# Convertir las predicciones en etiquetas de lluvia
rain_predictions = ["YES" if pred >= 0.5 else "NO" for pred in y_pred.flatten()]

# Guardar las predicciones en un archivo CSV
pd.DataFrame(y_pred, columns=["Predicciones"]).to_csv("predicciones.csv", index=False)

# Mostrar los primeros registros para ver el resultado
print(X_new_scaled_df.head())
print(pd.DataFrame(y_pred, columns=["Predicciones"]).head())

# Guardar las predicciones en un archivo CSV
predictions_df = pd.DataFrame(
    {"Predicciones": y_pred.flatten(), "Lluvia": rain_predictions}
)
predictions_df.to_csv("resultados/predicciones.csv", index=False)

# Mostrar los primeros registros para ver el resultado
print(true_rain_value)
print(X_new_scaled_df.head())
print(predictions_df.head())
