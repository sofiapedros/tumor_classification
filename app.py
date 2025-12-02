import torch
import torch.nn as nn
import joblib
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import numpy as np

device = torch.device("cpu")
PREDICTION_COUNTER = Counter(
    'tumor_prediction_count', 
    'contador de predicciones por tipo de tumor',
    ['tumor'])

try:
    torch_load_orig = torch.load

    def cpu_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = 'cpu'
        return torch_load_orig(*args, **kwargs)

    torch.load = cpu_torch_load

    # Ahora podemos cargar el modelo con joblib
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("Error: 'model.pkl' no encontrado. Por favor, ejecuta primero el script de entrenamiento.")
    model = None

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado. Entrena el modelo primero."}), 500
    try:
        # Obtener los datos de la petición JSON
        data = request.get_json(force=True)
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)

        # Convertir a tensor y enviar al dispositivo
        features_tensor = torch.from_numpy(features).to(device)

        # Realizar la predicción
        with torch.no_grad():
            outputs = model(features_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        tumor_map = {0: "bening", 1:'malignant', 2: 'normal'}

        predicted_tumor = tumor_map.get(predicted_class, 'unknown')
        PREDICTION_COUNTER.labels(tumor=predicted_tumor).inc()
        # Devolver la predicción
        return jsonify({"prediction": int(predicted_class), 'tumor': predicted_tumor})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
