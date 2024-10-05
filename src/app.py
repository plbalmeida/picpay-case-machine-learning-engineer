from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict, Any


app = FastAPI()

# variáveis globais
model = None
history = []


class FlightData(BaseModel):
    month: int
    day: int
    hour: int
    sched_dep_time: int
    sched_arr_time: int
    origin: str
    dest: str
    carrier: str
    distance: float


# endpoint para carregar o modelo
@app.post("/model/load/")
async def load_model(file: UploadFile = File(...)):
    global model
    try:
        model = joblib.load(file.file)
        return {"status": "Modelo carregado com sucesso"}
    except Exception as e:
        return {"error": str(e)}


# endpoint para realizar predição
@app.post("/model/predict/")
async def predict(data: FlightData):
    global model, history
    if model is None:
        return {"error": "Modelo não carregado"}

    # preparar os dados de entrada
    input_data = pd.DataFrame([data.dict()])
    input_data = pd.get_dummies(input_data, columns=["carrier", "origin", "dest"], drop_first=True)

    # certificar-se de que as colunas do modelo estão presentes
    missing_cols = set(model.feature_names_in_) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[model.feature_names_in_]

    # realiza a predição
    prediction = model.predict(input_data)[0]

    # salva no histórico
    history_entry = {"input": data.dict(), "prediction": prediction}
    history.append(history_entry)

    return {"prediction": prediction}


# endpoint para visualizar o histórico
@app.get("/model/history/")
async def get_history():
    return {"history": history}


# endpoint de saúde
@app.get("/health/")
async def health():
    return {"status": "API está funcionando corretamente"}
