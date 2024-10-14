from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict, Any

app = FastAPI(
    title="API de Predição de Atraso em Voos",
    description="Uma API para prever atrasos de voos com base em informações fornecidas.",
    version="1.0.0",
    contact={
        "name": "Equipe de Desenvolvimento",
        "email": "dev@example.com",
    },
)

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
    dep_delay: float

    class Config:
        json_schema_extra = {
            "example": {
                "month": 7,
                "day": 23,
                "hour": 16,
                "sched_dep_time": 1630,
                "sched_arr_time": 1930,
                "origin": "JFK",
                "dest": "LAX",
                "carrier": "DL",
                "distance": 3983.0,
                "dep_delay": 15.0
            }
        }


# endpoint para carregar o modelo
@app.post(
    "/model/load/",
    summary="Carregar Modelo",
    description="Carrega o modelo treinado a partir de um arquivo .joblib.",
    responses={
        200: {
            "description": "Resposta para todas as situações.",
            "content": {
                "application/json": {
                    "example": {"status": "Modelo carregado com sucesso"}
                }
            },
        }
    },
    status_code=status.HTTP_200_OK,
)
async def load_model(file: UploadFile = File(...)):
    """
    Carrega um modelo de aprendizado de máquina treinado a partir de um arquivo fornecido pelo usuário.

    Parameters
    ----------
    file : UploadFile
        Arquivo .joblib contendo o modelo treinado.

    Returns
    -------
    dict
        Mensagem de sucesso ou erro.
    """
    global model
    try:
        model = joblib.load(file.file)
        return {"status": "Modelo carregado com sucesso"}
    except Exception as e:
         return {"status": f"Erro ao carregar o modelo. Detalhes: {str(e)}"}


# endpoint para realizar predição
@app.post(
    "/model/predict/",
    summary="Realizar Predição",
    description="Realiza uma predição de atraso de voo com base nos dados fornecidos.",
    responses={
        200: {
            "description": "Resposta para todas as situações.",
            "content": {
                "application/json": {
                    "example": {"prediction": -29.344152735629503}
                }
            },
        }
    },
    status_code=status.HTTP_200_OK,
)
async def predict(data: FlightData):
    """
    Realiza uma predição de atraso de voo com base nas informações fornecidas pelo usuário.

    Parameters
    ----------
    data : FlightData
        Dados do voo contendo informações como mês, dia, horário, aeroporto de origem, destino, companhia aérea, etc.

    Returns
    -------
    dict
        Predição do atraso em minutos ou mensagem de erro.
    """
    global model, history
    if model is None:
        return {"status": "Modelo não carregado"}

    try:
        # prepara os dados de entrada
        input_data = pd.DataFrame([data.dict()])
        input_data = pd.get_dummies(input_data, columns=["carrier", "origin", "dest"], drop_first=True)

        # certifica que as colunas do modelo estão presentes
        missing_cols = set(model.feature_names_in_) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[model.feature_names_in_]

        # predição
        prediction = model.predict(input_data)[0]

        # salva no histórico
        history_entry = {"input": data.dict(), "prediction": prediction}
        history.append(history_entry)

        return {"prediction": prediction}
    except ValueError as e:
        return e


# endpoint para visualizar o histórico
@app.get(
    "/model/history/",
    summary="Visualizar Histórico",
    description="Retorna o histórico de previsões realizadas pela API.",
    responses={
        200: {
            "description": "Resposta para todas as situações.",
            "content": {
                "application/json": {
                    "example": {
                        "history": [
                            {
                                "input": {
                                    "month": 7,
                                    "day": 23,
                                    "hour": 16,
                                    "sched_dep_time": 1630,
                                    "sched_arr_time": 1930,
                                    "origin": "JFK",
                                    "dest": "LAX",
                                    "carrier": "DL",
                                    "distance": 3983.0,
                                    "dep_delay": 15.0
                                }, 
                                "prediction": -29.344152735629503
                            }
                        ]
                    },
                    "example_no_history": {
                        "status": "Histórico não encontrado"
                    }
                }
            },
        }
    },
    status_code=status.HTTP_200_OK,
)
async def get_history():
    """
    Retorna o histórico de todas as previsões realizadas, incluindo os dados de entrada e as saídas preditas.

    Returns
    -------
    dict
        Histórico de previsões ou mensagem de erro.
    """
    if not history:
        return {"status": "Histórico não encontrado"}
    return {"history": history}


# endpoint de saúde
@app.get(
    "/health/",
    summary="Verificar Saúde da API",
    description="Verifica se a API está funcionando corretamente.",
    responses={
        200: {
            "description": "API está funcionando corretamente.",
            "content": {
                "application/json": {
                    "example": {"status": "API está funcionando corretamente"}
                }
            },
        },
    },
)
async def health():
    """
    Verifica o status da API para garantir que está operacional.

    Returns
    -------
    dict
        Mensagem indicando o status da API.
    """
    return {"status": "API está funcionando corretamente"}