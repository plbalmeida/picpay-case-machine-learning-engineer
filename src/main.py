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
            }
        }


# endpoint para carregar o modelo
@app.post(
    "/model/load/",
    summary="Carregar Modelo",
    description="Carrega o modelo treinado a partir de um arquivo .joblib.",
    responses={
        200: {
            "description": "Modelo carregado com sucesso.",
            "content": {
                "application/json": {
                    "example": {"status": "Modelo carregado com sucesso"}
                }
            },
        },
        400: {
            "description": "Erro ao carregar o modelo.",
            "content": {
                "application/json": {
                    "example": {"error": "Erro ao carregar o modelo. Verifique o arquivo."}
                }
            },
        },
        422: {
            "description": "Formato do arquivo inválido.",
            "content": {
                "application/json": {
                    "example": {"detail": "Formato do arquivo inválido."}
                }
            },
        },
        500: {
            "description": "Erro interno do servidor.",
            "content": {
                "application/json": {
                    "example": {"error": "Erro interno do servidor."}
                }
            },
        },
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
        raise HTTPException(status_code=400, detail=str(e))


# endpoint para realizar predição
@app.post(
    "/model/predict/",
    summary="Realizar Predição",
    description="Realiza uma predição de atraso de voo com base nos dados fornecidos.",
    responses={
        200: {
            "description": "Predição realizada com sucesso.",
            "content": {
                "application/json": {
                    "example": {"prediction": 23.5}
                }
            },
        },
        400: {
            "description": "Modelo não carregado.",
            "content": {
                "application/json": {
                    "example": {"error": "Modelo não carregado"}
                }
            },
        },
        422: {
            "description": "Dados de entrada inválidos.",
            "content": {
                "application/json": {
                    "example": {"detail": "Dados de entrada inválidos. Verifique os valores fornecidos."}
                }
            },
        },
        500: {
            "description": "Erro interno do servidor.",
            "content": {
                "application/json": {
                    "example": {"error": "Erro interno do servidor."}
                }
            },
        },
    },
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
        Predição do atraso em minutos.
    """
    global model, history
    if model is None:
        raise HTTPException(status_code=400, detail="Modelo não carregado")

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
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erro interno do servidor: " + str(e))


# endpoint para visualizar o histórico
@app.get(
    "/model/history/",
    summary="Visualizar Histórico",
    description="Retorna o histórico de previsões realizadas pela API.",
    responses={
        200: {
            "description": "Histórico retornado com sucesso.",
            "content": {
                "application/json": {
                    "example": {
                        "history": [
                            {"input": {
                                "month": 7,
                                "day": 23,
                                "hour": 16,
                                "sched_dep_time": 1630,
                                "sched_arr_time": 1930,
                                "origin": "JFK",
                                "dest": "LAX",
                                "carrier": "DL",
                                "distance": 3983.0
                            }, "prediction": 23.5}
                        ]
                    }
                }
            },
        },
        404: {
            "description": "Histórico não encontrado.",
            "content": {
                "application/json": {
                    "example": {"detail": "Histórico não encontrado"}
                }
            },
        },
    },
)
async def get_history():
    """
    Retorna o histórico de todas as previsões realizadas, incluindo os dados de entrada e as saídas preditas.

    Returns
    -------
    dict
        Histórico de previsões.
    """
    if not history:
        raise HTTPException(status_code=404, detail="Histórico não encontrado")
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
