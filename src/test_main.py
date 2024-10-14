import os
import tempfile
from fastapi.testclient import TestClient
from main import app
from fastapi import status

client = TestClient(app)


def test_health():
    """
    Testa o endpoint de verificação de saúde da API.
    Verifica se a resposta é 200 e se a mensagem retornada indica que a API está funcionando corretamente.
    """
    response = client.get("/health/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "API está funcionando corretamente"}


def test_predict_without_model_loaded():
    """
    Testa o endpoint de predição sem que um modelo tenha sido carregado.
    Verifica se a resposta é 200 indicando que o modelo não foi carregado.
    """
    data = {
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
    response = client.post("/model/predict/", json=data)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "Modelo não carregado"}


def test_get_history_empty():
    """
    Testa o endpoint de histórico de previsões quando o histórico está vazio.
    Verifica se a resposta é 200 indicando que não há histórico disponível.
    """
    response = client.get("/model/history/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "Histórico não encontrado"}


def test_load_model():
    """
    Testa o endpoint de carregamento do modelo.
    Envia um arquivo válido .joblib e verifica se o modelo é carregado com sucesso.
    """
    model_path = os.path.join(os.getcwd(), "model", "flight_delay_model.joblib")
    assert os.path.exists(model_path), f"O modelo não foi encontrado em {model_path}"
    with open(model_path, "rb") as file:
        response = client.post("/model/load/", files={"file": ("flight_delay_model.joblib", file, "application/octet-stream")})
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "Modelo carregado com sucesso"}


def test_load_model_invalid_file():
    """
    Testa o endpoint de carregamento do modelo com um arquivo inválido.
    Cria um arquivo temporário que não é um modelo .joblib e verifica se a resposta é um erro apropriado.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv") as file:
        file.write(b"Este nao e um modelo valido.")  # conteúdo inválido para simular erro
        file.seek(0)
        response = client.post("/model/load/", files={"file": (file.name, file, "text/csv")})
    assert response.status_code == status.HTTP_200_OK
    assert "status" in response.json()
    assert "Erro" in response.json()["status"]  # espera que a resposta contenha uma mensagem de erro genérica


def test_predict_with_model_loaded():
    """
    Testa o endpoint de predição após o carregamento do modelo.
    Carrega o modelo e realiza uma predição, verificando se a resposta é 200 e contém a previsão.
    """
    model_path = os.path.join(os.getcwd(), "model", "flight_delay_model.joblib")
    assert os.path.exists(model_path), f"O modelo não foi encontrado em {model_path}"
    with open(model_path, "rb") as file:
        client.post("/model/load/", files={"file": ("flight_delay_model.joblib", file, "application/octet-stream")})

    data = {
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
    response = client.post("/model/predict/", json=data)
    assert response.status_code == status.HTTP_200_OK
    assert "prediction" in response.json()


def test_get_history():
    """
    Testa o endpoint de histórico de previsões após realizar uma predição.
    Carrega o modelo, realiza uma predição e verifica se o histórico contém a predição realizada.
    """
    model_path = os.path.join(os.getcwd(), "model", "flight_delay_model.joblib")
    assert os.path.exists(model_path), f"O modelo não foi encontrado em {model_path}"
    with open(model_path, "rb") as file:
        client.post("/model/load/", files={"file": ("flight_delay_model.joblib", file, "application/octet-stream")})

    data = {
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
    client.post("/model/predict/", json=data)

    response = client.get("/model/history/")
    assert response.status_code == status.HTTP_200_OK
    assert "history" in response.json()
    assert len(response.json()["history"]) > 0  # verifica se o histórico não está vazio
