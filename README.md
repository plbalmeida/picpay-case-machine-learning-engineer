# Desafio de MLE da PicPay

O presente repositório tem a solução do desafio para a posição de MLE na PicPay.

A solução consiste em uma API para previsão de atraso em voos baseado em um modelo de ML treinado.

## Stack utilizada

- Python (PySpark, sci-kit learn, FastAPI)
- git
- Docker
- Kubernetes

## Estrutura do repo

```bash
.
├── README.md
├── data
│   ├── processed
│   └── raw
│       └── airports-database.zip
├── docker
│   ├── Dockerfile
│   └── requirements.txt
├── docs
│   ├── diagram.drawio
│   └── diagram.png
├── kubernetes
│   ├── deployment.yaml
│   └── service.yaml
├── model
│   └── flight_delay_model.joblib
├── notebook
│   ├── 01-etl.ipynb
│   ├── 02-eda.ipynb
│   └── 03-ml.ipynb
├── requirements.txt
└── src
    ├── main.py
    └── test_main.py
```

## Execução da aplicação

Clonar o repo localmente:

```bash
git clone https://github.com/plbalmeida/picpay-case-machine-learning-engineer.git
```

Criar o ambiente virtual:

```bash
python3 -m venv venv
```

Ativar o ambiente:

```bash
source venv/bin/activate
```

Instalar as libs necessárias:

```bash
pip install -r requirements.txt  
```

Para executar os testes unitários, executar:

```bash
pytest src/test_main.py
```

Build da imagem Docker:

```bash
docker build -t <SEU USUÁRIO DO DOCKER HUB>/flight_delay_api:latest -f docker/Dockerfile .
```

Envio da imagem para o Docker Hub:

```bash
docker push <SEU USUÁRIO DO DOCKER HUB>/flight_delay_api:latest
```

Execução do container no Docker, disponibilizando acesso localmente em `http://localhost:8000`:

```bash
docker run -p 8000:8000 <SEU USUÁRIO DO DOCKER HUB>/flight_delay_api
```

Para testar as rotas da API que está rodando no Docker, primeiro checamos qual container está rodando:

```bash
docker ps
```

Agora é possível testar as rotas da API usando `curl` diretamente do terminal. Aqui estão alguns exemplos de testes:

   - **Verificar Saúde da API**:

     ```bash
     curl -X GET http://localhost:8000/health/
     ```

     Você deve receber uma resposta semelhante a:

     ```json
     {"status": "API está funcionando corretamente"}
     ```

   - **Carregar um Modelo**:
     Você pode fazer o upload de um modelo `.joblib` usando a seguinte solicitação:

     ```bash
     curl -X POST -F "file=@model/flight_delay_model.joblib" http://localhost:8000/model/load/
     ```

     Isso deve retornar:

     ```json
     {"status": "Modelo carregado com sucesso"}
     ```

   - **Realizar Predição**:

     ```bash
     curl -X POST http://localhost:8000/model/predict/ \
     -H "Content-Type: application/json" \
     -d '{
         "month": 7,
         "day": 23,
         "hour": 16,
         "sched_dep_time": 1630,
         "sched_arr_time": 1930,
         "origin": "JFK",
         "dest": "LAX",
         "carrier": "DL",
         "distance": 3983.0,
         "dep_delay": 10.0
     }'
     ```

     Isso deve retornar a predição para o atraso:

     ```json
     {"prediction": -29.344152735629503}
     ```

   - **Obter Histórico de Predições**:

     ```bash
     curl -X GET http://localhost:8000/model/history/
     ```

     Isso deve retornar o histórico de predições realizadas:

     ```json
     {
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
             "dep_delay": 10.0
           },
           "prediction": -29.344152735629503
         }
       ]
     }
     ```

Com a aplicação em execução, você pode acessar a documentação **Swagger** da API em:

```bash
http://localhost:8000/docs
```

Ou a documentação **ReDoc** em:

```bash
http://localhost:8000/redoc
```

## Execução da aplicação com Kubernetes

Antes de executar a aplicação no Kubernetes, precisamos ter a imagem Docker (não é necessário executar o build e push se realizou os passos anteriores de **Execução da aplicação com Docker**).

Build da imagem do Docker:

```bash
docker build -t <SEU USUÁRIO DO DOCKER HUB>/flight_delay_api:latest -f docker/Dockerfile .
```

Envio da Imagem para o Docker Hub:

```bash
docker push <SEU USUÁRIO DO DOCKER HUB>/flight_delay_api:latest
```

A aplicação é gerenciada pelo Kubernetes pelos seguintes manifestos:

- `kubernetes/deployment.yaml` define como a aplicação será implantada (replicas, pods, containers).
- `kubernetes/service.yaml` cria um serviço para acessar a aplicação exposta pelo Kubernetes.

No manifesto `kubernetes/deployment.yaml` deve ser incluido o usuário do Docker Hub:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flight-delay-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: flight-delay-api
  template:
    metadata:
      labels:
        app: flight-delay-api
    spec:
      containers:
        - name: flight-delay-api-container
          image: <SEU USUÁRIO DO DOCKER HUB>/flight-delay-api:latest
          ports:
            - containerPort: 8000
```

Navegar até a pasta do projeto e aplique os manifestos YAML:

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

É interessante verificar se os pods e serviços estão em execução:

```bash
kubectl get pods
kubectl get services
```

É esperado termos três pods do deployment `flight-delay-api` em execução e um serviço chamado `flight-delay-api-service` com a porta `80:30007/TCP` exposta.

Para acessar a API, você pode usar a porta exposta `30007` em qualquer nó do seu cluster Kubernetes (assumindo que o serviço foi configurado como `NodePort`).

Agora é possível testar as rotas da API usando `curl` diretamente do terminal. Aqui estão alguns exemplos de testes:

   - **Verificar Saúde da API**:

     ```bash
     curl -X GET http://localhost:30007/health/
     ```

     Você deve receber uma resposta semelhante a:

     ```json
     {"status": "API está funcionando corretamente"}
     ```

   - **Carregar um Modelo**:
     Você pode fazer o upload de um modelo `.joblib` usando a seguinte solicitação:

     ```bash
     curl -X POST -F "file=@model/flight_delay_model.joblib" http://localhost:30007/model/load/
     ```

     Isso deve retornar:

     ```json
     {"status": "Modelo carregado com sucesso"}
     ```

   - **Realizar Predição**:

     ```bash
     curl -X POST http://localhost:30007/model/predict/ \
     -H "Content-Type: application/json" \
     -d '{
         "month": 7,
         "day": 23,
         "hour": 16,
         "sched_dep_time": 1630,
         "sched_arr_time": 1930,
         "origin": "JFK",
         "dest": "LAX",
         "carrier": "DL",
         "distance": 3983.0,
         "dep_delay": 10.0
     }'
     ```

     Isso deve retornar a predição para o atraso:

     ```json
     {"prediction": -29.344152735629503}
     ```

   - **Obter Histórico de Predições**:

     ```bash
     curl -X GET http://localhost:30007/model/history/
     ```

     Isso deve retornar o histórico de predições realizadas:

     ```json
     {
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
             "dep_delay": 10.0
           },
           "prediction": -29.344152735629503
         }
       ]
     }
     ```

## Desligando/deletanado o Kubernetes e Docker

Execute o comando abaixo para deletar os recursos criados no Kubernetes:

```bash
kubectl delete -f kubernetes/deployment.yaml
kubectl delete -f kubernetes/service.yaml
```

Parar o container Docker em execução, primeiro liste os containers em execução:

```bash
docker ps
```

Depois, encontre o `CONTAINER ID` do container que deseja parar e execute:

```bash
docker stop <CONTAINER_ID>
```

Depois de parar o container, você pode removê-lo com:

```bash
docker rm <CONTAINER_ID>
```

Para remover a imagem criada, primeiro liste as imagens disponíveis:

```bash
docker images
```

Em seguida, remova a imagem usando o `IMAGE ID`:

```bash
docker rmi <IMAGE_ID>
```