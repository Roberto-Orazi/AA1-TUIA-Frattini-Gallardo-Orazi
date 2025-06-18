FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY inferencia.py .
COPY modelo_nn.h5 ./
COPY preprocess.pkl .
COPY weatherAUS.csv .

VOLUME ["/app/resultados"]

EXPOSE 5000

CMD ["python", "inferencia.py"]