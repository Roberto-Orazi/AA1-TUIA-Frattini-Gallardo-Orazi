FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY app.py .
COPY modelo.h5 ./
COPY templates/ templates/
COPY static/ static/

EXPOSE 5000

CMD ["python", "app.py"]