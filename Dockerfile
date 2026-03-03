# Dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY model93.h5 .
COPY templates/ ./templates
COPY main_flask.py .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000:8000

CMD ["python", "main_flask.py"]




