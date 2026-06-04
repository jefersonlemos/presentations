FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/model.joblib

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.serve_failure_model:app", "--host", "0.0.0.0", "--port", "8000"]
