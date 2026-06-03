FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/artifacts/failure_model_events/model.joblib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/failure_model_events/ ./artifacts/failure_model_events/

EXPOSE 8000

CMD ["uvicorn", "src.serve_failure_model:app", "--host", "0.0.0.0", "--port", "8000"]
