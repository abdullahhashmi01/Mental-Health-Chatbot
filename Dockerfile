# Mental Health Support Chatbot - Docker image
FROM python:3.10-slim

WORKDIR /app

# System deps (build tools sometimes needed for tokenizers/sentencepiece wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Model must already be fine-tuned and present at artifacts/model_trainer/...
# (run training separately, or mount it as a volume at container run time)

EXPOSE 8000

CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
