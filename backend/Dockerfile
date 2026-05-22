FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies separately for better Docker layer caching
COPY backend/requirements.txt ./requirements-backend.txt
ENV RPY2_CFFI_MODE=ABI
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy the application source code
COPY config.py /app/config.py
COPY backend/ /app/backend/
COPY causal/ /app/causal/
COPY data/ /app/data/
COPY explainability/ /app/explainability/

# Bug Fix: PYTHONPATH must be /app so `from backend.X import Y` resolves
ENV PYTHONPATH=/app

# Bug Fix: create the data/raw directory so ingestion.py can write there
RUN mkdir -p /app/data/raw

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
