FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Bug Fix: copy requirements first for layer caching
COPY frontend/requirements.txt ./requirements-frontend.txt
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY frontend/ /app/frontend/
COPY frontend/.streamlit/ /app/.streamlit/
ENV PYTHONPATH=/app

EXPOSE 8501

# Bug Fix: headless=true disables the browser auto-open; enableCORS=false for iframe embedding
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
