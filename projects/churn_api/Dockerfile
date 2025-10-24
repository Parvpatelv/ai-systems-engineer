# churn_api/Dockerfile
# Lightweight base with Python 3.11
FROM python:3.11-slim

# ---- Runtime settings ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=models/model.pkl

# Workdir
WORKDIR /app

# ---- System deps (for building some wheels) ----
# Install, then remove after pip install to keep image lean
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (cache-friendly) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optionally remove build tools after wheels are installed
RUN apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ---- App code & model artifacts ----
COPY src/ src/
COPY app/ app/
COPY models/ models/

# Create non-root user (safer in prod) AFTER files are present, then fix ownership
RUN useradd -m appuser && chown -R appuser:appuser /app

# Expose FastAPI port
EXPOSE 8000

# Drop privileges
USER appuser

# Start the API (PORT is flexible; defaults to 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
