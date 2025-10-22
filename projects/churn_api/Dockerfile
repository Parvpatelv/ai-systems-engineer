# churn_api/Dockerfile
# Lightweight base with Python 3.11
FROM python:3.11-slim

# ---- Runtime settings ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=models/model.pkl

# Create non-root user (safer in prod)
RUN useradd -m appuser

# Workdir
WORKDIR /app

# ---- System deps (for building some wheels) ----
# Install, then remove after pip install to keep image lean
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (cache-friendly) ----
COPY requirements.txt .
RUN pip install -r requirements.txt

# Optionally remove build tools after wheels are installed
RUN apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ---- App code & model artifacts ----
COPY src/ src/
COPY app/ app/
COPY models/ models/

# If you REALLY need sample data inside the image, uncomment the next line:
# COPY data/ data/

# Expose FastAPI port
EXPOSE 8000

# Drop privileges
USER appuser

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
