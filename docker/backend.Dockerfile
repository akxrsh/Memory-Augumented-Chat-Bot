# Use standard slim Python image
FROM python:3.11-slim

# Set environment paths
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies (needed for compiling some C extensions if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files and frontend files (since backend mounts frontend statically)
COPY backend /app/backend
COPY frontend /app/frontend

# Set Python path to find backend modules
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run uvicorn server
CMD ["python", "backend/main.py"]
