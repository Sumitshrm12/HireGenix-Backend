# Stage 1: Build stage with all dependencies
# Using Python 3.11.11 (closest available slim image to local 3.11.14)
FROM python:3.11.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies with specific pip version (matching local .venv)
RUN pip install --no-cache-dir pip==25.3 wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt --prefer-binary --use-pep517

# Stage 2: Runtime stage (smaller image)
FROM python:3.11.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Install Playwright browsers (required for Crawl4AI)
RUN playwright install --with-deps chromium

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "uvicorn main:socket_app --host 0.0.0.0 --port ${PORT}"]
