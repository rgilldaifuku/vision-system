# Multi-stage build for ARM (Raspoberry  Pi) and x86
# Build: docker buildx build --platform linux/arm64 -t vision-system:latest
# Run on Pi: docker run --device /dev/video0 --device /dev/mem -v $(pwd)/data:/app/data vision-system:latest

ARG PYTHON_VERSION=3.11

#Stage 1: Builder  (compile dependencies)
FROM python:${PYTHON_VERSION}-slim-bullseye as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libjped-dev \
    libopenblas-dev \
    libjarfbuzz0b \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip -install --user --no-cache-dir -r requirements.txt

#Stage 2: Runtime (minimal)
FROM python:${PYTHON_VERSION}-slim-bullseye

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libatlas3-base \
    libjpeg62-turbo \
    libopenblas0 \
    libharfbuzz0b \
    libwebp6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Add local pip installs to PATH
ENV PATH=/root/.local/bin:$PATH

#Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY requirements/txt .

# Create directories for logs and review images
RUN mkdir -p data/logs data/review_images

# Default: run headless inference
# Override with: docker run vision-system:latest python app/main.py --model models/mou
ENTRYPOINT ["python", "-u", "app/headless_runner.py"]
CMD ["--model", ",models/mouse/best.pt"]