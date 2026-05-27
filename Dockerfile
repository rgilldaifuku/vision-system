# Lightweight runtime container for ARM64 Raspberry Pi or x86 Linux.
# Build: docker build -t vision-system:latest .
# Run with a USB camera: docker run --device /dev/video0 -p 8000:8000 vision-system:latest

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libopenblas0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

RUN mkdir -p data/datasets data/logs data/review_images models

EXPOSE 8000

ENTRYPOINT ["python", "-m", "app.runtime.detector_service"]
CMD ["--profile", "yellow_daifuku", "--camera", "0", "--host", "0.0.0.0", "--port", "8000", "--imgsz", "256", "--frame-width", "424", "--frame-height", "240", "--inference-interval-ms", "300"]
