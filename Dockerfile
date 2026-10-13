# Hugging Face Spaces (Docker SDK) + general deployment.
FROM python:3.12-slim

# ffmpeg for cutting/rendering, DejaVu for burned-in captions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects the app on port 7860
ENV DATA_DIR=/app/data PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
