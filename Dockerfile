FROM python:3.10-slim

WORKDIR /app

# System deps for pyahocorasick
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY filter/ ./filter/
COPY process_articles.py .

# MODEL_DIR must be mounted or set at runtime, e.g.:
#   docker run -e MODEL_DIR=/models/distilled_v2 -v /host/models:/models ...
ENV MODEL_DIR=""
ENV PORT=8003
ENV DEFAULT_THRESHOLD=0.25
ENV DEVICE=auto

EXPOSE 8003

CMD ["python", "api/app.py"]
