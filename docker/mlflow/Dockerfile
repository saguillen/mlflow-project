FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.18.0 \
    psycopg2-binary==2.9.10 \
    google-cloud-storage==2.19.0 \
    gunicorn==23.0.0

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
