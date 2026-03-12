FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.18.0 \
    psycopg2-binary==2.9.10 \
    google-cloud-storage==2.19.0 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==1.26.4

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
