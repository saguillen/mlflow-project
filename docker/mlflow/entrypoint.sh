#!/bin/bash
set -e

# ─── Auto-detectar GCP Project ID ───
# Las credenciales tipo "authorized_user" (generadas con gcloud auth application-default login)
# NO incluyen el Project ID, a diferencia de las service account keys.
# La librería google-cloud-storage lo necesita para operar con GCS.
# Este bloque lo extrae automáticamente del campo "quota_project_id" o "project_id"
# del fichero de credenciales, evitando tener que hardcodear el Project ID en el
# docker-compose.yml o en variables de entorno.
if [ -z "$GCLOUD_PROJECT" ]; then
    export GCLOUD_PROJECT=$(python3 -c "
import json, os
cred_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
if cred_file and os.path.exists(cred_file):
    with open(cred_file) as f:
        creds = json.load(f)
    print(creds.get('quota_project_id', creds.get('project_id', '')))
" 2>/dev/null)
    export GOOGLE_CLOUD_PROJECT="$GCLOUD_PROJECT"
    echo "  Auto-detected GCP Project: ${GCLOUD_PROJECT}"
fi

echo "Starting MLflow Tracking Server..."
echo "  Backend: ${MLFLOW_BACKEND_URI}"
echo "  Artifacts: ${MLFLOW_ARTIFACT_ROOT}"
echo "  Project: ${GCLOUD_PROJECT}"

exec mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2 \
    --gunicorn-opts "--worker-class gthread --threads 2 --timeout 120"
