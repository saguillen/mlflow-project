#!/bin/bash
set -e

# ─── Auto-detectar GCP Project ID (mismo mecanismo que el MLflow server) ───
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

echo "Starting MLflow Model Serving..."
echo "  Model: models:/${MODEL_NAME}/Production"
echo "  Tracking URI: ${MLFLOW_TRACKING_URI}"
echo "  Project: ${GCLOUD_PROJECT}"

exec mlflow models serve \
    --model-uri "models:/${MODEL_NAME}/Production" \
    --host 0.0.0.0 \
    --port 5001 \
    --no-conda \
    --env-manager=local
