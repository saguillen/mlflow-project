# Archivo: scripts/train_model.py

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor headless
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile

# Tracking URI apunta a Nginx (puerto 80)
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:80'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def load_data():
    """Carga datos procesados"""
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Genera confusion matrix como archivo temporal"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad Wine', 'Good Wine'],
                yticklabels=['Bad Wine', 'Good Wine'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmpfile.name)
    plt.close()
    return tmpfile.name


def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """Genera plot de feature importance"""
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()

    tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmpfile.name)
    plt.close()
    return tmpfile.name


def train_model(model_type='random_forest', hyperparameters=None):
    """
    Entrena modelo con MLflow tracking.

    Args:
        model_type: 'random_forest', 'logistic_regression', o 'svm'
        hyperparameters: dict de hiperparámetros (None usa defaults)
    """
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'wine-quality-classification')
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=f"{model_type}_training") as run:
        print(f"\n{'=' * 60}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"Model Type: {model_type}")
        print(f"{'=' * 60}\n")

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Seleccionar y configurar modelo
        model_configs = {
            'random_forest': {
                'class': RandomForestClassifier,
                'defaults': {'n_estimators': 100, 'max_depth': 10,
                            'min_samples_split': 5, 'random_state': 42}
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'defaults': {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
            },
            'svm': {
                'class': SVC,
                'defaults': {'C': 1.0, 'kernel': 'rbf',
                            'random_state': 42, 'probability': True}
            }
        }

        if model_type not in model_configs:
            raise ValueError(f"Unknown model_type: {model_type}")

        config = model_configs[model_type]
        params = {**config['defaults'], **(hyperparameters or {})}
        model = config['class'](**params)

        # Log hiperparámetros
        mlflow.log_params(params)

        # Entrenar
        print("Entrenando modelo...")
        model.fit(X_train, y_train)
        print("✓ Entrenamiento completado")

        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Métricas
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test)
        }
        mlflow.log_metrics(metrics)

        print("\nMétricas:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test,
                                    target_names=['Bad Wine', 'Good Wine']))

        # Artifacts: Confusion Matrix
        cm_file = plot_confusion_matrix(y_test, y_pred_test)
        mlflow.log_artifact(cm_file, "plots")
        os.unlink(cm_file)

        # Artifacts: Feature Importance
        fi_file = plot_feature_importance(model, X_train.columns.tolist())
        if fi_file:
            mlflow.log_artifact(fi_file, "plots")
            os.unlink(fi_file)

        # Log modelo con signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"wine-quality-{model_type}"
        )

        print(f"\n✓ Modelo registrado: wine-quality-{model_type}")
        print(f"✓ Run ID: {run.info.run_id}")
        print(f"✓ Artifact URI: {mlflow.get_artifact_uri()}")

        return run.info.run_id, metrics


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'random_forest'
    run_id, metrics = train_model(model_type=model_type)

    print(f"\n{'=' * 60}")
    print(f"Training completado!")
    print(f"Ver resultados en MLflow UI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    print(f"{'=' * 60}")