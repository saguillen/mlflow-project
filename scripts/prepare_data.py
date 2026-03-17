
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import subprocess

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


def load_wine_data():
    """Carga y combina datasets de vino tinto y blanco"""
    red_wine = pd.read_csv(os.path.join(DATA_DIR, 'winequality-red.csv'), sep=';')
    white_wine = pd.read_csv(os.path.join(DATA_DIR, 'winequality-white.csv'), sep=';')

    red_wine['wine_type'] = 0  # Red
    white_wine['wine_type'] = 1  # White

    wine_data = pd.concat([red_wine, white_wine], ignore_index=True)

    print(f"Total samples: {len(wine_data)}")
    print(f"Red wine: {len(red_wine)}, White wine: {len(white_wine)}")

    return wine_data


def create_quality_binary(df):
    """
    Convierte quality (0-10) a clasificación binaria:
    - quality >= 7: good wine (1)
    - quality < 7: bad wine (0)
    """
    df['quality_binary'] = (df['quality'] >= 7).astype(int)

    print("\nDistribución de calidad:")
    print(df['quality'].value_counts().sort_index())
    print(f"\nClase positiva (good wine): {df['quality_binary'].sum()}")
    print(f"Clase negativa (bad wine): {(1 - df['quality_binary']).sum()}")
    print(f"Balance: {df['quality_binary'].mean():.2%}")

    return df


def prepare_features(df):
    """Prepara features para modelo"""
    feature_cols = [col for col in df.columns if col not in ['quality', 'quality_binary']]

    X = df[feature_cols]
    y = df['quality_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {list(feature_cols)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    """Guarda datos procesados localmente y sube a GCS"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False, header=True)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False, header=True)

    with open(os.path.join(PROCESSED_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print("\n✓ Datos procesados guardados en data/processed/")

    # Subir a GCS
    bucket_name = os.environ.get('BUCKET_NAME')
    if bucket_name:
        subprocess.run(
            ['gcloud', 'storage', 'cp', '-r',
             f'{PROCESSED_DIR}/*',
             f'gs://{bucket_name}/datasets/wine-quality/'],
            check=True
        )
        print(f"✓ Datos subidos a gs://{bucket_name}/datasets/wine-quality/")


if __name__ == "__main__":
    print("=" * 60)
    print("Preparación de Wine Quality Dataset")
    print("=" * 60)

    wine_data = load_wine_data()
    wine_data = create_quality_binary(wine_data)
    X_train, X_test, y_train, y_test, scaler = prepare_features(wine_data)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)

    print("\n" + "=" * 60)
    print("✓ Preparación completada!")
    print("=" * 60
