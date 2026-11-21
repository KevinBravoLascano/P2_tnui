"""Métricas de evaluación del sistema de recomendación."""

import numpy as np
import pandas as pd

from src.model import MatrixFactorization


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el error cuadrático medio en su escala original."""
    valores_reales = np.asarray(y_true, dtype=float)
    valores_predichos = np.asarray(y_pred, dtype=float)

    if valores_reales.size == 0:
        raise ValueError("Los valores reales no pueden estar vacíos.")
    if valores_reales.shape != valores_predichos.shape:
        raise ValueError(
            "Los arrays de valores reales y predichos deben tener la misma forma."
        )

    errores_cuadraticos = (valores_reales - valores_predichos) ** 2
    return float(np.sqrt(np.mean(errores_cuadraticos)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la media de los errores absolutos."""
    valores_reales = np.asarray(y_true, dtype=float)
    valores_predichos = np.asarray(y_pred, dtype=float)

    if valores_reales.size == 0:
        raise ValueError("Los valores reales no pueden estar vacíos.")
    if valores_reales.shape != valores_predichos.shape:
        raise ValueError(
            "Los arrays de valores reales y predichos deben tener la misma forma."
        )

    errores_absolutos = np.abs(valores_reales - valores_predichos)
    return float(np.mean(errores_absolutos))


def dcg_at_k(relevancias: np.ndarray, k: int) -> float:
    """Calcula DCG@k para una lista de relevancias ya ordenada."""
    if k <= 0:
        return 0.0

    relevancias_array = np.asarray(relevancias, dtype=float)
    relevancias_cortadas = relevancias_array[:k]
    if relevancias_cortadas.size == 0:
        return 0.0

    posiciones = np.arange(1, relevancias_cortadas.size + 1)
    ganancias = relevancias_cortadas / np.log2(posiciones + 1)
    return float(np.sum(ganancias))


def idcg_at_k(relevancias: np.ndarray, k: int) -> float:
    """Calcula el DCG ideal ordenando las relevancias de mayor a menor."""
    if k <= 0:
        return 0.0

    relevancias_array = np.asarray(relevancias, dtype=float)
    relevancias_ordenadas = np.sort(relevancias_array)[::-1]
    return dcg_at_k(relevancias_ordenadas, k)


def compute_ndcg_at_k(
    test_df: pd.DataFrame,
    model: MatrixFactorization,
    k: int = 10,
) -> float:
    """Calcula el NDCG@k medio para los usuarios con datos en test."""
    if k <= 0:
        return 0.0

    columnas_necesarias = {"userId", "movieId", "rating"}
    columnas_faltantes = columnas_necesarias - set(test_df.columns)
    if columnas_faltantes:
        raise ValueError(
            "El DataFrame de prueba no contiene las columnas: "
            f"{sorted(columnas_faltantes)}"
        )
    if not model.esta_entrenado:
        raise RuntimeError(
            "El modelo debe entrenarse antes de calcular NDCG."
        )

    valores_ndcg: list[float] = []

    for _, valoraciones_usuario in test_df.groupby("userId", sort=False):
        relevancias_reales = valoraciones_usuario["rating"].to_numpy(
            dtype=float
        )
        valoraciones_usuario = valoraciones_usuario.copy()
        valoraciones_usuario["predicted_rating"] = [
            model.predict_one(user_id, movie_id)
            for user_id, movie_id in zip(
                valoraciones_usuario["userId"],
                valoraciones_usuario["movieId"],
            )
        ]

        orden_predicho = valoraciones_usuario.sort_values(
            by="predicted_rating",
            ascending=False,
            kind="mergesort",
        )
        relevancias_predichas = orden_predicho["rating"].to_numpy(
            dtype=float
        )

        dcg = dcg_at_k(relevancias_predichas, k)
        idcg = idcg_at_k(relevancias_reales, k)
        if idcg > 0:
            valores_ndcg.append(dcg / idcg)

    if not valores_ndcg:
        return 0.0

    return float(np.mean(valores_ndcg))
