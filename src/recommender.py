"""Funciones de recomendación e historial de usuarios."""

import pandas as pd

from src.model import MatrixFactorization


def get_user_history(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_id: int,
) -> pd.DataFrame:
    """Devuelve las películas valoradas por un usuario, de mayor a menor."""
    columnas_necesarias_valoraciones = {"userId", "movieId", "rating"}
    columnas_faltantes_valoraciones = (
        columnas_necesarias_valoraciones - set(ratings_df.columns)
    )
    if columnas_faltantes_valoraciones:
        raise ValueError(
            "El DataFrame de valoraciones no contiene las columnas: "
            f"{sorted(columnas_faltantes_valoraciones)}"
        )

    columnas_necesarias_peliculas = {"movieId", "title", "genres"}
    columnas_faltantes_peliculas = (
        columnas_necesarias_peliculas - set(movies_df.columns)
    )
    if columnas_faltantes_peliculas:
        raise ValueError(
            "El DataFrame de películas no contiene las columnas: "
            f"{sorted(columnas_faltantes_peliculas)}"
        )

    valoraciones_usuario = ratings_df[
        ratings_df["userId"] == user_id
    ].copy()

    historial = valoraciones_usuario.merge(
        movies_df,
        on="movieId",
        how="inner",
    )
    historial = historial.sort_values(
        by="rating",
        ascending=False,
        kind="mergesort",
    )

    return historial[
        ["movieId", "title", "genres", "rating"]
    ].reset_index(drop=True)


def recommend_top_n(
    model: MatrixFactorization,
    movies_df: pd.DataFrame,
    user_id: int,
    n: int = 10,
) -> pd.DataFrame:
    """Devuelve las mejores películas que todavía no ha visto el usuario."""
    if n <= 0:
        return pd.DataFrame(
            columns=["movieId", "title", "genres", "predicted_rating"]
        )

    columnas_necesarias = {"movieId", "title", "genres"}
    columnas_faltantes = columnas_necesarias - set(movies_df.columns)
    if columnas_faltantes:
        raise ValueError(
            "El DataFrame de películas no contiene las columnas: "
            f"{sorted(columnas_faltantes)}"
        )
    if not model.esta_entrenado:
        raise RuntimeError(
            "El modelo debe entrenarse antes de generar recomendaciones."
        )

    peliculas_candidatas = movies_df.copy()
    usuarios_mapeados = model.mapeos["user2idx"]

    if user_id in usuarios_mapeados:
        peliculas_vistas = model.mapeos["user_seen_movies"].get(user_id, set())
        peliculas_candidatas = peliculas_candidatas[
            ~peliculas_candidatas["movieId"].isin(peliculas_vistas)
        ]

    if peliculas_candidatas.empty:
        return pd.DataFrame(
            columns=["movieId", "title", "genres", "predicted_rating"]
        )

    peliculas_candidatas["predicted_rating"] = peliculas_candidatas[
        "movieId"
    ].apply(
        lambda movie_id: model.predict_one(user_id, movie_id)
    )

    recomendaciones = peliculas_candidatas.sort_values(
        by="predicted_rating",
        ascending=False,
        kind="mergesort",
    ).head(n)

    return recomendaciones[
        ["movieId", "title", "genres", "predicted_rating"]
    ].reset_index(drop=True)
