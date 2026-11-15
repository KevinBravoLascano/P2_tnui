"""Funciones para cargar los datos del sistema de recomendación."""

import pandas as pd


def load_data(
    ratings_path: str, movies_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga las valoraciones y las películas desde sus archivos CSV.

    También comprueba que cada archivo tenga las columnas que necesita el
    resto del proyecto.
    """
    datos_valoraciones = pd.read_csv(ratings_path)
    datos_peliculas = pd.read_csv(movies_path)

    columnas_valoraciones_esperadas = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
    }
    columnas_peliculas_esperadas = {"movieId", "title", "genres"}

    columnas_faltantes_valoraciones = (
        columnas_valoraciones_esperadas - set(datos_valoraciones.columns)
    )
    columnas_faltantes_peliculas = (
        columnas_peliculas_esperadas - set(datos_peliculas.columns)
    )

    if columnas_faltantes_valoraciones:
        raise ValueError(
            "El archivo de valoraciones no contiene las columnas: "
            f"{sorted(columnas_faltantes_valoraciones)}"
        )

    if columnas_faltantes_peliculas:
        raise ValueError(
            "El archivo de películas no contiene las columnas: "
            f"{sorted(columnas_faltantes_peliculas)}"
        )

    return datos_valoraciones, datos_peliculas


def temporal_train_test_split(
    ratings_df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide las valoraciones de cada usuario respetando su orden temporal.

    Cada usuario conserva al menos una valoración en el conjunto de
    entrenamiento. Las filas de prueba son siempre las valoraciones más
    recientes de ese usuario.
    """
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio debe ser mayor que 0 y como máximo 1.")

    columnas_necesarias = {"userId", "timestamp"}
    columnas_faltantes = columnas_necesarias - set(ratings_df.columns)
    if columnas_faltantes:
        raise ValueError(
            "El DataFrame de valoraciones no contiene las columnas: "
            f"{sorted(columnas_faltantes)}"
        )

    # mergesort es estable y conserva el orden original si coinciden timestamps.
    valoraciones_ordenadas = ratings_df.sort_values(
        by=["userId", "timestamp"],
        kind="mergesort",
    )

    partes_entrenamiento: list[pd.DataFrame] = []
    partes_prueba: list[pd.DataFrame] = []

    for _, valoraciones_usuario in valoraciones_ordenadas.groupby(
        "userId", sort=False
    ):
        cantidad_entrenamiento = max(
            1, int(train_ratio * len(valoraciones_usuario))
        )
        partes_entrenamiento.append(
            valoraciones_usuario.iloc[:cantidad_entrenamiento]
        )
        partes_prueba.append(valoraciones_usuario.iloc[cantidad_entrenamiento:])

    if partes_entrenamiento:
        datos_entrenamiento = pd.concat(
            partes_entrenamiento,
            ignore_index=True,
        )
    else:
        datos_entrenamiento = ratings_df.iloc[0:0].copy()

    if partes_prueba:
        datos_prueba = pd.concat(
            partes_prueba,
            ignore_index=True,
        )
    else:
        datos_prueba = ratings_df.iloc[0:0].copy()

    return datos_entrenamiento, datos_prueba


def compute_dataset_stats(ratings_df: pd.DataFrame) -> dict:
    """Calcula usuarios, películas, valoraciones y densidad del conjunto."""
    columnas_necesarias = {"userId", "movieId"}
    columnas_faltantes = columnas_necesarias - set(ratings_df.columns)
    if columnas_faltantes:
        raise ValueError(
            "El DataFrame de valoraciones no contiene las columnas: "
            f"{sorted(columnas_faltantes)}"
        )

    cantidad_usuarios = ratings_df["userId"].nunique()
    cantidad_peliculas = ratings_df["movieId"].nunique()
    cantidad_valoraciones = len(ratings_df)
    cantidad_combinaciones = cantidad_usuarios * cantidad_peliculas

    if cantidad_combinaciones == 0:
        densidad = 0.0
    else:
        densidad = cantidad_valoraciones / cantidad_combinaciones

    return {
        "n_users": cantidad_usuarios,
        "n_movies": cantidad_peliculas,
        "n_ratings": cantidad_valoraciones,
        "density": densidad,
    }


def build_mappings(train_df: pd.DataFrame) -> dict:
    """Crea los mapeos entre los identificadores originales y sus índices.

    Los mapeos y el historial de películas se construyen solo con
    ``train_df`` para evitar utilizar información del conjunto de prueba.
    """
    columnas_necesarias = {"userId", "movieId"}
    columnas_faltantes = columnas_necesarias - set(train_df.columns)
    if columnas_faltantes:
        raise ValueError(
            "El DataFrame de entrenamiento no contiene las columnas: "
            f"{sorted(columnas_faltantes)}"
        )

    identificadores_usuarios = train_df["userId"].drop_duplicates().tolist()
    identificadores_peliculas = train_df["movieId"].drop_duplicates().tolist()

    user2idx = {
        identificador: indice
        for indice, identificador in enumerate(identificadores_usuarios)
    }
    movie2idx = {
        identificador: indice
        for indice, identificador in enumerate(identificadores_peliculas)
    }

    idx2user = {
        indice: identificador
        for identificador, indice in user2idx.items()
    }
    idx2movie = {
        indice: identificador
        for identificador, indice in movie2idx.items()
    }

    user_seen_movies: dict = {}
    for identificador_usuario, valoraciones_usuario in train_df.groupby(
        "userId",
        sort=False,
    ):
        user_seen_movies[identificador_usuario] = set(
            valoraciones_usuario["movieId"]
        )

    return {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "movie2idx": movie2idx,
        "idx2movie": idx2movie,
        "user_seen_movies": user_seen_movies,
    }