"""Modelo de factorización matricial para predecir valoraciones."""

import numpy as np
import pandas as pd


class MatrixFactorization:
    """Estructura base del modelo de factorización matricial."""

    def __init__(
        self,
        k: int = 20,
        lr: float = 0.005,
        lambda_: float = 0.02,
        epochs: int = 20,
        seed: int = 42,
    ):
        """Guarda los hiperparámetros y prepara el estado del modelo."""
        if k <= 0:
            raise ValueError("k debe ser un entero positivo.")
        if lr <= 0:
            raise ValueError("lr debe ser mayor que cero.")
        if lambda_ < 0:
            raise ValueError("lambda_ no puede ser negativo.")
        if epochs <= 0:
            raise ValueError("epochs debe ser un entero positivo.")

        self.k = k
        self.lr = lr
        self.lambda_ = lambda_
        self.epochs = epochs
        self.seed = seed

        self.media_global = 0.0
        self.mapeos = {}
        self.sesgos_usuarios = np.array([], dtype=float)
        self.sesgos_peliculas = np.array([], dtype=float)
        self.factores_usuarios = np.empty((0, k), dtype=float)
        self.factores_peliculas = np.empty((0, k), dtype=float)
        self.esta_entrenado = False

    def predict_one(self, user_id: int, movie_id: int) -> float:
        """Predice la valoración de una pareja usuario-película.

        Si un usuario o una película no aparecen en los mapeos de
        entrenamiento, sus parámetros se consideran cero. Así se aplican
        directamente las reglas de *cold-start* de la especificación.
        """
        if not self.esta_entrenado:
            raise RuntimeError(
                "El modelo debe entrenarse antes de realizar predicciones."
            )

        usuarios_mapeados = self.mapeos["user2idx"]
        peliculas_mapeadas = self.mapeos["movie2idx"]

        indice_usuario = usuarios_mapeados.get(user_id)
        indice_pelicula = peliculas_mapeadas.get(movie_id)

        sesgo_usuario = 0.0
        sesgo_pelicula = 0.0
        factores_usuario = np.zeros(self.k)
        factores_pelicula = np.zeros(self.k)

        if indice_usuario is not None:
            sesgo_usuario = self.sesgos_usuarios[indice_usuario]
            factores_usuario = self.factores_usuarios[indice_usuario]

        if indice_pelicula is not None:
            sesgo_pelicula = self.sesgos_peliculas[indice_pelicula]
            factores_pelicula = self.factores_peliculas[indice_pelicula]

        prediccion = (
            self.media_global
            + sesgo_usuario
            + sesgo_pelicula
            + np.dot(factores_usuario, factores_pelicula)
        )

        return float(np.clip(prediccion, 0.5, 5.0))

    def fit(self, train_df: pd.DataFrame, mappings: dict) -> list[float]:
        """Entrena el modelo con SGD y devuelve el MSE de cada época."""
        columnas_necesarias = {"userId", "movieId", "rating"}
        columnas_faltantes = columnas_necesarias - set(train_df.columns)
        if columnas_faltantes:
            raise ValueError(
                "El DataFrame de entrenamiento no contiene las columnas: "
                f"{sorted(columnas_faltantes)}"
            )
        if train_df.empty:
            raise ValueError(
                "El DataFrame de entrenamiento no puede estar vacío."
            )

        claves_necesarias = {"user2idx", "movie2idx"}
        claves_faltantes = claves_necesarias - set(mappings)
        if claves_faltantes:
            raise ValueError(
                "Faltan mapeos necesarios para entrenar: "
                f"{sorted(claves_faltantes)}"
            )

        indices_usuarios = train_df["userId"].map(
            mappings["user2idx"]
        ).to_numpy(dtype=np.int64)
        indices_peliculas = train_df["movieId"].map(
            mappings["movie2idx"]
        ).to_numpy(dtype=np.int64)
        valoraciones = train_df["rating"].to_numpy(dtype=float)

        cantidad_usuarios = len(mappings["user2idx"])
        cantidad_peliculas = len(mappings["movie2idx"])
        generador = np.random.default_rng(self.seed)

        self.media_global = float(np.mean(valoraciones))
        self.sesgos_usuarios = np.zeros(cantidad_usuarios, dtype=float)
        self.sesgos_peliculas = np.zeros(cantidad_peliculas, dtype=float)
        self.factores_usuarios = generador.normal(
            0.0, 0.1, size=(cantidad_usuarios, self.k)
        )
        self.factores_peliculas = generador.normal(
            0.0, 0.1, size=(cantidad_peliculas, self.k)
        )
        self.mapeos = mappings
        self.esta_entrenado = True

        historial_mse: list[float] = []
        indices_muestras = np.arange(len(valoraciones))

        for _ in range(self.epochs):
            generador.shuffle(indices_muestras)

            for indice_muestra in indices_muestras:
                indice_usuario = indices_usuarios[indice_muestra]
                indice_pelicula = indices_peliculas[indice_muestra]
                valoracion_real = valoraciones[indice_muestra]

                sesgo_usuario = self.sesgos_usuarios[indice_usuario]
                sesgo_pelicula = self.sesgos_peliculas[indice_pelicula]
                factores_usuario = self.factores_usuarios[indice_usuario]
                factores_pelicula = self.factores_peliculas[indice_pelicula]

                prediccion = (
                    self.media_global
                    + sesgo_usuario
                    + sesgo_pelicula
                    + np.dot(factores_usuario, factores_pelicula)
                )
                error = valoracion_real - prediccion

                self.sesgos_usuarios[indice_usuario] += self.lr * (
                    error - self.lambda_ * sesgo_usuario
                )
                self.sesgos_peliculas[indice_pelicula] += self.lr * (
                    error - self.lambda_ * sesgo_pelicula
                )

                nuevo_factor_usuario = factores_usuario + self.lr * (
                    error * factores_pelicula
                    - self.lambda_ * factores_usuario
                )
                nuevo_factor_pelicula = factores_pelicula + self.lr * (
                    error * factores_usuario
                    - self.lambda_ * factores_pelicula
                )

                self.factores_usuarios[indice_usuario] = nuevo_factor_usuario
                self.factores_peliculas[indice_pelicula] = (
                    nuevo_factor_pelicula
                )

            predicciones = (
                self.media_global
                + self.sesgos_usuarios[indices_usuarios]
                + self.sesgos_peliculas[indices_peliculas]
                + np.sum(
                    self.factores_usuarios[indices_usuarios]
                    * self.factores_peliculas[indices_peliculas],
                    axis=1,
                )
            )
            historial_mse.append(
                float(np.mean((valoraciones - predicciones) ** 2))
            )

        return historial_mse
