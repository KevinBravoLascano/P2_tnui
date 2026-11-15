# Arquitectura del Proyecto

## 1. Estructura de Directorios
```text
P2_tnui/
├── data/
│   ├── ratings.csv
│   └── movies.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Carga, estadísticas, mapeo de IDs y split temporal
│   ├── model.py          # Clase MatrixFactorization (entrenamiento SGD y predicción)
│   ├── recommender.py    # Generación de Top-N recomendaciones e historial de usuario
│   └── evaluation.py     # Cálculo de RMSE, MAE y NDCG@10
├──docs/   
    ├── notebook_demo.ipynb   # Demostración de los 9 pasos requeridos
    ├── specification.md
    ├── architecture.md
    └── tasks.md
    ├──reglesIA.md
```

## 2. Diseño Modular (`src/`)

### `src/data_loader.py`
- `load_data(ratings_path: str, movies_path: str) -> tuple[pd.DataFrame, pd.DataFrame]`
  - Lee ambos CSV y valida columnas esperadas.
- `temporal_train_test_split(ratings_df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]`
  - Ordena por `userId` y `timestamp`. Separa 80/20 por usuario manteniendo $\ge 1$ registro en `train`.
- `build_mappings(train_df: pd.DataFrame) -> dict`
  - Genera `user2idx`, `idx2user`, `movie2idx`, `idx2movie`, y un diccionario `user_seen_movies` (`userId -> set(movieId)`).
- `compute_dataset_stats(ratings_df: pd.DataFrame) -> dict`
  - Calcula número de usuarios, número de películas, total de valoraciones y densidad de la matriz ($\frac{\text{valoraciones}}{\text{usuarios} \times \text{películas}}$).

### `src/model.py`
- `class MatrixFactorization`:
  - `__init__(self, k: int = 20, lr: float = 0.005, lambda_: float = 0.02, epochs: int = 20, seed: int = 42)`
  - `fit(self, train_df: pd.DataFrame, mappings: dict) -> list[float]`
    - Calcula $\mu$, inicializa $P, Q, b_u, b_i$, ejecuta el bucle SGD barajando muestras en cada época y devuelve la lista del MSE de entrenamiento por época.
  - `predict_one(self, user_id: int, movie_id: int) -> float`
    - Predice una pareja `(userId, movieId)` aplicando las reglas de *cold-start* y `np.clip(..., 0.5, 5.0)`.
  - `predict_batch(self, pairs: list[tuple[int, int]]) -> np.ndarray`
    - Vectoriza o itera eficientemente para evaluar todo el conjunto `test`.

### `src/recommender.py`
- `recommend_top_n(model: MatrixFactorization, movies_df: pd.DataFrame, user_id: int, n: int = 10) -> pd.DataFrame`
  - Filtra las películas conocidas que el usuario **no** ha visto en `train`, predice su valoración, ordena descendentemente y devuelve DataFrame con `movieId`, `title`, `genres`, `predicted_rating`.
- `get_user_history(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int) -> pd.DataFrame`
  - Devuelve las películas valoradas por el usuario ordenadas de mayor a menor `rating` con `movieId`, `title`, `genres`, `rating`.

### `src/evaluation.py`
- `compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float`
- `compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float`
- `compute_ndcg_at_k(test_df: pd.DataFrame, model: MatrixFactorization, k: int = 10) -> float`
  - Agrupa `test_df` por `userId`, ordena por `predicted_rating` para calcular DCG@k y por `rating` real para calcular IDCG@k, devolviendo el promedio.