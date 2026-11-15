# Especificacion del proyecto

## 1. Objetivo
Implementar desde cero en Python un sistema de recomendación basado en Matrix Factorization (MF) con sesgos entrenado mediante SGD sobre el dataset movies.csv y ratings.csv

## 2. Datos y Partición (Train / Test)
- Entrada:  
`data/ratings.csv` (`userId`, `movieId`, `rating`, `timestamp`) y `data/movies.csv` (`movieId`, `title`, `genres`).
- Partición 80/20:
Para cada usuario, se ordenan sus valoraciones mediante `timestamp`. El 80% inicial va a `train` y el 20% más reciente va a `test`.
- Regla de seguridad: Si un usuario tiene pocas valoraciones, se garantiza que al menos 1 valoración permanezca en `train` (ej.`train_size = max(1, int(0.8 * n))`).
- Prevención de Data Leakage: 
La media global mu, los diccionarios de mapeo de IDs a índices internos y el historial de películas vistas se construyen -exclusivamente a partir del conjunto `train`.

## 3. Modelo Matemático
- Predicción: r_ui = mu + b_u + b_i + p_u*q_i. dentro de los valores [0.5 , 5.0]
- Inicialización: Sesgos b_u, b_i inicializados a `0.0`. Matrices latentes P y Q inicializadas con distribución normal pequeña $\mathcal{N}(0, 0.1)$ 
- Entrenamiento : En cada época se barajan aleatoriamente las valoraciones de `train` y se actualizan los parámetros minimizando:
  $$L = \sum_{(u,i)} (r_{ui} - \hat{r}_{ui})^2 + \lambda \left( \Vert{}\mathbf{p}_u\Vert{}^2 + \Vert{}\mathbf{q}_i\Vert{}^2 + b_u^2 + b_i^2 \right)$$
- Hiperparámetros por defecto: `k=20`, `lr=0.005`, `lambda_=0.02`, `epochs=20`.

## 4. Casos Extremos
1. Usuario o película desconocidos en test:
   - Usuario conocido, película nueva: $b_i = 0$, $\mathbf{q}_i = \mathbf{0} \implies \hat{r}_{ui} = \mu + b_u$.
   - Usuario nuevo, película conocida: $b_u = 0$, $\mathbf{p}_u = \mathbf{0} \implies \hat{r}_{ui} = \mu + b_i$.
   - Ambos nuevos: $\hat{r}_{ui} = \mu$.
2. Recomendación a usuario desconocido: Si se pide recomendar a un `userId` que no existe en `train`, se devuelven las $N$ películas no vistas con mayor valoración media en `train`.
3. Usuario sin candidatos: Si un usuario ya ha valorado todas las películas de `train`, se devuelve una lista/DataFrame vacío sin lanzar excepción.
4. Rango de predicción: Toda predicción se acota estrictamente al intervalo `[0.5, 5.0]`.

## 5. Evaluación y Criterios de Aceptación
- Métricas en test: RMSE, MAE y NDCG@10 (usando el `rating` real como grado de relevancia).
  - Nota NDCG@10: Se calcula por usuario sobre sus ítems en `test` y se promedia entre todos los usuarios que tengan al menos 1 valoración en `test`.
- Criterios de aceptación:
  - La pérdida MSE de entrenamiento disminuye de forma consistente por época.
  - `RMSE < 0.95` y `MAE ≈ 0.67` sobre el conjunto `test` con los parámetros por defecto.
  - El notebook se ejecuta de principio a fin (`Restart & Run All`) sin errores ni warnings críticos.