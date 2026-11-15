# Plan de Implementación por Tareas

## Fase 1: Preparación y Datos
- [x] **T1.1:** Crear estructura de directorios (`src/`, `data/`) y verificar carga de `ratings.csv` y `movies.csv`.
- [x] **T1.2:** Implementar `compute_dataset_stats` y `temporal_train_test_split` en `src/data_loader.py`. Verificar que no hay *data leakage* ni usuarios sin datos en `train`.
- [x] **T1.3:** Implementar `build_mappings` en `src/data_loader.py` para traducir `userId` y `movieId` no consecutivos a índices `0..N-1`.

## Fase 2: Modelo Matrix Factorization
- [ ] **T2.1:** Implementar la estructura base de `MatrixFactorization` y el método `predict_one` (incluyendo manejo de *cold-start* y acotación `[0.5, 5.0]`) en `src/model.py`.
- [ ] **T2.2:** Implementar el entrenamiento `fit` con SGD y regularización L2 en `src/model.py`[cite: 3]. Comprobar con un test rápido que la pérdida MSE baja en cada época.

## Fase 3: Recomendación y Evaluación
- [ ] **T3.1:** Implementar `recommend_top_n` y `get_user_history` en `src/recommender.py`. Probar casos extremos (usuario desconocido, $N$ mayor que películas disponibles).
- [ ] **T3.2:** Implementar `compute_rmse`, `compute_mae` y `compute_ndcg_at_k` en `src/evaluation.py`.
- [ ] **T3.3:** Ejecutar evaluación completa sobre el conjunto `test` y verificar que `RMSE < 0.95` y `MAE ≈ 0.67`.

## Fase 4: Notebook Demostrativo y Revisión Final
- [ ] **T4.1:** Crear `notebook_demo.ipynb` siguiendo exactamente los 9 puntos de la sección 3.3 (incluyendo gráfica de curva de aprendizaje con `matplotlib`).
- [ ] **T4.2:** Ejecutar `Restart & Run All` en el notebook, revisar cumplimiento de PEP8 en `src/` y comprobar que las decisiones de diseño coinciden con `specification.md` y `architecture.md`.