# Projet 2 — Prédiction énergétique et estimation de la facturation

Projet d'analyse des données énergétiques de SUP'COM, de comparaison de modèles de prédiction, d'estimation tarifaire et de visualisation dans un tableau de bord Streamlit.

## Démarrage

1. Les fichiers SINERT sont à conserver dans `data/raw/SINERT_DATA_CONCENTRATOR/`, organisés par mois.
2. Installer les dépendances :

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Commencer par `notebooks/01_exploration.ipynb`.
4. Exécuter `notebooks/02_pretraitement.ipynb` pour produire `data/processed/sinert_features.csv` avec les trous courts interpolés et les retards temporels.
5. Exécuter `notebooks/03_visualisation.ipynb` pour produire les graphiques dans `results/`.
6. Exécuter `notebooks/04_parametres_consommation.ipynb` pour extraire les totaux, pics, creux et indicateurs de comportement.
7. Exécuter `notebooks/05_structuration_fenetres.ipynb` pour créer les fenêtres de 672 pas et les 11 variables caractéristiques.
8. Exécuter `notebooks/06_benchmark.ipynb` pour entraîner et comparer les modèles de prédiction.
   Pour l'exécution Kaggle de la phase 2 complète, utiliser `notebooks/phase2_kaggle_benchmark.ipynb` avec `sinert_features.csv` comme dataset d'entrée.
9. Exécuter `notebooks/07_facturation.ipynb` pour calculer la facture réelle et prédite, lorsque la grille tarifaire validée est disponible.
10. Exécuter `notebooks/08_evaluation_multicriteres.ipynb` pour produire le classement multi-critères de la Phase 4.
11. Lancer le tableau de bord quand les données préparées et les résultats existent :

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
   ```

## Organisation

- `data/raw/` : fichiers originaux, jamais modifiés. Les exports SINERT observés sont des CSV séparés par `;` et des fichiers Excel.
- `data/processed/` : données nettoyées et fenêtres temporelles.
- `notebooks/` : exploration et expérimentations.
- `src/` : code réutilisable du projet.
- `app/` : interface Streamlit.
- `results/` : prédictions, métriques et graphiques exportés.

## Règles importantes

- Les séparations entraînement/validation/test doivent respecter l'ordre temporel.
- Les transformations apprises, comme la normalisation, doivent être ajustées uniquement sur l'entraînement.
- Les tarifs doivent être configurés dans `config/tariff.example.json`, puis adaptés avec la grille tarifaire validée.
