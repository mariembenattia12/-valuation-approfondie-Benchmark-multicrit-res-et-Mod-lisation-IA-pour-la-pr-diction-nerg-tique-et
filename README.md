# Analyse SINERT — phase 1

Ce dépôt contient le pipeline de préparation des données de la personne A et
l'analyse du comportement énergétique réalisée par la personne B.

## Travail réalisé par la personne B

L'analyse a été menée dans Kaggle à partir du fichier propre produit par le
pipeline A (`sinert_clean.parquet`). Le compteur étudié est `EM_VIRTUAL`, avec
une mesure toutes les 15 minutes sur la période du 1er juin 2022 au
1er mars 2026.

### 1. Contrôle de la qualité

Le fichier propre contient 131 425 intervalles et 18 colonnes. Le pas temporel
est régulier et vaut 15 minutes pour les 131 424 différences observées.
Les variables physiques analysées sont :

- `delta_ea_kwh` : énergie active consommée pendant un intervalle ;
- `delta_er_kvarh` : énergie réactive consommée pendant un intervalle ;
- `pmax_kw` : puissance active maximale ;
- `smax_kva` : puissance apparente maximale.

Après nettoyage, environ 2,19 % des valeurs physiques restent manquantes et
3 401 lignes contiennent au moins une valeur imputée. Les valeurs manquantes
restantes ne sont pas remplacées arbitrairement par zéro.

### 2. Visualisations

Les séries temporelles des quatre variables physiques ont été visualisées pour
repérer les tendances, les périodes de faible activité et les pointes. Les
valeurs imputées ont également été distinguées afin de ne pas les confondre
avec des mesures originales.

Un profil moyen par créneau horaire de 15 minutes et une carte de chaleur
`jour de la semaine × heure` ont été produits. Les maxima du profil journalier
moyen sont :

- énergie active : 18,82 kWh à 11:15 ;
- puissance active : 75,21 kW à 11:30 ;
- puissance apparente : 78,93 kVA à 11:30.

Les minima moyens apparaissent autour de 04:30–05:15, mais la consommation ne
tombe pas à zéro : il s'agit d'une charge de fond.

### 3. Énergie cumulée par période

La colonne `delta_ea_kwh` a été agrégée par jour, semaine, mois et année. Les
valeurs `<NA>` indiquent qu'aucune mesure valide n'était disponible ; elles ne
signifient pas une consommation nulle.

Les premières journées de juin 2022 sont incomplètes. Février 2026 est
également incomplet et ne doit pas être comparé directement aux années
complètes. Les totaux annuels observés sont :

| Année | Énergie active |
|---|---:|
| 2022 | 347 169,3 kWh, période partielle |
| 2023 | 434 651,8 kWh |
| 2024 | 517 674,8 kWh |
| 2025 | 545 959,9 kWh |
| 2026 | 32 486,0 kWh, période partielle |

### 4. Charge de base et pointes

La charge de base est décrite par le minimum absolu et par le 5e percentile,
qui est plus robuste aux valeurs extrêmes. Les pointes sont définies comme les
créneaux dont `pmax_kw` est supérieur ou égal au 90e percentile. Ainsi, les
10 % de créneaux les plus élevés sont étudiés comme des périodes de pointe.

Un créneau correspond à une mesure de 15 minutes. Un épisode de pointe est une
suite de créneaux de pointe consécutifs. Le fichier
`reports/peak_episodes.csv` contient, pour chaque épisode, son début, sa fin,
sa puissance maximale, son nombre de créneaux et sa durée en heures.

Cette méthode distingue une pointe isolée d'une pointe prolongée et évite de
définir le comportement du site à partir d'un seul maximum exceptionnel.

### 5. Comparaison des périodes

Les jours sont séparés avec `is_nonworking_day` :

- `0` : jour ouvré ;
- `1` : samedi, dimanche ou jour férié fourni.

Les moyennes, médianes, écarts-types, variances, minima et maxima de l'énergie,
de `pmax_kw` et de `smax_kva` ont été comparés entre jours ouvrés et non
ouvrés. Les vacances et semaines d'études n'ont pas été inventées, car leur
calendrier officiel n'était pas fourni.

### 6. Analyse saisonnière

Les mois ont été regroupés en quatre saisons. Les moyennes par créneau de
15 minutes sont :

| Saison | Énergie moyenne | Pmax moyenne | Smax moyenne |
|---|---:|---:|---:|
| Été | 20,87 kWh | 83,41 kW | 87,69 kVA |
| Automne | 18,46 kWh | 73,83 kW | 77,24 kVA |
| Hiver | 9,59 kWh | 38,33 kW | 40,94 kVA |
| Printemps | 9,41 kWh | 37,62 kW | 41,40 kVA |

L'été est donc la saison présentant l'activité moyenne la plus élevée.
L'utilisation accrue de la climatisation est une hypothèse possible, mais elle
devrait être confirmée par des données météorologiques ou d'occupation.

### 7. Activation et désactivation

Avec un seuil de 1 kW, 99,95 % des créneaux sont actifs. Le site possède donc
une charge de fond presque permanente ; ce seuil ne permet pas d'identifier
l'occupation humaine. Des seuils complémentaires ont été testés :

| Seuil | Créneaux au-dessus du seuil |
|---|---:|
| 1 kW | 99,95 % |
| 10 kW | 99,94 % |
| 30 kW | 79,38 % |
| 50 kW | 48,81 % |
| 70 kW | 30,96 % |

Le seuil de 1 kW est donc interprété comme un indicateur de charge minimale.
Les seuils de 30, 50 et 70 kW décrivent mieux les niveaux d'activité
importante. L'heure d'activation/désactivation calculée avec 1 kW est presque
toujours 00:00–23:45, avec une durée active moyenne de 23,97 heures par jour ;
cela ne signifie pas que le site est occupé 24 heures sur 24.

### Livrables de l'analyse B

- `analyse-personne-b.ipynb` : notebook d'analyse ;
- `figures/` : graphiques produits ;
- `reports/` : tableaux statistiques exportés, dont
  `peak_episodes.csv`, ainsi que l'interprétation finale dans
  `interpretation_personne_b.md`.

Les fichiers Parquet détaillés et les données brutes restent dans Kaggle et ne
doivent pas être publiés dans GitHub s'ils sont volumineux ou confidentiels.

## Pipeline de préparation de la personne A

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
