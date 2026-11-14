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

Ce pipeline lit les exports `SINERT_DATA_CONCENTRATOR`, conserve par défaut les
mesures du compteur global `EM_VIRTUAL`, crée une série régulière de 15 minutes,
nettoie les compteurs cumulés et prépare les fenêtres chronologiques.

## Installation et exécution

Depuis la racine du projet, dans PowerShell (exécution locale) :

```powershell
python -m pip install -r requirements.txt
python .\pipeline.py --input "C:\SINERT_DATA_CONCENTRATOR" --output ".\data\processed"
```

Ou donne directement l'archive :

```powershell
python .\pipeline.py --input "C:\SINERT_DATA_CONCENTRATOR.zip" --output ".\data\processed"
```

Le dossier d'entrée peut contenir des exports `.csv` séparés par des points-
virgules et des exports `.xlsx`. Il peut aussi s'agir directement de l'archive
`.zip` contenant ces exports. Les colonnes requises pour le compteur choisi
sont les énergies cumulées actives/réactives et les demandes maximales kW/kVA.
Une colonne manquante ou un fichier illisible provoque une erreur explicite.

## Utilisation dans Kaggle

Le notebook [kaggle_phase1.ipynb](./kaggle_phase1.ipynb) repère un ZIP ou un
dossier d'exports CSV/XLSX sous `/kaggle/input`, lance le pipeline et écrit les livrables dans
`/kaggle/working/phase1`. Dans Kaggle :

1. Crée un Notebook et ajoute le dataset SINERT avec **Add Input**. Il peut
   contenir le ZIP ou directement le dossier extrait
   `SINERT_DATA_CONCENTRATOR`.
2. Publie `pipeline.py` comme un petit dataset Kaggle privé et ajoute-le aussi
   avec **Add Input**. Le notebook doit pouvoir repérer ce fichier sous
   `/kaggle/input`; il n'a pas besoin d'accéder à ton disque Windows.
3. Importe ou copie `kaggle_phase1.ipynb` dans le Notebook, puis exécute les
   cellules dans l'ordre.
4. Télécharge les résultats de `/kaggle/working/phase1` depuis l'onglet
   **Output** du Notebook.

Le ZIP est lu directement, sans extraction sur le disque Kaggle. Si plusieurs
ZIP sont attachés, sélectionne celui voulu dans la cellule de configuration.

### Options principales

- `--meter EM_VIRTUAL` : préfixe du compteur (modifiable si les exports le
  nomment autrement).
- `--window-steps 672` : sept jours d'historique à 96 pas par jour.
- `--horizon-steps 1` : horizon par défaut d'un pas de 15 minutes. À confirmer
  avec les encadrantes; la valeur est configurable.
- `--holidays chemin.csv` : calendrier optionnel avec une colonne `date`
  (ou une première colonne de dates). Sans fichier, seuls les week-ends sont
  marqués comme non ouvrés.
- `--max-short-gap 4` : durée maximale d'un petit trou comblé par report causal
  une interpolation linéaire entre les mesures valides qui l'encadrent.

## Résultats

- `sinert_clean.parquet` : une ligne par pas de 15 minutes, avec quatre
  variables physiques (`delta_ea_kwh`, `delta_er_kvarh`, `pmax_kw`, `smax_kva`),
  sept variables calendaires, `was_imputed` et `has_missing_physical`.
- `window_index.parquet` : début/fin des entrées et cibles, origine de
  prévision et partition chronologique de chaque fenêtre valide. Les valeurs
  chevauchantes ne sont pas dupliquées, ce qui évite de produire plusieurs Go
  de données redondantes.
- `quality_report.json` : fichiers lus, doublons, trous, remises à zéro,
  valeurs de demande négatives, imputations, anomalies signalées et nombre de
  fenêtres par partition.

Pour fournir des tableaux NumPy aux modèles sans matérialiser toutes les
fenêtres en mémoire :

```python
from pathlib import Path
import pandas as pd
from pipeline import iter_window_batches

clean = pd.read_parquet(Path("data/processed/sinert_clean.parquet"))
clean = clean.set_index("timestamp") if "timestamp" in clean.columns else clean
for x, y, target_times in iter_window_batches(clean, split="train"):
    # x: (batch, 672, 11); y: (batch, horizon, 4)
    ...
```

## Règles de traitement

- La différence d'énergie n'est calculée qu'entre deux lectures consécutives.
  Après une remise à zéro détectée, la valeur courante du compteur est utilisée
  comme différence. Les différences traversant un trou restent manquantes au
  lieu d'étaler une mesure future sur le passé. Les heures `24:00:00` des
  rapports sont normalisées au minuit du jour suivant avant dédoublonnage.
- Une lacune présente dans les fichiers sources ne peut pas être reconstituée
  fidèlement sans mesure. Les valeurs restent manquantes si aucun donneur passé
  n'existe (par exemple au début de la période) ou si le trou dépasse les
  donneurs disponibles. Ces lignes ne sont pas artificiellement remplies.
- Les trous de 1 à 4 pas sont interpolés linéairement uniquement s'ils sont
  encadrés par deux mesures réelles. Les trous plus longs sont remplacés, si
  possible, par le même créneau de la veille puis d'une à quatre semaines
  auparavant. Le champ `imputation_future_boundary` garde la date de la mesure
  droite utilisée pour l'interpolation; les fenêtres excluent les historiques
  où cette mesure ne serait pas encore connue à l'origine de prévision.
- `was_imputed=True` indique qu'au moins une mesure physique de la ligne a été
  imputée. `has_missing_physical=True` indique qu'au moins une mesure physique
  reste absente. Les deux indicateurs sont indépendants : une ligne initiale
  sans données peut avoir `was_imputed=False` et `has_missing_physical=True`.
  Le rapport donne, pour chaque variable, les taux de manque et les plus longues
  périodes de valeurs manquantes avant et après imputation.
- Les valeurs négatives de demande kW/kVA deviennent manquantes. Les zéros ne
  sont pas supprimés, car ils peuvent être réels. Des anomalies robustes sont
  signalées dans le rapport mais non supprimées automatiquement afin de ne pas
  effacer des pointes valides. La détection utilise la médiane/MAD des 672 pas
  précédents, avec un seuil de repli quand la MAD est nulle; les points suspects
  et les incohérences énergie-demande sont consignés dans le rapport. Un
  compteur actif est signalé comme bloqué après quatre différences consécutives
  nulles avec une demande supérieure à 1 kW; la période et la différence de
  reprise sont alors réimputées.
- Les partitions train/validation/test sont chronologiques (70/15/15 par
  défaut) et les cibles d'une fenêtre ne traversent pas leur frontière. Les
  entrées de validation/test peuvent utiliser l'historique antérieur, comme en
  production. Une fenêtre dont la cible est imputée est exclue; les entrées
  historiques peuvent, elles, contenir des valeurs imputées.
- Les sept variables calendaires sont les sin/cos de l'heure, du jour de la
  semaine et du mois, ainsi qu'un indicateur week-end/jour férié. Les jours
  fériés nécessitent le fichier optionnel, car les dates de vacances/examens de
  SUP'COM ne sont pas fournies.

## Vérification

```powershell
python -m unittest .\test_pipeline.py
```
