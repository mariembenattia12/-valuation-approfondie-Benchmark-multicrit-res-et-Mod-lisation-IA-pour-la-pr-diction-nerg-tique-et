# Analyse SINERT — phase 1

Ce dépôt contient le pipeline reproductible d'ingestion, de nettoyage,
d'imputation, d'analyse comportementale et de préparation des fenêtres
d'apprentissage pour le compteur `EM_VIRTUAL`.

## Données et confidentialité

Les exports énergétiques, le calendrier institutionnel source, le contrat et
les résultats dérivés sont des fichiers locaux et ne sont pas versionnés. Le
fichier `.gitignore` exclut les données brutes et les sorties locales afin
d'éviter leur publication accidentelle. Fournir ces fichiers localement avant
d'exécuter le pipeline.

Le calendrier quotidien conserve les dates confirmées et les événements de
confiance limitée séparément. Les estimations ne sont pas traitées comme des
faits. Les cellules inconnues restent vides; les comparaisons par semaines
d'études ou examens ne sont possibles que si ces dates sont documentées.
Consulter le rapport qualité généré localement avant d'interpréter les résultats
ou d'entraîner un modèle.

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
- `--calendar chemin.csv` ou `chemin.xlsx` : calendrier quotidien optionnel
  avec une colonne `date` et les indicateurs `is_holiday`, `is_vacation`,
  `is_study_week` et/ou `is_exam_period`. Sans fichier, seuls les week-ends
  sont marqués comme non ouvrés. `--holidays` reste accepté pour compatibilité.
- `--max-short-gap 4` : nombre maximal de pas manquants interpolés linéairement
  entre deux mesures valides. Les trous plus longs utilisent un créneau antérieur
  comparable si disponible.

Le calendrier quotidien est converti depuis le classeur fourni
`data/calendrier_complete_SUPCOM_2022-2027.xlsx`. Il couvre automatiquement
les dates du Parquet SINERT déjà préparé. Pour le régénérer puis relancer le
pipeline avec les données brutes :

```powershell
python .\pipeline.py --input .\data\SINERT_DATA_CONCENTRATOR `
  --output .\data\processed
python .\build_calendar.py --supcom-workbook `
  .\data\calendrier_complete_SUPCOM_2022-2027.xlsx
python .\pipeline.py --input .\data\SINERT_DATA_CONCENTRATOR `
  --output .\data\processed --calendar .\data\calendar_daily_features.csv
```

`calendar_events.csv` conserve toutes les lignes du classeur, y compris les
estimations. `calendar_sources.csv` retranscrit les catégories de provenance
indiquées dans sa légende; aucune URL absente du classeur n'est inventée.
Seuls les événements de confiance moyenne ou élevée alimentent les indicateurs
de calendrier des rapports et la variable `is_nonworking_day`. Les événements
estimés/hypothétiques restent visibles dans les tables événementielles, mais
ne sont pas assimilés à des faits. Les semaines d'études restent inconnues :
le classeur n'en définit aucune. Les périodes non renseignées ne sont pas
étiquetées « non-vacances ».

## Résultats

- `sinert_clean.parquet` : une ligne par pas de 15 minutes, avec quatre
  variables physiques (`delta_ea_kwh`, `delta_er_kvarh`, `pmax_kw`, `smax_kva`),
  sept variables calendaires, `was_imputed` et `has_missing_physical`.
- `window_index.parquet` : début/fin des entrées et cibles, origine de
  prévision et partition chronologique de chaque fenêtre valide. Les valeurs
  chevauchantes ne sont pas dupliquées, ce qui évite de produire plusieurs Go
  de données redondantes.
- `quality_report.json` : fichiers lus, doublons, trous, remises à zéro,
  valeurs de demande négatives, imputations, anomalies signalées, nombre de
  fenêtres par partition et disponibilité/confiance des étiquettes du
  calendrier.
- `reports/energy_daily.csv`, `energy_weekly.csv`, `energy_monthly.csv` et
  `energy_annual.csv` : énergie active estimée (mesures + imputations), énergie
  mesurée seulement, nombres de mesures observées/imputées et couverture
  réellement mesurée par période. Une période sans mesure valide reste vide,
  elle n'est pas présentée comme une consommation nulle.
- `reports/behavior_summary.csv` : minimum et heure du minimum, 5e percentile,
  maximum et heure du maximum pour chaque grandeur physique, statistiques
  globales, seuil/durée des pointes et statistiques circulaires des heures
  d'activation/désactivation (moyenne, écart-type et variance circulaires).
- `reports/behavior_by_group.csv` : moyennes, médianes, écarts-types et
  variances par jour ouvré/week-end et par saison, avec le nombre de lignes
  imputées. Les événements de vacances, examens et jours fériés sont ventilés
  seulement lorsque le calendrier les documente avec une confiance suffisante;
  les semaines d'études non renseignées ne sont pas extrapolées.
- `reports/activity_by_day.csv` : couverture, créneaux actifs, horaires
  d'activation/désactivation, durée active et statistiques journalières.
- `reports/peak_episodes.csv` : épisodes consécutifs au-dessus du 90e
  percentile de Pmax, avec début, fin exclusive, durée, Pmax/Smax et énergie.
  Une mesure manquante interrompt un épisode.
- `reports/coverage_by_month.csv`, `coverage_by_year.csv` et
  `coverage_summary.csv` : couverture mesurée, énergie observée et énergie
  incluant les imputations. Les périodes partielles sont signalées et ne
  doivent pas être comparées comme des périodes complètes.
- `reports/suspect_power_zeros.csv`, `reactive_outliers.csv`,
  `missing_imputation_quality.csv` et `missing_runs.csv` : détails des zéros
  suspects, anomalies robustes, imputations et lacunes restantes. Ces alertes
  sont conservées pour vérification; elles ne signifient pas que les mesures
  ont été automatiquement corrigées.
- `reports/window_validation.csv` : audit des fenêtres, de leurs frontières
  chronologiques, de la séparation des cibles et de la fuite potentielle
  d'information issue des imputations.
- `figures/` : graphiques SVG reproductibles de couverture mensuelle/annuelle
  et séries temporelles journalières mesurées des quatre grandeurs physiques.

Les horaires d'activation/désactivation sont estimés avec un seuil par défaut
de 1 kW. Ils décrivent le franchissement d'un seuil de puissance et ne prouvent
pas l'occupation humaine du site. Les heures moyennes sont calculées de façon
circulaire pour gérer correctement le passage de minuit.

Pour les analyses institutionnelles, fournir un CSV/XLSX quotidien avec une colonne
`date` et, selon les informations disponibles, `is_holiday`, `is_vacation`,
`is_study_week` et `is_exam_period`. Les indicateurs acceptent `true/false`, `1/0`, `yes/no` ou
`oui/non`; les dates ISO (`2024-09-02`) et jour/mois/année sont acceptées. Une
ligne par date est requise. Laisser une cellule vide lorsque l'information
n'est pas connue; ne pas coder une hypothèse en `no`. Exemple :

```csv
date,is_holiday,is_vacation,is_study_week,is_exam_period
2024-09-02,,yes,,
2024-09-03,,yes,,
```

Le classeur fourni distingue les événements confirmés par SUP'COM, les dates
ministérielles rapportées, les sources secondaires et les estimations. Les
estimations et hypothèses ne servent pas aux calculs; les URL primaires,
semaines d'études, vacances d'été et périodes d'activité complètes restent à
confirmer avec la scolarité.
Les 7 variables calendaires du modèle restent inchangées; vacances et semaines
d'études servent ici aux rapports d'analyse, pas aux entrées d'apprentissage.
Les étiquettes d'examens sont également réservées aux rapports et ne changent
pas les 11 variables du modèle. Passer le calendrier avec `--calendar`;
`--holidays` reste accepté pour compatibilité avec les anciens fichiers de
dates fériées.

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
  fériés nécessitent le fichier optionnel. Les indicateurs de vacances,
  d'examens et de semaines d'études, eux aussi optionnels, enrichissent les
  tableaux d'analyse sans modifier les 11 variables d'apprentissage.

## Vérification

```powershell
python -m unittest .\test_pipeline.py
```

## État de validation de la phase 1

L'exécution sur les exports bruts, avec le calendrier extrait du classeur
SUP'COM, est terminée. Les 7 graphiques SVG, les rapports, le Parquet propre,
l'index de fenêtres et `quality_report.json` ont été régénérés sous
`data/processed/`. Les 19 tests automatisés passent.

L'audit des fenêtres confirme les 672 pas d'historique, 11 variables,
l'horizon configuré, les frontières chronologiques, l'absence de cibles
imputées et l'absence de référence future causée par l'interpolation. La
qualité des mesures reste à valider métier : les 61 créneaux de puissance nulle
et 6 647 anomalies robustes d'énergie réactive sont signalés, non supprimés
automatiquement. Les couvertures mesurées sont d'environ 99,2 % en 2023,
99,4 % en 2024 et 91,1 % en 2025; les périodes partielles et imputées doivent
être prises en compte lors des comparaisons.

Le calendrier quotidien contient 1 342 dates sans doublon. Les catégories
incertaines restent traçables dans `calendar_events.csv`, mais ne sont pas
utilisées pour les analyses. La scolarité doit confirmer les estimations,
fournir les semaines d'études et compléter les sources primaires pour permettre
des comparaisons universitaires plus solides.
