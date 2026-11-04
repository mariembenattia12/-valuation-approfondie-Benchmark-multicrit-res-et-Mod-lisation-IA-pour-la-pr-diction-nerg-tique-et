# Pipeline de préparation SINERT — phase 1, personne A

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
