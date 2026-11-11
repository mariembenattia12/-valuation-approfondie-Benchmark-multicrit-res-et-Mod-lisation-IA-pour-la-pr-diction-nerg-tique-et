# Interprétation de l'analyse énergétique — personne B

## 1. Périmètre de l'étude

L'analyse porte sur le compteur global `EM_VIRTUAL`, à partir du fichier propre
produit par le pipeline de préparation. La période couvre le 1er juin 2022 au
1er février 2026, avec un pas régulier de 15 minutes.

Le fichier contient 128 737 intervalles. Environ 2,24 % des valeurs physiques
restent manquantes après imputation et 3 427 lignes contiennent au moins une
valeur imputée. Les valeurs manquantes restantes n'ont pas été remplacées par
zéro ; elles sont donc prises en compte avec prudence dans les agrégations.

## 2. Profil journalier

Le profil moyen atteint son maximum d'énergie active à 11 h 15, avec
18,82 kWh par intervalle. Les maxima moyens de puissance active et apparente
apparaissent à 11 h 30, avec respectivement 75,21 kW et 78,93 kVA.

Le creux moyen se situe entre 4 h 30 et 5 h 15. La puissance reste néanmoins
positive : le site possède une charge de fond et ne s'arrête pas complètement
pendant la nuit.

## 3. Énergie consommée

L'énergie active `delta_ea_kwh` a été agrégée par jour, semaine, mois et année.
Une valeur `<NA>` signifie qu'aucune mesure valide n'était disponible ; elle ne
correspond pas à une consommation nulle.

Les totaux annuels observés sont :

| Année | Énergie active |
|---|---:|
| 2022 | 347 169,3 kWh, période partielle |
| 2023 | 434 651,8 kWh |
| 2024 | 517 674,8 kWh |
| 2025 | 545 959,9 kWh |
| 2026 | 32 486,0 kWh, période partielle |

Les années 2022 et 2026 ne sont pas complètes. Les comparaisons annuelles
doivent donc porter prioritairement sur 2023, 2024 et 2025.

## 4. Charge de base et pointes

Le 5e percentile de `pmax_kw` est utilisé comme estimation robuste de la
charge de base. Le 90e percentile est utilisé comme seuil de pointe : les
10 % des créneaux les plus élevés sont considérés comme des créneaux de pointe.

Un créneau représente 15 minutes. Un épisode de pointe est une suite de
créneaux de pointe consécutifs. Le fichier `peak_episodes.csv` détaille le
début, la fin, la puissance maximale, le nombre de créneaux et la durée de
chaque épisode.

La puissance maximale observée est de 290,25 kW et la puissance apparente
maximale est de 301,17 kVA. Le maximum de puissance active apparaît le
1er septembre 2025 à 14 h 30.

## 5. Jours ouvrés et jours non ouvrés

Les jours ouvrés présentent une puissance active moyenne de 63,47 kW, contre
50,07 kW pour les jours non ouvrés. L'énergie moyenne par créneau est
également plus élevée les jours ouvrés :

- jours ouvrés : 15,88 kWh par créneau ;
- jours non ouvrés : 12,52 kWh par créneau.

La variabilité est plus forte les jours ouvrés, avec un écart-type de 37,64 kW
contre 30,50 kW les jours non ouvrés. Cela est cohérent avec une activité plus
importante et des pointes plus fréquentes pendant les jours ouvrés.

Cette comparaison distingue les week-ends et les jours fériés fournis au
pipeline. Les vacances et les semaines d'études n'ont pas été comparées, car
un calendrier officiel n'était pas disponible.

## 6. Saisons

Les moyennes par créneau de 15 minutes sont :

| Saison | Énergie moyenne | Pmax moyenne | Smax moyenne |
|---|---:|---:|---:|
| Été | 20,87 kWh | 83,41 kW | 87,69 kVA |
| Automne | 18,46 kWh | 73,83 kW | 77,24 kVA |
| Hiver | 9,59 kWh | 38,33 kW | 40,94 kVA |
| Printemps | 9,41 kWh | 37,62 kW | 41,40 kVA |

L'été est la saison la plus consommatrice en moyenne, suivi de l'automne.
L'hiver et le printemps présentent les niveaux les plus faibles. L'utilisation
plus importante de la climatisation est une hypothèse possible pour expliquer
le niveau estival, mais elle devrait être confirmée par des données de
température ou d'occupation.

## 7. Activation et niveaux d'activité

Avec un seuil de 1 kW, 99,95 % des créneaux sont actifs. Le seuil est donc
adapté pour mesurer la présence d'une charge minimale, mais pas pour détecter
les horaires d'occupation humaine. L'activité selon ce seuil couvre presque
toute la journée, avec une durée moyenne de 23,97 heures par jour.

Des seuils complémentaires donnent une lecture plus utile des niveaux
d'activité :

| Seuil | Créneaux au-dessus du seuil |
|---|---:|
| 1 kW | 99,95 % |
| 10 kW | 99,94 % |
| 30 kW | 79,38 % |
| 50 kW | 48,81 % |
| 70 kW | 30,96 % |

Le site possède donc une charge de fond presque permanente, avec des périodes
d'activité forte identifiables aux seuils de 30, 50 et 70 kW. Ces seuils ne
doivent pas être interprétés comme des horaires exacts d'ouverture sans
information complémentaire sur l'occupation du site.

## Conclusion

Le site fonctionne avec une charge de fond continue et une activité plus forte
autour de la fin de matinée. Les jours ouvrés sont globalement plus
consommateurs et plus variables que les jours non ouvrés. L'été présente la
puissance et l'énergie moyennes les plus élevées. Les résultats sont
exploitables pour décrire le comportement énergétique, avec deux limites
principales : les périodes incomplètes au début et à la fin de la série, et
l'absence de calendrier officiel des vacances et semaines d'études.
