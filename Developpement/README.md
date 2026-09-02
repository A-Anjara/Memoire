# Système Intelligent de Rizipisciculture — INSI

Application Streamlit implémentant les 3 modules décrits dans le mémoire :

- **Module 1** — Régression : densité de tilapia à partir des caractéristiques
  du sol (Régression Linéaire, Arbre de Décision, Random Forest, SVR, XGBoost).
- **Module 2** — Classification : état actuel de la qualité de l'eau (bonne /
  mauvaise), avec ré-équilibrage des classes.
- **Module 3** — Anticipation à H+1 : décomposition STL (tendance + saisonnalité)
  puis LSTM sur la tendance, un modèle par variable (PH, TEMP, DO, TURBIDITY).

Tous les modèles sont optimisés par **GridSearchCV** (grilles réduites par
rapport au mémoire pour rester rapide dans une app interactive, mais mêmes
familles de modèles et même logique).

## ⚠️ À propos des données

Aucun jeu de données réel n'a été fourni. Les 3 modules s'entraînent donc sur
des **données synthétiques**, générées dans `data_gen.py` pour reproduire les
distributions et tendances décrites dans le mémoire (violin plots, histogrammes,
corrélations, déséquilibre 73%/27%, etc.). Pour utiliser vos vraies données,
remplacez les fonctions `generate_module1_data()`, `generate_water_quality_dataset()`
et `generate_hourly_series()` par du code de chargement (`pd.read_csv(...)`).

## Base de données SQLite — rôle unique et immuable

`seasonality.db` contient une seule table `seasonality` :

| Colonne | Description |
|---|---|
| date | jour de l'année (1-366) |
| heure | heure (0-23) |
| s_temp | composante saisonnière S_t — Température |
| s_do | composante saisonnière S_t — Oxygène dissous |
| s_ph | composante saisonnière S_t — pH |
| s_turbidity | composante saisonnière S_t — Turbidité |

Cette table est **peuplée une seule fois** (au premier clic sur "Décomposer STL"
dans le Module 3), à partir de la décomposition STL (période journalière = 24h)
appliquée à une série temporelle horaire simulée sur un an. Elle n'est **ni
modifiable, ni affichée** dans l'interface : elle sert uniquement, en interne,
de table de lookup en lecture seule pour reconstituer la prédiction finale du
Module 3 :

```
valeur_finale(H+1) = tendance_prédite_par_LSTM + S_t(jour_H+1, heure_H+1)
```

Voir `db.py` (fonctions `init_db`, `populate_from_stl`, `lookup` — pas de
fonction `update`).

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Structure du projet

```
app.py          # interface Streamlit (3 onglets, un par module)
db.py           # base SQLite immuable pour S_t (Module 3 uniquement)
data_gen.py     # générateurs de données synthétiques (3 modules)
module1.py      # entraînement + prédiction — densité de tilapia
module2.py      # entraînement + prédiction — qualité actuelle de l'eau
module3.py      # décomposition STL + entraînement/prédiction LSTM
requirements.txt
```

## Utilisation

Dans chaque onglet, cliquez d'abord sur le bouton d'entraînement (les modèles
sont gardés en mémoire de session le temps de la session Streamlit, pas
ré-entraînés à chaque interaction). Le tableau de métriques apparaît, suivi
d'un formulaire de prédiction.

Pour le Module 3, un premier entraînement peut prendre 1 à 2 minutes (4 modèles
LSTM, un par variable). La table `seasonality.db` n'est calculée qu'une seule
fois, au tout premier entraînement.
