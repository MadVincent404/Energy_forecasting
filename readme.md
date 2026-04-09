# Prévision des Pics de Consommation Électrique en France

Ce projet implémente un pipeline complet de Machine Learning (MLOps) visant à modéliser et prédire l'historique journalier de la puissance maximale (GW) nécessaire pour couvrir les pics de la consommation brute française. 

Les modèles s'appuient sur l'historique de consommation ainsi que sur les données météorologiques (températures moyennes et de référence).

## Source des Données

Les données sont extraites de l'API OpenDataSoft et sont consolidées par RTE et Météo-France.

* **Identifiant du jeu de données :** `pic-journalier-consommation-brute`
* **Producteurs :** RTE, METEO-FRANCE
* **Thèmes :** Consommation, Météorologie
* **Maille géographique :** National (France)
* **Pas temporel :** Journalier
* **Profondeur d'historique :** 2012 à aujourd'hui (Mois M-1)
* **Fréquence de mise à jour :** Mensuelle
* **Licence :** Licence Ouverte v2.0 (Etalab)

## Architecture & Technologies

Ce projet respecte les standards MLOps de l'industrie pour assurer la reproductibilité, le suivi des expérimentations et l'explicabilité des modèles.

* **Modélisation :** XGBoost, LightGBM (Scikit-Learn API).
* **Optimisation :** Optuna (Recherche bayésienne d'hyperparamètres).
* **Data Version Control :** DVC (Gestion de versions des datasets et pipelines).
* **Experiment Tracking :** MLflow (Enregistrement des métriques, paramètres et modèles).
* **Explicabilité :** SHAP (Valeurs de Shapley pour l'audit des modèles).
* **Interface d'Audit :** Streamlit.

## Environnement Matériel (Hardware)

Le pipeline d'entraînement a été conçu et optimisé pour tirer parti de l'accélération matérielle locale :
* **CPU :** AMD Ryzen 5 5600
* **GPU :** NVIDIA GeForce RTX 4060 (Utilisé pour l'accélération de l'entraînement XGBoost via `device='cuda'`).

## Installation et Exécution

1. Installer les dépendances :
   pip install -r requirements.txt

2. Lancer la récupération et le traitement des données (DVC) :
   dvc repro

3. Lancer les entraînements :
   python src/train_xgb.py
   python src/train_lgbm.py

4. Lancer l'interface d'audit :
   streamlit run app.py