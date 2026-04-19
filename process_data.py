import pandas as pd
from pathlib import Path
import argparse
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def perform_feature_engineering(file_path, train_name, test_name, lags, windows):
    logging.info(f"Démarrage avec paramètres : Lags={lags} jours, Moyenne Mobile={windows} jours")
    
    df = pd.read_csv(file_path, sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.set_index('date').sort_index()

    df['annee'] = df.index.year
    df['mois'] = df.index.month
    df['jour'] = df.index.day

    df['jour_semaine'] = df.index.dayofweek
    df['est_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['jour_annee'] = df.index.dayofyear
    df['trimestre'] = df.index.quarter

    features = ["pic_journalier_consommation","temperature_moyenne"]

    for feature in features:
        cleaned_name = feature.replace(' (°C)', '').replace(' (MW)', '').replace(' ', '_')

        for lag in lags:
            column_name = f"{cleaned_name}_lag_{lag}"
            df[column_name] = df[feature].shift(lag)
            
        for window in windows:
            column_name = f"{cleaned_name}_roll_{window}"
            df[column_name] = df[feature].rolling(window=window).mean().shift(1)

    max_lag = max(lags)
    col_max_lag = f"pic_journalier_consommation_lag_{max_lag}"
    df = df.dropna(subset=[col_max_lag])

    nb_datas = len(df)
    nb_tail = 90
    df.head(nb_datas- nb_tail).to_csv(train_name)
    df.tail(nb_tail).to_csv(test_name)

    
if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("Le fichier params.yaml est introuvable.")
        exit(1)
        
    # 2. Extraction des paramètres spécifiques au preprocessing
    try:
        file_path = config['preprocessing']['file_path']
        train_name = config['preprocessing']['train_name']
        test_name = config['preprocessing']['test_name']
        lags = config['preprocessing']['lags_jours']
        windows = config['preprocessing']['fenetre_moyenne_mobile']

    except KeyError as e:
        logging.error(f"Paramètre manquant dans le fichier YAML : {e}")
        exit(1)

    data_path = Path(file_path)
    if  data_path.exists() == False:
        AssertionError(f"Fichier manquant: {data_path}")
    
    # 3. Lancement
    perform_feature_engineering(file_path=file_path, train_name=train_name, test_name=test_name, lags=lags, windows=windows)