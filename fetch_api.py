import requests
import pandas as pd
import time
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_incremental_energy_data():
    BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/pic-journalier-consommation-brute/records"
    FILE_PATH = "data/pic-journalier-consommation-brute.csv"

    if os.path.exists(FILE_PATH):
        df_existing = pd.read_csv(FILE_PATH, parse_dates=['date'], index_col='date')
        
        last_date = df_existing.index.max().strftime('%Y-%m-%d')
        logging.info(f"Dataset existant trouvé. Dernière mise à jour : {last_date}")
        
        where_clause = f"date > '{last_date}'"
    else:
        logging.info("Aucun dataset local trouvé. Déclenchement d'un téléchargement complet (Initial Load).")
        df_existing = pd.DataFrame()
        where_clause = None


    limit = 100
    offset = 0 
    new_records = []
    has_more_data = True

    while has_more_data:
        params = {
            "limit": limit,
            "offset": offset
        }

        if where_clause:
            params['where'] = where_clause
        
        reponse = requests.get(BASE_URL, params)

        if reponse.status_code != 200:
            logging.error(f"Erreur API {reponse.status_code}: {reponse.text}")
            raise Exception("L'appel api a échoué")

        data = reponse.json()
        results = data.get("results", [])

        if not results:
            has_more_data = False

        else:
            new_records.extend(results)
            offset +=limit
            time.sleep(0.5)
        
    if new_records:
        logging.info(f"{len(new_records)} nouvelles lignes récupérées depuis l'API")
    
        df_new = pd.DataFrame(new_records)
        df_new["date"] = pd.to_datetime(df_new['date'])
        df_new = df_new.set_index('date')

        df_final = pd.concat([df_existing, df_new])
            
        df_final = df_final[~df_final.index.duplicated(keep='last')].sort_index()
        
        df_final.to_csv(FILE_PATH)
        logging.info(f"Mise à jour réussie. Nouveau total : {len(df_final)} lignes.")
    else:
        logging.info("Aucune nouvelle donnée disponible sur le serveur. Le dataset est déjà à jour.")

if __name__ == "__main__":
    fetch_incremental_energy_data()