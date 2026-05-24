import requests
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_incremental_energy_data():
    base_url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/pic-journalier-consommation-brute/records"
    file_path = "data/pic-journalier-consommation-brute.csv"

    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_existing["date"] = pd.to_datetime(df_existing["date"])
        last_date = df_existing["date"].max().strftime("%Y-%m-%d")
        logging.info(f"Dataset existant trouvé. Dernière mise à jour : {last_date}")
        where_clause = f"date > date'{last_date}'"
    else:
        logging.info("Aucun dataset local trouvé. Téléchargement complet.")
        df_existing = pd.DataFrame()
        where_clause = None

    new_records = []
    limit = 100

    while True:
        params = {
            "limit": limit,
            "order_by": "date ASC"
        }
        if where_clause:
            params["where"] = where_clause

        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        results = response.json().get("results", [])
        if not results:
            break

        new_records.extend(results)

        if len(results) < limit:
            break

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_new["date"] = pd.to_datetime(df_new["date"])

        if not df_existing.empty:
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final = df_final.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        df_final.to_csv(file_path, index=False)

        logging.info(f"Mise à jour réussie. Nouveau total : {len(df_final)} lignes.")
    else:
        logging.info("Aucune nouvelle donnée disponible sur le serveur.")

if __name__ == "__main__":
    fetch_incremental_energy_data()