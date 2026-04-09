import mlflow
from mlflow.tracking import MlflowClient
import shutil
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def export_models():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    client = MlflowClient()
    
    modeles_a_exporter = {
        "Champion_XGBoost": "deploy_xgb",
        "Champion_LightGBM": "deploy_lgbm"
    }
    
    for nom_modele, dossier_dest in modeles_a_exporter.items():
        try:
            latest_version = client.get_latest_versions(nom_modele)[-1]
            run_id = latest_version.run_id
            
            artifact_name = "best_xgboost_model" if "XGBoost" in nom_modele else "best_lightgbm_model"
            
            artifact_path = client.download_artifacts(run_id, artifact_name)
            
            if os.path.exists(dossier_dest):
                shutil.rmtree(dossier_dest)
                
            shutil.copytree(artifact_path, dossier_dest)
            logging.info(f"{nom_modele} exporté dans ./{dossier_dest}")
            
        except Exception as e:
            logging.error(f"Erreur pour {nom_modele} : {e}")

if __name__ == "__main__":
    export_models()