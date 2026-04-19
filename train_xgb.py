import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import yaml
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import shap
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)
    
xgb_cfg = config['xgboost_training']
search = xgb_cfg['search_space']
train_file = config['preprocessing']['train_name']

# Définition de l'expérience MLflow
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Energy_Forecasting_GPU")

df = pd.read_csv(train_file) 

# Feature Engineering sur GPU
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()
cible = 'pic_journalier_consommation'


def objective(trial):
    # Démarrer un sous-run MLflow pour chaque test d'Optuna
    with mlflow.start_run(nested=True):
        
        params = {
            "objective": xgb_cfg['objective'],
            "tree_method": xgb_cfg['tree_method'],
            "device": xgb_cfg['device'],
            "n_estimators": xgb_cfg['n_estimators'],
            "early_stopping_rounds": xgb_cfg['early_stopping_rounds'],
            "max_depth": trial.suggest_int("max_depth", search['max_depth']['min'], search['max_depth']['max']),
            "learning_rate": trial.suggest_float("learning_rate", search['learning_rate']['min'], search['learning_rate']['max'], log=True),
            "subsample": trial.suggest_float("subsample", search['subsample']['min'], search['subsample']['max']),
            "colsample_bytree": trial.suggest_float("colsample_bytree", search['colsample_bytree']['min'], search['colsample_bytree']['max']),
        }
        
        mlflow.log_params(params)
        
        tscv = TimeSeriesSplit(n_splits=5, test_size=250)
        scores_rmse = []
        
        indices = np.arange(len(df))
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(indices)):
            
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            X_train, y_train = train_df.drop(columns=[cible]), train_df[cible]
            X_test, y_test = test_df.drop(columns=[cible]), test_df[cible]
            
            # Entraînement
            modele = xgb.XGBRegressor(**params)
            modele.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Stockage de la meilleure RMSE du fold
            fold_rmse = modele.best_score
            scores_rmse.append(fold_rmse)
            mlflow.log_metric(f"rmse_fold_{fold}", fold_rmse)
        
        # D. Calcul de la moyenne de la Validation Croisée
        mean_cv_rmse = float(np.mean(scores_rmse))
        mlflow.log_metric("mean_cv_rmse", mean_cv_rmse)
        
        return mean_cv_rmse

# EXÉCUTION DE L'ORCHESTRATION
if __name__ == "__main__":
    # Run principal MLflow qui englobe toute l'étude Optuna
    with mlflow.start_run(run_name="Optuna_Bayesian_Search"):
        
        print("Démarrage de l'optimisation Bayésienne sur GPU...")
        study = optuna.create_study(direction="minimize", study_name="XGB_Energy_Tuning")
        study.optimize(objective, n_trials=30)
        
        print(f"\nMeilleure RMSE trouvée : {study.best_value}")
        print(f"Meilleurs hyperparamètres : {study.best_params}")
        
        # et le sauvegarder dans MLflow (Model Registry)
        best_params = study.best_trial.params
        best_params.update({"device": "cuda", "tree_method": "hist", "n_estimators": 500})
        
        modele_final = xgb.XGBRegressor(**best_params)
        X_all, y_all = df.drop(columns=[cible]), df[cible]
        modele_final.fit(X_all, y_all, verbose=False)
        
        # Enregistrement du modèle complet en tant qu'artefact
        mlflow.xgboost.log_model(modele_final, artifact_path="best_xgboost_model", registered_model_name="Champion_XGBoost")


        print("Génération des explications SHAP...")
        
        explainer = shap.TreeExplainer(modele_final)
        
        X_recent = X_all.tail(30)
        shap_values = explainer.shap_values(X_recent)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_recent, show=False) 
        plt.tight_layout()
        
        shap_path = "xgboost_shap_summary.png"
        plt.savefig(shap_path)
        plt.close()
        
        mlflow.log_artifact(shap_path, artifact_path="explicabilite")
        
        print("Entraînement et Explicabilité terminés avec succès.")