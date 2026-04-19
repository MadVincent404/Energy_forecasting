import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import mlflow
import yaml
from sklearn.model_selection import TimeSeriesSplit
import shap
import matplotlib.pyplot as plt

#INITIALISATION ET CONFIGURATION
with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)
    
lgb_cfg = config['lightgbm_training']
search = lgb_cfg['search_space']
train_file = config['preprocessing']['train_name']


mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Energy_Forecasting_LGBM")

print("Chargement des données avec Pandas...")
df = pd.read_csv(train_file) 
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()
cible = 'pic_journalier_consommation'

# LA FONCTION OBJECTIF OPTUNA
def objective(trial):
    with mlflow.start_run(nested=True):
        
        # A. Définition des paramètres LightGBM
        params = {
            "objective": lgb_cfg['objective'],
            "n_estimators": lgb_cfg['n_estimators'],
            "random_state": 42,
            "verbosity": -1,
            
            "num_leaves": trial.suggest_int("num_leaves", search['num_leaves']['min'], search['num_leaves']['max']),
            "max_depth": trial.suggest_int("max_depth", search['max_depth']['min'], search['max_depth']['max']),
            "learning_rate": trial.suggest_float("learning_rate", search['learning_rate']['min'], search['learning_rate']['max'], log=True),
            "subsample": trial.suggest_float("subsample", search['subsample']['min'], search['subsample']['max']),
            "subsample_freq": 1, 
            "colsample_bytree": trial.suggest_float("colsample_bytree", search['colsample_bytree']['min'], search['colsample_bytree']['max']),
            "min_child_samples": trial.suggest_int("min_child_samples", search['min_child_samples']['min'], search['min_child_samples']['max']),
        }
        
        mlflow.log_params(params)
        
        # B. Validation Croisée Temporelle
        tscv = TimeSeriesSplit(n_splits=5, test_size=250)
        scores_rmse = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            
            X_train, y_train = df.iloc[train_idx].drop(columns=[cible]), df.iloc[train_idx][cible]
            X_test, y_test = df.iloc[test_idx].drop(columns=[cible]), df.iloc[test_idx][cible]
            
            # Entraînement avec la syntaxe moderne des Callbacks
            modele = lgb.LGBMRegressor(**params)
            
            # Callback pour l'Early Stopping
            early_stopping_cb = lgb.early_stopping(stopping_rounds=50, verbose=False)
            
            modele.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[early_stopping_cb]
            )
            
            fold_mse = modele.best_score_['valid_0']['l2']
            fold_rmse = np.sqrt(fold_mse)
            
            scores_rmse.append(fold_rmse)
            mlflow.log_metric(f"rmse_fold_{fold}", fold_rmse)
            
        mean_cv_rmse = float(np.mean(scores_rmse))
        mlflow.log_metric("mean_cv_rmse", mean_cv_rmse)
        
        return mean_cv_rmse

# EXÉCUTION
if __name__ == "__main__":
    with mlflow.start_run(run_name="Optuna_LightGBM"):
        print("Démarrage de l'optimisation Bayésienne LightGBM...")
        
        study = optuna.create_study(direction="minimize", study_name="LGBM_Energy_Tuning")
        study.optimize(objective, n_trials=30)
        
        print(f"\nMeilleure RMSE trouvée : {study.best_value}")
        print(f"Meilleurs hyperparamètres : {study.best_params}")
        
        # Entraînement du modèle final
        best_params = study.best_trial.params
        best_params.update({"n_estimators": 500, "random_state": 42})
        
        modele_final = lgb.LGBMRegressor(**best_params)
        X_all, y_all = df.drop(columns=[cible]), df[cible]
        modele_final.fit(X_all, y_all)
        
        # Sauvegarde du modèle dans MLflow (Notez l'utilisation de mlflow.lightgbm)
        mlflow.lightgbm.log_model(modele_final, artifact_path="best_lightgbm_model", registered_model_name="Champion_LightGBM")

        print("Génération des explications SHAP...")
        
        explainer = shap.TreeExplainer(modele_final)
        
        X_recent = X_all.tail(30)
        shap_values = explainer.shap_values(X_recent)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_recent, show=False) 
        plt.tight_layout()
        
        shap_path = "lightgbm_shap_summary.png"
        plt.savefig(shap_path)
        plt.close()
        
        mlflow.log_artifact(shap_path, artifact_path="explicabilite")
        
        print("Entraînement et Explicabilité terminés avec succès.")