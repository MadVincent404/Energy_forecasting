from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    'energy_forecasting_pipeline', 
    start_date=datetime(2023, 1, 1), 
    schedule_interval='@weekly',
    catchup=False
) as dag:
    
    # 1. Mise à jour des données (API)
    fetch_data = BashOperator(
        task_id='fetch_energy_data',
        bash_command='python fetch_api.py'
    )

    # 2. Entraînement complet :DVC gère le preprocessing + XGBoost + LightGBM
    run_dvc_pipeline = BashOperator(
        task_id='run_dvc_repro',
        bash_command='dvc repro'
    )
    
    # 3. Exportation des meilleurs modèles hors de la base MLflow
    export_models = BashOperator(
        task_id='export_best_models',
        bash_command='python export_model.py'
    )

    # 4. Le Push automatique vers GitHub 
    deploy_to_github = BashOperator(
        task_id='git_push_to_production',
        bash_command="""
        git add deploy_xgb/ deploy_lgbm/ app.py
        git commit -m "Déploiement automatique : Mise à jour hebdomadaire des modèles"
        git push origin main
        """
    )

    fetch_data >> run_dvc_pipeline >> export_models >> deploy_to_github