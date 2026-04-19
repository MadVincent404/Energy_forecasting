import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import mlflow
import subprocess
import yaml
import requests

from process_data import perform_feature_engineering

st.set_page_config(page_title="Audit des Modèles", layout="wide")
BASE_DIR = Path(__file__).parent

@st.cache_data(ttl=3600)  # re-fetch max 1x par heure
def load_test_data():
    with open(BASE_DIR / "params.yaml", "r") as f:
        config = yaml.safe_load(f)

    lags = config["preprocessing"]["lags_jours"]
    windows = config["preprocessing"]["fenetre_moyenne_mobile"]

    # 1. Télécharger les données brutes depuis l'API publique
    BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/pic-journalier-consommation-brute/records"
    limit, offset, records = 100, 0, []
    while True:
        r = requests.get(BASE_URL, params={"limit": limit, "offset": offset})
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        records.extend(batch)
        offset += limit

    df_raw = pd.DataFrame(records)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.set_index("date").sort_index()

    # Sauvegarder temporairement pour réutiliser perform_feature_engineering
    tmp_raw = BASE_DIR / "data_tmp_raw.csv"
    tmp_train = BASE_DIR / "data_tmp_train.csv"
    tmp_test = BASE_DIR / "data_tmp_test.csv"
    df_raw.reset_index().to_csv(tmp_raw, index=False)

    perform_feature_engineering(
        filepath=str(tmp_raw),
        trainname=str(tmp_train),
        testname=str(tmp_test),
        lags=lags,
        windows=windows
    )

    df_test = pd.read_csv(tmp_test, sep=",")

    # Nettoyage des fichiers temporaires
    for f in [tmp_raw, tmp_train, tmp_test]:
        f.unlink(missing_ok=True)

    return df_test

@st.cache_resource
def pull_dvc_data():
    base_dir = Path(__file__).parent
    result = subprocess.run(
        ["dvc", "pull", "train_data/test.csv"],
        cwd=base_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc pull échoué : {result.stderr}")

@st.cache_resource 
def load_models_and_explainers():
    path_xgb = "deploy_xgb"
    path_lgbm = "deploy_lgbm"
    
    model_xgb = mlflow.xgboost.load_model(path_xgb)
    model_lgbm = mlflow.lightgbm.load_model(path_lgbm)
    
    explainer_xgb = shap.TreeExplainer(model_xgb)
    explainer_lgbm = shap.TreeExplainer(model_lgbm)
    
    return model_xgb, model_lgbm, explainer_xgb, explainer_lgbm


def main():
    st.title("Audit Éducatif : XGBoost vs LightGBM")

    try:
        with st.spinner("Récupération des données depuis l'API OpenDataSoft..."):
            X_data = load_test_data()
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        return
    
    base_dir = Path(__file__).parent
    with open(base_dir / "params.yaml", "r") as f:
        config = yaml.safe_load(f)

    testfilepath = base_dir / config["preprocessing"]["testname"]

    try:
        with st.spinner('Chargement des modèles depuis MLflow...'):
            model_xgb, model_lgbm, explainer_xgb, explainer_lgbm = load_models_and_explainers()
    except Exception as e:
        st.error(f"Erreur de chargement des modèles : {e}")
        return # Pylance comprend que le script s'arrête ici en cas d'erreur

    try:
        test_file_path = Path("train_data/test.csv")
        X_data = pd.read_csv(test_file_path, sep=",")

        X_data['date'] = pd.to_datetime(X_data['date'])
        
        # On met la date en Index (XGBoost ne la verra plus)
        X_data = X_data.set_index('date').sort_index()
        
        cible = 'Pic journalier consommation (MW)'
        
        # On sauvegarde les vraies valeurs pour le graphique
        y_true = X_data[cible] 
        
        # On supprime la colonne pour que XGBoost ait exactement les 48 colonnes attendues
        X_data = X_data.drop(columns=[cible])

    except FileNotFoundError as e:
        st.error(f"Erreur de chargement du CSV : {e}")
        return


    st.subheader("Historique des prédictions sur 90 jours")

    preds_xgb_90 = model_xgb.predict(X_data)
    preds_lgbm_90 = model_lgbm.predict(X_data)

    df_graph = pd.DataFrame({
        "Valeur Réelle": y_true,
        "XGBoost": preds_xgb_90,
        "LightGBM": preds_lgbm_90
    }, index=X_data.index)

    st.line_chart(df_graph, height=350)

    st.markdown("---")
    st.subheader("Analyse détaillée du dernier jour")

    last_day_data = X_data.iloc[[-1]]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Modèle : XGBoost**")
        pred_xgb = preds_xgb_90[-1]
        st.metric(label="Prédiction XGBoost", value=f"{int(pred_xgb):,} MW".replace(',', ' '))
        
        shap_val_xgb = explainer_xgb.shap_values(last_day_data)
        fig_xgb, ax_xgb = plt.subplots(figsize=(6, 4))
        exp_xgb = shap.Explanation(
            values=shap_val_xgb[0], 
            base_values=explainer_xgb.expected_value, 
            data=last_day_data.iloc[0], 
            feature_names=last_day_data.columns
        )
        shap.plots.waterfall(exp_xgb, show=False)
        st.pyplot(fig_xgb)

    with col2:
        st.markdown("**Modèle : LightGBM**")
        pred_lgbm = preds_lgbm_90[-1]
        st.metric(label="Prédiction LightGBM", value=f"{int(pred_lgbm):,} MW".replace(',', ' '))
        
        shap_val_lgbm = explainer_lgbm.shap_values(last_day_data)
        fig_lgbm, ax_lgbm = plt.subplots(figsize=(6, 4))
        exp_lgbm = shap.Explanation(
            values=shap_val_lgbm[0], 
            base_values=explainer_lgbm.expected_value, 
            data=last_day_data.iloc[0], 
            feature_names=last_day_data.columns
        )
        shap.plots.waterfall(exp_lgbm, show=False)
        st.pyplot(fig_lgbm)


if __name__ == "__main__":
    main()