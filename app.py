import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import mlflow


st.set_page_config(page_title="Audit des Modèles", layout="wide")

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
        with st.spinner('Chargement des modèles depuis MLflow...'):
            model_xgb, model_lgbm, explainer_xgb, explainer_lgbm = load_models_and_explainers()
    except Exception as e:
        st.error(f"Erreur de chargement des modèles : {e}")
        return # Pylance comprend que le script s'arrête ici en cas d'erreur

    try:
        test_file_path = Path(r"train_data\test.csv")
        X_data = pd.read_csv(test_file_path, sep=",")

        X_data['Date'] = pd.to_datetime(X_data['Date'])
        
        # 3. On met la date en Index (XGBoost ne la verra plus)
        X_data = X_data.set_index('Date').sort_index()
        
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