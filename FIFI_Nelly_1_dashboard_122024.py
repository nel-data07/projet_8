import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# URL de l'API
API_URL = "https://projet7-1.onrender.com"

# Configuration de la page
st.set_page_config(
    page_title="Simulation de Risque de Crédit",
    page_icon="💳",
    layout="wide"
)

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Accueil", "Prédictions", "Analyse des Caractéristiques", "Analyse Bi-Variée", "Modification des informations"],
        icons=["house", "graph-up", "list-task", "scatter-chart", "pencil-square"],
        menu_icon="menu-button",
        default_index=0
    )

if selected == "Prédictions":
    st.title("Prédictions pour un Client Existant")

    response = requests.get(f"{API_URL}/get_client_ids")
    client_ids = response.json().get("client_ids", []) if response.status_code == 200 else []

    if client_ids:
        selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)
        if st.button("Obtenir la prédiction"):
            response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_id})
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("probability_of_default", None)
                shap_values = data.get("shap_values", [])
                feature_names = data.get("feature_names", [])
                
                optimal_threshold = 0.08
                if prediction > optimal_threshold:
                    st.error(f"Crédit REFUSÉ (Probabilité de défaut : {prediction:.2f})")
                else:
                    st.success(f"Crédit ACCEPTÉ (Probabilité de défaut : {prediction:.2f})")

                shap_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_values})
                shap_df = shap_df.sort_values(by='Importance', ascending=False)

                # Graphique des SHAP values avec annotations
                fig, ax = plt.subplots()
                sns.barplot(x='Importance', y='Feature', data=shap_df, palette="viridis", ax=ax)
                ax.set_title('Importance des caractéristiques locales (SHAP values)', fontsize=14)
                
                for i, (imp, feature) in enumerate(zip(shap_df['Importance'], shap_df['Feature'])):
                    ax.text(imp, i, f'{imp:.2f}', ha='left', va='center', color='black')

                st.pyplot(fig)
                st.caption("Ce graphique montre l'importance des caractéristiques locales (SHAP values) pour la prédiction de ce client. Plus la valeur est élevée, plus la feature a influencé la prédiction.")
        
if selected == "Analyse des Caractéristiques":
    st.title("Analyse des Caractéristiques Clients")

    feature_selected = st.selectbox("Choisissez une variable à comparer", ['age', 'income', 'loan_amount', 'credit_score'])

    if feature_selected:
        response = requests.get(f"{API_URL}/get_client_ids")
        client_ids = response.json().get("client_ids", []) if response.status_code == 200 else []
        
        if client_ids:
            selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)
            client_data = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_id}).json()
            feature_value = client_data.get('client_data', {}).get(feature_selected, None)
            
            st.write(f"Valeur pour le client sélectionné : {feature_value}")
            
            # Histogramme des valeurs de la variable
            fig, ax = plt.subplots()
            sns.histplot(data=clients_data[feature_selected], kde=True, color='#1b9e77', ax=ax)
            ax.axvline(x=feature_value, color='#d95f02', linestyle='--')
            ax.set_title(f'Répartition de la variable {feature_selected}', fontsize=14)
            
            st.pyplot(fig)
            st.caption(f"Ce graphique montre la répartition de la variable {feature_selected} dans l'ensemble des clients. La ligne en pointillés rouges représente la valeur de la variable pour le client sélectionné.")

if selected == "Analyse Bi-Variée":
    st.title("Analyse Bi-Variée")

    feature_x = st.selectbox("Choisissez la 1ère variable", ['age', 'income', 'loan_amount'])
    feature_y = st.selectbox("Choisissez la 2ème variable", ['credit_score', 'debt_ratio', 'installment'])

    if feature_x and feature_y:
        fig, ax = plt.subplots()
        sns.scatterplot(data=clients_data, x=feature_x, y=feature_y, hue='default_status', palette='viridis', ax=ax)
        ax.set_title(f'Analyse bi-variée: {feature_x} vs {feature_y}', fontsize=14)
        
        st.pyplot(fig)
        st.caption(f"Ce graphique de dispersion montre la relation entre {feature_x} et {feature_y} pour l'ensemble des clients. Les points sont colorés en fonction de la variable 'default_status'.")

if selected == "Modification des informations":
    st.title("Modification des Informations Client")

    response = requests.get(f"{API_URL}/get_client_ids")
    client_ids = response.json().get("client_ids", []) if response.status_code == 200 else []
    
    if client_ids:
        selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)
        
        new_income = st.number_input("Revenus du client", value=0)
        new_loan_amount = st.number_input("Montant du prêt", value=0)

        if st.button("Mettre à jour et prédire"):
            payload = {
                "SK_ID_CURR": selected_id,
                "income": new_income,
                "loan_amount": new_loan_amount
            }
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("probability_of_default", None)
                if prediction > 0.08:
                    st.error(f"Crédit REFUSÉ (Probabilité de défaut : {prediction:.2f})")
                else:
                    st.success(f"Crédit ACCEPTÉ (Probabilité de défaut : {prediction:.2f})")
