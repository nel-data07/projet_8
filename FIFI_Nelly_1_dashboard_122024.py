################## package ########################
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
################################################

##### URL de l'API
API_URL = "https://projet7-1.onrender.com"

##### Configuration de la page
st.set_page_config(
    page_title="Simulation de Risque de Crédit",
    page_icon="💳",
    layout="wide"
)

##### Chargement des données des clients #####
CLIENTS_DATA_PATH = "clients_data.csv"  # Assurez-vous que ce fichier est dans le même dossier que app.py

try:
    clients_data = pd.read_csv(CLIENTS_DATA_PATH)
    st.success(f"Données clients chargées ({len(clients_data)} lignes).")
except FileNotFoundError:
    st.error("Le fichier 'clients_data.csv' est introuvable. Assurez-vous qu'il est dans le dossier contenant app.py.")
    clients_data = pd.DataFrame()  # DataFrame vide pour éviter des erreurs

##### Définition du menu du dashboard
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Accueil", "Prédictions", "Analyse des Caractéristiques", "Analyse Bi-Variée", "Modification des informations"],
        icons=["house", "graph-up", "list-task", "scatter-chart", "pencil-square"],
        menu_icon="menu-button",
        default_index=0
    )

##### Page d'accueil
if selected == "Accueil":
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Bienvenue sur le Dashboard de Simulation de Risque de Crédit</h1>",
        unsafe_allow_html=True,
    )
    st.image("pret_a_depense.png", use_column_width=True, caption="Prêt à dépenser - Analyse de risque de crédit")
    st.markdown(
        """
        <h2 style='text-align: center;'>Ce tableau de bord vous permet de :</h2>
        <ul style='font-size: 20px;'>
            <li><b>Visualiser</b> les prédictions de risque de crédit pour chaque client.</li>
            <li><b>Comparer</b> les caractéristiques d'un client à l'ensemble de la population.</li>
            <li><b>Explorer</b> les relations entre différentes variables.</li>
            <li><b>Modifier</b> les informations client pour recalculer les scores en temps réel.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <hr>
        <p style='text-align: center; color: gray;'>Utilisez le menu latéral pour naviguer entre les sections du tableau de bord.</p>
        """,
        unsafe_allow_html=True,
    )

##### Page Prédictions
if selected == "Prédictions":
    st.title("Prédictions pour un Client Existant")

    if not clients_data.empty and "SK_ID_CURR" in clients_data.columns:
        # Obtenez tous les IDs clients pour le menu déroulant
        client_ids = clients_data["SK_ID_CURR"].tolist()
        selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)

        if st.button("Obtenir la prédiction"):
            response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_id})
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("probability_of_default", None)
                shap_values = data.get("shap_values", [])
                feature_names = data.get("feature_names", [])
                client_info = clients_data[clients_data["SK_ID_CURR"] == selected_id]

                # SECTION 1 : Informations descriptives du client
                st.subheader("Informations descriptives du client")
                if not client_info.empty:
                    st.table(client_info)
                else:
                    st.warning("Aucune information disponible pour ce client.")

                # SECTION 2 : Résultat de la prédiction
                st.subheader("Résultat de la prédiction")
                optimal_threshold = 0.08
                if prediction > optimal_threshold:
                    st.error(f"Crédit REFUSÉ (Probabilité de défaut : {prediction:.2f})")
                else:
                    st.success(f"Crédit ACCEPTÉ (Probabilité de défaut : {prediction:.2f})")
                st.markdown(f"**Seuil utilisé pour la décision : {optimal_threshold:.2f}**")

                # SECTION 3 : Graphique des 10 principales caractéristiques locales importantes
                st.subheader("Top 10 des caractéristiques locales importantes")
                shap_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_values})
                shap_df = shap_df.sort_values(by='Importance', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=shap_df, palette="viridis", ax=ax)
                ax.set_title('Top 10 des caractéristiques locales importantes (SHAP values)', fontsize=14)
                st.pyplot(fig)

                # SECTION 4 : Tableau interactif des SHAP values
                st.subheader("Tableau interactif des SHAP values")
                st.dataframe(shap_df.style.set_properties(**{'font-size': '14pt', 'padding': '5px'}))

                # SECTION 5 : Tableau interactif des données associées au client
                st.subheader("Données associées au client")
                st.dataframe(client_info.transpose().reset_index().rename(columns={'index': 'Feature', 0: 'Value'}))
            else:
                st.error("Erreur lors de l'appel API pour la prédiction.")
    else:
        st.error("Les données des clients sont introuvables ou incomplètes.")

##### page analyse caracteristique        
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

##### page analyse bi-variée
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

##### page modification des informations
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
