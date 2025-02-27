################## package ########################
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import os
import joblib
import plotly.graph_objects as go
################################################
    
##### URL de l'API
API_URL = "https://projet7-1.onrender.com"

# Charger les données clients
FILE_PATH = "clients_data.csv"

##### Configuration de la page
st.set_page_config(
    page_title="Simulation de Risque de Crédit",
    page_icon="💳",
    layout="wide"
)

##### definition du menu du dashboard
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Accueil", "Prédictions", "Analyse des Caractéristiques", "Analyse Bi-Variée", "Modification des informations","Prédiction nouveau client"],
        icons=["house", "graph-up", "list-task", "bi-graph-up-arrow", "pencil-square","file-plus"],
        menu_icon="menu-button",
        default_index=0
    )

##### page d'accueil
if selected == "Accueil":
    # Titre principal avec HTML pour du style
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Bienvenue sur le Dashboard de Simulation de Risque de Crédit</h1>",
        unsafe_allow_html=True,
    )

    # Intégration de l'image
    st.image("pret_a_depense.png", use_container_width=True, caption="Prêt à dépenser - Analyse de risque de crédit")

    # Description sous l'image avec texte plus grand
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

    # Note ou footer en bas
    st.markdown(
        """
        <hr>
        <p style='text-align: center; color: gray;'>Utilisez le menu latéral pour naviguer entre les sections du tableau de bord.</p>
        """,
        unsafe_allow_html=True,
    )

############################################################################################################################# page prediction
##### Page Prédictions
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
                client_info = data.get("client_info", {})

                # SECTION 0 : Informations descriptives du client
                st.subheader("Informations descriptives du client")

                if client_info:  # Vérifiez si les informations sont disponibles
                    filtered_info = {
                        "Sexe": "Femme" if client_info.get("CODE_GENDER_F", 0) == 1 else "Homme",
                        "Âge (années)": abs(client_info.get("DAYS_BIRTH", 0)) // 365,
                        "Nombre d'enfants": client_info.get("CNT_CHILDREN", 0),
                        "Revenu annuel total (€)": f"{client_info.get('AMT_INCOME_TOTAL', 0):,.2f}",
                        "Montant du crédit (€)": f"{client_info.get('AMT_CREDIT', 0):,.2f}",
                        "Durée d'emploi (années)": abs(client_info.get("DAYS_EMPLOYED", 0)) // 365 
                                                    if client_info.get("DAYS_EMPLOYED", 0) < 0 else "Non employé"
                    }
                    filtered_info_df = pd.DataFrame(filtered_info.items(), columns=["Caractéristique", "Valeur"])
                    st.table(filtered_info_df)
                else:
                    st.warning("Les informations descriptives du client ne sont pas disponibles.")

                # SECTION 1 : Résultat de la prédiction avec jauge
                st.subheader("Résultat de la prédiction")
                optimal_threshold = 0.08

                # Afficher la jauge avec Plotly
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilité de défaut", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "green" if prediction < optimal_threshold else "red"},
                        'steps': [
                            {'range': [0, optimal_threshold], 'color': 'lightgreen'},
                            {'range': [optimal_threshold, 1], 'color': 'lightcoral'}
                        ],
                        'threshold': {
                            'line': {'color': "blue", 'width': 4},
                            'thickness': 0.75,
                            'value': optimal_threshold
                        }
                    }
                ))

                st.plotly_chart(fig)

                # SECTION 2 : Graphique des 10 principales caractéristiques locales importantes
                st.subheader("Caractéristiques locales")
                shap_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_values})
                shap_df = shap_df.sort_values(by='Importance', ascending=False)
                shap_df_top = shap_df.head(10)

                # Graphique des SHAP values
                fig, ax = plt.subplots(figsize=(10, 8))  # Taille ajustée
                sns.barplot(x='Importance', y='Feature', data=shap_df_top, palette="viridis", ax=ax)
                ax.set_title('Top 10 des caractéristiques locales importantes (SHAP values)', fontsize=14)
                ax.set_xlabel("Importance", fontsize=12)
                ax.set_ylabel("Caractéristiques", fontsize=12)

                # Ajouter des annotations
                for i, (imp, feature) in enumerate(zip(shap_df_top['Importance'], shap_df_top['Feature'])):
                    ax.text(imp, i, f'{imp:.2f}', ha='left', va='center', color='black')

                st.pyplot(fig)

                # SECTION 3 : Tableau interactif des SHAP values
                st.subheader("Tableau interactif des SHAP values")
                st.dataframe(shap_df.style.set_properties(**{'font-size': '14pt', 'padding': '5px'}), height=400)

                # SECTION 4 : Comparaison des caractéristiques locales et globales
                st.subheader("Comparaison des caractéristiques locales et globales")
                
                # Appel à l'API pour obtenir les importances globales
                global_response = requests.get(f"{API_URL}/get_global_importance")
                if global_response.status_code == 200:
                    global_data = global_response.json().get("global_importances", [])
                    global_shap_df = pd.DataFrame(global_data)

                    # Fusion des données locales et globales
                    comparison_df = shap_df_top.merge(global_shap_df, on="Feature", how="inner")
                    # Renommer les colonnes pour le graphique
                    comparison_df.rename(columns={
                        "Importance": "Caracteristiques locale client",
                        "Global Importance": "Caracteristiques globales"
                    }, inplace=True)
                    # Créer un graphique comparatif
                    fig, ax = plt.subplots(figsize=(12, 8))
                    comparison_df.plot(
                        x="Feature",
                        y=["Caracteristiques locale client", "Caracteristiques globales"],
                        kind="bar",
                        ax=ax,
                        color=["#1f77b4", "#ff7f0e"],
                        title="Comparaison des caractéristiques locales et globales"
                    )
                    ax.set_ylabel("Importance")
                    ax.set_xlabel("Caractéristiques")
                    plt.xticks(rotation=45, ha="right")

                    # Afficher le graphique
                    st.pyplot(fig)
                else:
                    st.warning("Impossible de récupérer les importances globales. Vérifiez l'API.")
              
################################################################################################################### page analyse caracteristique        
if selected == "Analyse des Caractéristiques":
    st.title("Analyse des Caractéristiques Clients")

    # Vérifier si les données globales des clients sont disponibles
    if os.path.exists(FILE_PATH):
        clients_data = pd.read_csv(FILE_PATH)

        # Liste de toutes les colonnes (features) disponibles, excluant `SK_ID_CURR`
        all_features = [col for col in clients_data.columns if col != "SK_ID_CURR"]

        # Récupérer les IDs clients via l'API
        response = requests.get(f"{API_URL}/get_client_ids")
        client_ids = response.json().get("client_ids", []) if response.status_code == 200 else []

        if client_ids:
            # Sélection de l'ID client
            selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)

            if selected_id:
                # Sélection de la caractéristique à analyser
                feature_selected = st.selectbox(
                    "Choisissez une caractéristique à explorer",
                    all_features
                )

                # Appel API pour obtenir les données du client
                response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_id})
                if response.status_code == 200:
                    data = response.json()
                    client_value = data.get("client_info", {}).get(feature_selected)

                    # Transformation des données spécifiques si nécessaire
                    if feature_selected == "DAYS_BIRTH":
                        clients_data["AGE"] = abs(clients_data["DAYS_BIRTH"]) // 365
                        client_value = abs(client_value) // 365 if client_value else None
                        feature_selected = "AGE"  # Renommer pour analyse
                    elif feature_selected == "DAYS_EMPLOYED":
                        clients_data["DAYS_EMPLOYED"] = clients_data["DAYS_EMPLOYED"].apply(
                            lambda x: abs(x) // 365 if x < 0 else None
                        )
                        client_value = abs(client_value) // 365 if client_value and client_value < 0 else "Non employé"

                    # Vérifier si la caractéristique existe dans les données
                    if feature_selected in clients_data.columns:
                        fig, ax = plt.subplots(figsize=(8, 5))  # Taille réduite
                        sns.histplot(
                            data=clients_data,
                            x=feature_selected,
                            kde=True,
                            color="#1b9e77",
                            ax=ax
                        )

                        # Ajouter une ligne verticale pour le client sélectionné
                        if client_value is not None:
                            ax.axvline(
                                x=client_value,
                                color="red",
                                linestyle="--",
                                label=f"Client sélectionné ({client_value})"
                            )

                        # Ajouter des titres et des labels
                        ax.set_title(f"Répartition de {feature_selected}", fontsize=14)
                        ax.set_xlabel(feature_selected, fontsize=12)
                        ax.set_ylabel("Fréquence", fontsize=12)
                        ax.legend()

                        # Afficher le graphique
                        st.pyplot(fig)

                        st.caption(
                            f"Ce graphique montre la répartition de la caractéristique '{feature_selected}' "
                            f"dans l'ensemble des clients. La ligne pointillée rouge représente la valeur pour le client sélectionné."
                        )
                    else:
                        st.warning(f"La caractéristique {feature_selected} n'est pas disponible dans les données des clients.")
                else:
                    st.error("Impossible de récupérer les informations du client sélectionné.")
        else:
            st.warning("Aucun client disponible. Veuillez vérifier les données.")
    else:
        st.warning("Les données globales des clients ne sont pas disponibles pour la comparaison.")
              
############################################################################################################################# page analyse bi-variée
if selected == "Analyse Bi-Variée":
    st.title("Analyse Bi-Variée")

    if os.path.exists(FILE_PATH):
        # Charger les données clients
        clients_data = pd.read_csv(FILE_PATH)
        
        # Vérifiez si les données sont bien chargées
        if clients_data.empty:
            st.warning("Le fichier de données des clients est vide ou n'a pas été chargé correctement.")
        else:
            # Liste des colonnes disponibles (excluant SK_ID_CURR)
            available_features = [col for col in clients_data.columns if col != "SK_ID_CURR"]

            # Sélection des deux features (X et Y)
            feature_x = st.selectbox("Choisissez la 1ère variable (X)", available_features)
            feature_y = st.selectbox("Choisissez la 2ème variable (Y)", available_features)

            if feature_x and feature_y:
                # Vérifiez si les colonnes sont disponibles dans les données
                if feature_x not in clients_data.columns or feature_y not in clients_data.columns:
                    st.error(f"Les colonnes '{feature_x}' ou '{feature_y}' ne sont pas présentes dans le DataFrame.")
                else:
                    # Nettoyage des colonnes X et Y
                    clients_data[feature_x] = pd.to_numeric(clients_data[feature_x], errors='coerce')
                    clients_data[feature_y] = pd.to_numeric(clients_data[feature_y], errors='coerce')

                    # Suppression des lignes avec des NaN dans X ou Y
                    filtered_data = clients_data.dropna(subset=[feature_x, feature_y])

                    if filtered_data.empty:
                        st.warning(f"Aucune donnée disponible après suppression des NaN pour les colonnes '{feature_x}' et '{feature_y}'.")
                    else:
                        # Création du graphique
                        fig, ax = plt.subplots(figsize=(6, 4))  #
                        sns.scatterplot(
                            data=filtered_data,
                            x=feature_x,
                            y=feature_y,
                            hue='default_status' if 'default_status' in filtered_data.columns else None,
                            palette='viridis',
                            ax=ax
                        )
                        ax.set_title(f'Analyse bi-variée: {feature_x} vs {feature_y}', fontsize=6)
                        ax.set_xlabel(feature_x, fontsize=6)
                        ax.set_ylabel(feature_y, fontsize=6)
                        
                        st.pyplot(fig)
                        st.caption(
                            f"Ce graphique de dispersion montre la relation entre {feature_x} et {feature_y} "
                            "pour l'ensemble des clients. Les points sont colorés en fonction de la variable 'default_status' (si disponible)."
                        )
    else:
        st.warning("Les données des clients ne sont pas disponibles.")
######################################################################################################### Page "Modification des informations"
if selected == "Modification des informations":
    st.title("Modification des informations")

    # Récupérer les IDs clients existants via l'API
    response = requests.get(f"{API_URL}/get_client_ids")
    client_ids = response.json().get("client_ids", []) if response.status_code == 200 else []

    if client_ids:
        # Sélection de l'ID client
        selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)

        # Utiliser l'endpoint /predict pour récupérer les informations du client
        if selected_id:
            response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_id})
            if response.status_code == 200:
                data = response.json()
                client_info = data.get("client_info", {})

                # Convertir les valeurs récupérées en types numériques appropriés
                current_income = float(client_info.get("AMT_INCOME_TOTAL", 0.0))
                current_credit_amount = float(client_info.get("AMT_CREDIT", 0.0))
                current_children = int(client_info.get("CNT_CHILDREN", 0))
                current_goods_price = float(client_info.get("AMT_GOODS_PRICE", 0.0))

                # Affichage des champs avec valeurs actuelles
                new_income = st.number_input("Revenus annuel total du client (€)", value=current_income, step=1000.0, min_value=0.0)
                new_credit_amount = st.number_input("Montant du crédit (€)", value=current_credit_amount, step=1000.0, min_value=0.0)
                new_children = st.number_input("Nombre d'enfants", value=current_children, step=1, min_value=0)
                new_goods_price = st.number_input("Montant des biens (€)", value=current_goods_price, step=1000.0, min_value=0.0)

                # Bouton pour mettre à jour et recalculer
                if st.button("Mettre à jour et prédire"):
                    # Préparer les données pour la requête
                    payload = {
                        "SK_ID_CURR": selected_id,
                        "AMT_INCOME_TOTAL": new_income,
                        "AMT_CREDIT": new_credit_amount,
                        "CNT_CHILDREN": new_children,
                        "AMT_GOODS_PRICE": new_goods_price
                    }

                    # Envoyer les données modifiées à l'API
                    response = requests.post(f"{API_URL}/predict_with_custom_values", json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        prediction = data.get("probability_of_default", None)

                        # Définir le seuil optimal
                        optimal_threshold = 0.08
                        st.subheader("Résultat de la prédiction")

                        # Afficher la jauge avec Plotly
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Probabilité de défaut", 'font': {'size': 24}},
                            gauge={
                                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "green" if prediction < optimal_threshold else "red"},
                                'steps': [
                                    {'range': [0, optimal_threshold], 'color': 'lightgreen'},
                                    {'range': [optimal_threshold, 1], 'color': 'lightcoral'}
                                ],
                                'threshold': {
                                    'line': {'color': "blue", 'width': 4},
                                    'thickness': 0.75,
                                    'value': optimal_threshold
                                }
                            }
                        ))

                        st.plotly_chart(fig)

                    else:
                        st.error("Erreur lors de la mise à jour ou de la prédiction.")
            else:
                st.error("Impossible de récupérer les informations du client.")
    else:
        st.warning("Aucun client disponible. Veuillez vérifier les données ou l'API.")

############################################################################################### Page "Prédiction nouveau client"
if selected == "Prédiction nouveau client":
    st.title("Prédiction nouveau client")

    # Initialiser le session_state pour stocker l'ID
    if "new_client_id" not in st.session_state:
        # Récupérer le prochain ID depuis l'API uniquement au premier chargement
        response = requests.get(f"{API_URL}/get_next_client_id")
        if response.status_code == 200:
            st.session_state.new_client_id = response.json().get("next_id")
        else:
            st.error("Erreur lors de la récupération du prochain ID client.")
            st.session_state.new_client_id = None

    # Utiliser l'ID stocké dans session_state
    new_id = st.session_state.new_client_id
    if new_id:
        st.write(f"**Nouvel ID client : {new_id}**")

        # Saisie des informations principales
        new_gender = st.selectbox("Sexe", options=["Homme", "Femme"], index=0)
        new_age = st.number_input("Âge (années)", value=30, step=1)
        new_children = st.number_input("Nombre d'enfants", value=0, step=1, min_value=0)
        new_income = st.number_input("Revenu annuel total (€)", value=0.0, step=1000.0, min_value=0.0)
        new_goods_price = st.number_input("Montant des biens (€)", value=0.0, step=1000.0, min_value=0.0)
        new_credit_amount = st.number_input("Montant du crédit (€)", value=0.0, step=1000.0, min_value=0.0)

        # Transformation du sexe pour correspondre au modèle
        code_gender_f = 1 if new_gender == "Femme" else 0
        code_gender_m = 1 if new_gender == "Homme" else 0

        # Envoyer la requête pour obtenir le score et la probabilité
        if st.button("Calculer le Score et la Probabilité"):
            # Préparer les données pour les colonnes nécessaires
            payload = {
                "SK_ID_CURR": new_id,
                "CODE_GENDER_F": code_gender_f,
                "CODE_GENDER_M": code_gender_m,
                "DAYS_BIRTH": -new_age * 365,  # Transformer l'âge en jours
                "CNT_CHILDREN": new_children,
                "AMT_INCOME_TOTAL": new_income,
                "AMT_GOODS_PRICE": new_goods_price,
                "AMT_CREDIT": new_credit_amount
            }

            # Appeler l'API pour calculer le score
            response = requests.post(f"{API_URL}/predict_new_client", json=payload)

            if response.status_code == 200:
                data = response.json()
                prediction = data.get("probability_of_default", None)
                shap_values = data.get("shap_values", [])
                feature_names = data.get("feature_names", [])

                # Définir le seuil optimal
                optimal_threshold = 0.08
                st.subheader("Résultat de la prédiction")

                # Afficher la jauge avec Plotly
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilité de défaut", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "green" if prediction < optimal_threshold else "red"},
                        'steps': [
                            {'range': [0, optimal_threshold], 'color': 'lightgreen'},
                            {'range': [optimal_threshold, 1], 'color': 'lightcoral'}
                        ],
                        'threshold': {
                            'line': {'color': "blue", 'width': 4},
                            'thickness': 0.75,
                            'value': optimal_threshold
                        }
                    }
                ))

                st.plotly_chart(fig)

                # SECTION 2 : Graphique des 10 principales caractéristiques locales importantes
                st.subheader("Caractéristiques locales importantes")
                shap_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_values})
                shap_df = shap_df.sort_values(by='Importance', ascending=False)
                shap_df_top = shap_df.head(10)

                # Graphique des SHAP values
                fig, ax = plt.subplots(figsize=(10, 8))  # Taille ajustée
                sns.barplot(x='Importance', y='Feature', data=shap_df_top, palette="viridis", ax=ax)
                ax.set_title('Top 10 des caractéristiques locales importantes (SHAP values)', fontsize=14)
                ax.set_xlabel("Importance", fontsize=12)
                ax.set_ylabel("Caractéristiques", fontsize=12)

                # Ajouter des annotations
                for i, (imp, feature) in enumerate(zip(shap_df_top['Importance'], shap_df_top['Feature'])):
                    ax.text(imp, i, f'{imp:.2f}', ha='left', va='center', color='black')

                st.pyplot(fig)

            else:
                st.error("Erreur lors du calcul de la probabilité pour le nouveau client.")
    else:
        st.warning("Impossible de générer un nouvel ID client. Vérifiez l'API.")
