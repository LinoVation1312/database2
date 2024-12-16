import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Visualisation Optimisée des Courbes d'Absorption Acoustique")
st.write("Sélectionnez plusieurs échantillons pour comparer leurs courbes d'absorption.")

@st.cache_data
def load_data(file):
    """
    Chargement des données depuis Excel et normalisation stricte des colonnes.
    """
    # Charger la feuille DATA
    df = pd.read_excel(file, sheet_name="DATA", engine="openpyxl")
    
    # Normaliser les noms de colonnes
    df.columns = (
        df.columns
        .str.strip()         # Retirer les espaces autour
        .str.lower()         # Convertir en minuscules
        .str.replace(r'[^\w\s]', '', regex=True)  # Retirer les caractères spéciaux
        .str.replace(' ', '_')  # Remplacer les espaces par des underscores
    )

    # Suppression des doublons de colonnes (par exemple, sample_number_stn1)
    if "sample_number_stn1" in df.columns:
        df = df.drop(columns=["sample_number_stn1"])

    return df

uploaded_file = st.file_uploader("Chargez votre fichier Excel", type=["xlsx"])

if uploaded_file:
    # Chargement et nettoyage des données
    df = load_data(uploaded_file)

    # Colonnes nécessaires (normalisées)
    columns_to_keep = [
        "sample_number_stn", "trim_level", "project_number", "material_family",
        "material_supplier", "detailed_description", "surface_mass_gm²", 
        "thickness_mm", "assembly_type", "finished_good_surface_aera", 
        "frequency", "alpha_cabin", "alpha_kundt"
    ]

    # Vérifier les colonnes manquantes
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        st.error(f"Colonnes manquantes : {missing_columns}")
    else:
        # Filtrer uniquement les colonnes nécessaires
        df = df[columns_to_keep]

        # Sélection de plusieurs échantillons
        sample_numbers = st.sidebar.multiselect("Sélectionnez plusieurs Sample Numbers :", df["sample_number_stn"].unique())

        # Si des échantillons sont sélectionnés
        if sample_numbers:
            # Filtrage des données pour les échantillons sélectionnés
            filtered_data = df[df["sample_number_stn"].isin(sample_numbers)]

            # Choix du type d'absorption
            absorption_type = st.sidebar.radio("Type d'absorption :", ["alpha_cabin", "alpha_kundt"])

            # Affichage du graphique pour chaque échantillon sélectionné
            st.subheader(f"Comparaison des courbes d'absorption pour {', '.join(sample_numbers)} - {absorption_type}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for sample in sample_numbers:
                data_sample = filtered_data[filtered_data["sample_number_stn"] == sample]
                ax.plot(
                    data_sample["frequency"], 
                    data_sample[absorption_type], 
                    marker='o', linestyle='-', label=f"{sample}"
                )
            
            ax.set_title(f"Absorption : {absorption_type}")
            ax.set_xlabel("Fréquence (Hz)")
            ax.set_ylabel(absorption_type)
            ax.legend(title="Échantillons")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("Veuillez sélectionner au moins un échantillon pour afficher les courbes.")
