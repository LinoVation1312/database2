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

    # Vérification de la présence de la colonne "surface_mass_gm2" (notez l'absence du "²")
    if "surface_mass_gm2" not in df.columns:
        st.error("La colonne 'surface_mass_gm2' est manquante dans les données.")
    else:
        # Colonnes nécessaires (normalisées)
        columns_to_keep = [
            "sample_number_stn", "trim_level", "project_number", "material_family",
            "material_supplier", "detailed_description", "surface_mass_gm2", 
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

            # Nettoyer les colonnes numériques pour éviter les erreurs
            df["surface_mass_gm2"] = pd.to_numeric(df["surface_mass_gm2"], errors='coerce')
            df["thickness_mm"] = pd.to_numeric(df["thickness_mm"], errors='coerce')

            # Sélection de critères de filtre
            trim_level = st.sidebar.selectbox("Sélectionnez un Trim Level :", ["Tous"] + list(df["trim_level"].unique()))
            supplier = st.sidebar.selectbox("Sélectionnez un Supplier :", ["Tous"] + list(df["material_supplier"].unique()))
            surface_mass = st.sidebar.slider(
                "Sélectionnez une plage de Surface Mass (g/m²) :", 
                min_value=int(df["surface_mass_gm2"].min()), 
                max_value=int(df["surface_mass_gm2"].max()), 
                value=(int(df["surface_mass_gm2"].min()), int(df["surface_mass_gm2"].max()))
            )
            thickness = st.sidebar.slider(
                "Sélectionnez une plage de Thickness (mm) :", 
                min_value=float(df["thickness_mm"].min()), 
                max_value=float(df["thickness_mm"].max()), 
                value=(float(df["thickness_mm"].min()), float(df["thickness_mm"].max()))
            )
            assembly_type = st.sidebar.selectbox("Sélectionnez un Assembly Type :", ["Tous"] + list(df["assembly_type"].unique()))

            # Appliquer les filtres sélectionnés
            filtered_df = df.copy()

            if trim_level != "Tous":
                filtered_df = filtered_df[filtered_df["trim_level"] == trim_level]
            if supplier != "Tous":
                filtered_df = filtered_df[filtered_df["material_supplier"] == supplier]
            filtered_df = filtered_df[
                (filtered_df["surface_mass_gm2"] >= surface_mass[0]) & 
                (filtered_df["surface_mass_gm2"] <= surface_mass[1])
            ]
            filtered_df = filtered_df[
                (filtered_df["thickness_mm"] >= thickness[0]) & 
                (filtered_df["thickness_mm"] <= thickness[1])
            ]
            if assembly_type != "Tous":
                filtered_df = filtered_df[filtered_df["assembly_type"] == assembly_type]

            # Sélection de plusieurs échantillons parmi les échantillons filtrés
            sample_numbers = st.sidebar.multiselect("Sélectionnez plusieurs Sample Numbers :", filtered_df["sample_number_stn"].unique())

            # Si des échantillons sont sélectionnés
            if sample_numbers:
                # Filtrage des données pour les échantillons sélectionnés
                filtered_data = filtered_df[filtered_df["sample_number_stn"].isin(sample_numbers)]

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
