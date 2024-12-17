import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import numpy as np

# Titre de l'application
st.title("Visualisation Optimisée des Courbes d'Absorption Acoustique")
st.write("Sélectionnez plusieurs échantillons pour comparer leurs courbes d'absorption.")

@st.cache_data
def load_data(file):
    """
    Chargement des données depuis Excel et normalisation stricte des colonnes.
    """
    # Charger la feuille DATA (avec ou sans espace à la fin)
    sheet_names = pd.ExcelFile(file, engine="openpyxl").sheet_names
    data_sheet = next((sheet for sheet in sheet_names if sheet.strip().lower() == 'data'), None)
    
    if data_sheet is None:
        st.error("Aucune feuille de données 'DATA' trouvée dans le fichier.")
        return None
    
    df = pd.read_excel(file, sheet_name=data_sheet, engine="openpyxl")
    
    # Normaliser les noms de colonnes
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(' ', '_')
    )

    # Suppression des doublons de colonnes
    if "sample_number_stn1" in df.columns:
        df = df.drop(columns=["sample_number_stn1"])

    return df

# Téléchargement du fichier Excel
uploaded_file = st.file_uploader("Chargez votre fichier Excel", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        # Sélection de la colonne de masse surfacique
        surface_mass_column = next((col for col in df.columns if 'surface_mass' in col), None)

        # Colonnes nécessaires pour le traitement
        columns_to_keep = [
            "sample_number_stn", "trim_level", "project_number", "material_family",
            "material_supplier", "detailed_description", surface_mass_column, 
            "thickness_mm", "assembly_type", "frequency", "alpha_cabin", "alpha_kundt"
        ]

        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            st.error(f"Colonnes manquantes : {missing_columns}")
        else:
            df = df[columns_to_keep]

            # Sélection des filtres
            trim_level = st.sidebar.selectbox("Sélectionnez un Trim Level :", ["Tous"] + list(df["trim_level"].unique()))
            supplier = st.sidebar.selectbox("Sélectionnez un Supplier :", ["Tous"] + list(df["material_supplier"].unique()))
            surface_mass = st.sidebar.slider(
                "Sélectionnez une plage de Surface Mass (g/m²) :",
                int(df[surface_mass_column].min()), 
                int(df[surface_mass_column].max()), 
                (int(df[surface_mass_column].min()), int(df[surface_mass_column].max()))
            )
            thickness = st.sidebar.slider(
                "Sélectionnez une plage de Thickness (mm) :",
                float(df["thickness_mm"].min()), 
                float(df["thickness_mm"].max()), 
                (float(df["thickness_mm"].min()), float(df["thickness_mm"].max()))
            )

            # Appliquer les filtres
            filtered_df = df[
                (df[surface_mass_column] >= surface_mass[0]) & 
                (df[surface_mass_column] <= surface_mass[1]) &
                (df["thickness_mm"] >= thickness[0]) & 
                (df["thickness_mm"] <= thickness[1])
            ]

            # Sélection des échantillons
            sample_numbers = st.sidebar.multiselect("Sélectionnez plusieurs Sample Numbers :", filtered_df["sample_number_stn"].unique())

            if sample_numbers:
                filtered_data = filtered_df[filtered_df["sample_number_stn"].isin(sample_numbers)]

                absorption_type = st.sidebar.radio("Type d'absorption :", ["alpha_cabin", "alpha_kundt"])

                # Fréquences définies pour l'axe X
                freq_ticks = [315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000] if absorption_type == "alpha_cabin" else [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300]

                # Graphique
                st.subheader(f"Comparaison des courbes d'absorption pour {absorption_type}")
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for sample in sample_numbers:
                    sample_data = filtered_data[filtered_data["sample_number_stn"] == sample]
                    ax.plot(sample_data["frequency"], sample_data[absorption_type],
                            marker='o', linestyle='-', label=sample)

                ax.set_xscale('log')  # Échelle logarithmique pour l'axe X
                ax.set_xticks(freq_ticks)
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.set_xticklabels(freq_ticks, rotation=30, ha="right")

                ax.set_title(f"Absorption : {absorption_type}")
                ax.set_xlabel("Fréquence (Hz)")
                ax.set_ylabel("Absorption")
                ax.legend(title="Échantillons")
                ax.grid(True, which="both", linestyle='--', alpha=0.7)

                st.pyplot(fig, use_container_width=True)

                # Création du PDF pour le téléchargement
                pdf_bytes = io.BytesIO()
                with PdfPages(pdf_bytes) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")
                pdf_bytes.seek(0)

                st.download_button(
                    label="Télécharger le graphique en PDF",
                    data=pdf_bytes,
                    file_name="courbes_absorption.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("Veuillez sélectionner au moins un échantillon pour afficher les courbes.")
