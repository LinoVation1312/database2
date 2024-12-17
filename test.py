import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import numpy as np

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
        .str.strip()         # Retirer les espaces autour
        .str.lower()         # Convertir en minuscules
        .str.replace(r'[^\w\s]', '', regex=True)  # Retirer les caractères spéciaux
        .str.replace(' ', '_')  # Remplacer les espaces par des underscores
    )

    # Suppression des doublons de colonnes
    if "sample_number_stn1" in df.columns:
        df = df.drop(columns=["sample_number_stn1"])

    return df

uploaded_file = st.file_uploader("Chargez votre fichier Excel", type=["xlsx"])

if uploaded_file:
    # Chargement et nettoyage des données
    df = load_data(uploaded_file)

    if df is not None:
        surface_mass_column = None
        for col in df.columns:
            if 'surface_mass' in col:
                surface_mass_column = col
                break

        if surface_mass_column is None:
            st.error("La colonne 'surface_mass_gm2' ou équivalente est manquante dans les données.")
        else:
            columns_to_keep = [
                "sample_number_stn", "trim_level", "project_number", "material_family",
                "material_supplier", "detailed_description", surface_mass_column, 
                "thickness_mm", "assembly_type", "finished_good_surface_aera", 
                "frequency", "alpha_cabin", "alpha_kundt"
            ]

            missing_columns = [col for col in columns_to_keep if col not in df.columns]
            if missing_columns:
                st.error(f"Colonnes manquantes : {missing_columns}")
            else:
                df = df[columns_to_keep]

                # Remplacer les valeurs NaN dans les colonnes pour l'étiquette
                df = df.fillna({"assembly_type": "Ø", "trim_level": "Ø", "material_family": "Ø", surface_mass_column: "Ø"})

                df[surface_mass_column] = pd.to_numeric(df[surface_mass_column], errors='coerce')
                df["thickness_mm"] = pd.to_numeric(df["thickness_mm"], errors='coerce')

                # Création d'une colonne enrichie pour affichage
                df["sample_info"] = (
                    df["sample_number_stn"] + " | " +
                    df["assembly_type"].astype(str) + " | " +
                    df[surface_mass_column].fillna("Ø").astype(str) + " g/m² | " +
                    df["material_family"].astype(str) + " | " +
                    df["trim_level"].astype(str)
                )

                # Sélection des filtres
                trim_level = st.sidebar.selectbox("Sélectionnez un Trim Level :", ["Tous"] + list(df["trim_level"].unique()))
                supplier = st.sidebar.selectbox("Sélectionnez un Supplier :", ["Tous"] + list(df["material_supplier"].unique()))
                surface_mass = st.sidebar.slider(
                    "Sélectionnez une plage de Surface Mass (g/m²) :", 
                    min_value=int(df[surface_mass_column].min(skipna=True)), 
                    max_value=int(df[surface_mass_column].max(skipna=True)), 
                    value=(int(df[surface_mass_column].min(skipna=True)), int(df[surface_mass_column].max(skipna=True)))
                )
                thickness = st.sidebar.slider(
                    "Sélectionnez une plage de Thickness (mm) :", 
                    min_value=float(df["thickness_mm"].min(skipna=True)), 
                    max_value=float(df["thickness_mm"].max(skipna=True)), 
                    value=(float(df["thickness_mm"].min(skipna=True)), float(df["thickness_mm"].max(skipna=True)))
                )
                assembly_type = st.sidebar.selectbox("Sélectionnez un Assembly Type :", ["Tous"] + list(df["assembly_type"].unique()))

                filtered_df = df.copy()
                if trim_level != "Tous":
                    filtered_df = filtered_df[filtered_df["trim_level"] == trim_level]
                if supplier != "Tous":
                    filtered_df = filtered_df[filtered_df["material_supplier"] == supplier]
                filtered_df = filtered_df[
                    (filtered_df[surface_mass_column] >= surface_mass[0]) & 
                    (filtered_df[surface_mass_column] <= surface_mass[1])
                ]
                filtered_df = filtered_df[
                    (filtered_df["thickness_mm"] >= thickness[0]) & 
                    (filtered_df["thickness_mm"] <= thickness[1])
                ]
                if assembly_type != "Tous":
                    filtered_df = filtered_df[filtered_df["assembly_type"] == assembly_type]

                sample_numbers = st.sidebar.multiselect("Sélectionnez plusieurs échantillons :", filtered_df["sample_info"].unique())

                if sample_numbers:
                    filtered_data = filtered_df[filtered_df["sample_info"].isin(sample_numbers)]

                    absorption_type = st.sidebar.radio("Type d'absorption :", ["alpha_cabin", "alpha_kundt"])
                    freq_ticks = {
                        "alpha_cabin": [315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
                        "alpha_kundt": [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300]
                    }[absorption_type]

                    st.subheader(f"Comparaison des courbes d'absorption pour {absorption_type}")
                    
                    # Ajustement de la taille du graphique pour plus de lisibilité
                    fig, ax = plt.subplots(figsize=(12, 8))  # Taille élargie
                    
                    # Tracer les courbes
                    for sample in sample_numbers:
                        data_sample = filtered_data[filtered_data["sample_info"] == sample]
                        ax.plot(data_sample["frequency"], data_sample[absorption_type],
                                marker='o', linestyle='-', label=sample)
                    
                    # Définir les fréquences spécifiques pour les étiquettes et l'axe X
                    ax.set_xscale('log')  # Échelle logarithmique pour l'axe X
                    ax.set_xticks(freq_ticks)  # Points spécifiques pour les étiquettes
                    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Forcer un format lisible
                    
                    # Rotation et alignement des étiquettes
                    ax.set_xticklabels(freq_ticks, rotation=25, ha="right")
                    
                    # Titres et légendes
                    ax.set_title(f"Absorption : {absorption_type}")
                    ax.set_xlabel("Fréquence (Hz)")
                    ax.set_ylabel(absorption_type)
                    ax.legend(title="Échantillons")  # Déplacer la légende
                    ax.grid(True, which="both", linestyle='--', alpha=0.7)
                    
                    # Afficher le graphique dans Streamlit avec largeur adaptative
                    st.pyplot(fig, use_container_width=True)

                    # Générer le PDF
                    pdf_bytes = io.BytesIO()
                    with PdfPages(pdf_bytes) as pdf:
                        pdf.savefig(fig, bbox_inches="tight")
                    pdf_bytes.seek(0)

                    # Bouton de téléchargement PDF
                    st.download_button(
                        label="Télécharger le graphique en PDF",
                        data=pdf_bytes,
                        file_name="courbes_absorption.pdf",
                        mime="application/pdf"
                    )

                    # Générer l'image JPEG
                    img_bytes = io.BytesIO()
                    fig.savefig(img_bytes, format="jpeg")
                    img_bytes.seek(0)

                    # Bouton de téléchargement JPEG
                    st.download_button(
                        label="Télécharger le graphique en JPEG",
                        data=img_bytes,
                        file_name="courbes_absorption.jpeg",
                        mime="image/jpeg"
                    )
