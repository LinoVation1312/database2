import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import matplotlib.ticker as ticker

st.title("Visualisation Optimisée des Courbes d'Absorption Acoustique")
st.write("Sélectionnez plusieurs échantillons pour comparer leurs courbes d'absorption.")

@st.cache_data
def load_data(file):
    """
    Chargement des données depuis Excel et normalisation stricte des colonnes.
    Cette version gère les variations dans le nom de la feuille, la ligne d'en-tête et fait correspondre les colonnes par position.
    """
    # Charger le fichier Excel
    xls = pd.ExcelFile(file, engine="openpyxl")

    # Chercher le nom de la feuille 'DATA' ou 'DATA ' (avec ou sans espace)
    sheet_name = None
    for sheet in xls.sheet_names:
        if 'data' in sheet.lower():  # Rechercher la feuille 'data' sans tenir compte de l'espace
            sheet_name = sheet
            break
    
    if sheet_name is None:
        raise ValueError("La feuille 'DATA' n'a pas été trouvée dans le fichier.")

    # Vérification des deux premières lignes pour voir si elles sont vides
    df_check = pd.read_excel(xls, sheet_name=sheet_name, header=None, engine="openpyxl", nrows=2)
    
    # Si les deux premières lignes sont vides, commencer à partir de la troisième ligne
    if df_check.isnull().all().all():
        header_row = 2
    else:
        header_row = 0

    # Charger les données en tenant compte de la ligne d'en-tête correcte
    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row, engine="openpyxl")

    # Liste des étiquettes de colonnes attendues
    expected_columns = [
        "sample_number_stn", "trim_level", "project_number", "material_family",
        "material_supplier", "detailed_description", "surface_mass_gm2", 
        "thickness_mm", "assembly_type", "finished_good_surface_aera", 
        "frequency", "alpha_cabin", "alpha_kundt"
    ]

    # Mappage dynamique des colonnes du fichier Excel à des noms standardisés
    column_mapping = {
        "Sample Number (STN)": "sample_number_stn",
        "Trim level": "trim_level",
        "Project number": "project_number",
        "Material family": "material_family",
        "Material supplier": "material_supplier",
        "Detailed Description": "detailed_description",
        "Surface Mass (g/m²)": "surface_mass_gm2",
        "Thickness (mm)": "thickness_mm",
        "Assembly Type": "assembly_type",
        "Finished Good surface aera": "finished_good_surface_aera",
        "Frequency": "frequency",
        "Alpha cabin": "alpha_cabin",
        "Alpha Kundt": "alpha_kundt"
    }

    # Normaliser les noms des colonnes
    df = df.rename(columns=lambda x: x.strip())  # Supprimer les espaces superflus
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)  # Retirer les caractères spéciaux
    df.columns = df.columns.str.lower().str.replace(' ', '_')  # Convertir en minuscules et remplacer les espaces

    # Appliquer le mappage des colonnes
    df = df.rename(columns=column_mapping)

    return df

uploaded_file = st.file_uploader("Chargez votre fichier Excel", type=["xlsx"])

if uploaded_file:
    # Chargement et nettoyage des données
    df = load_data(uploaded_file)

    # Vérification de la présence de la colonne 'surface_mass_gm2'
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
                
                # Mettre l'échelle logarithmique pour l'axe des fréquences
                ax.set_xscale('log', base=2)

                # Configurer la grille avec échelle logarithmique
                ax.grid(True, which="both", axis="x", linestyle='--', color='gray', alpha=0.7)
                ax.set_title(f"Absorption : {absorption_type}")
                ax.set_xlabel("Fréquence (Hz)")
                ax.set_ylabel(absorption_type)
                ax.legend(title="Échantillons")

                # Ajout des ticks log-spacés pour l'axe des fréquences
                ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, subs='auto', numticks=10))

                # Changer l'échelle pour afficher des valeurs entre 1 et 10000
                def custom_ticks(x, pos):
                    if x == 0:
                        return "0"
                    return f"{int(2**x):,}"

                ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_ticks))

                ax.grid(True)
                st.pyplot(fig)

                # Générer un lien pour télécharger le graphique en PDF
                pdf_bytes = io.BytesIO()
                with PdfPages(pdf_bytes) as pdf:
                    pdf.savefig(fig)
                pdf_bytes.seek(0)

                st.download_button(
                    label="Télécharger le graphique en PDF",
                    data=pdf_bytes,
                    file_name="courbes_absorption.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("Veuillez sélectionner au moins un échantillon pour afficher les courbes.")
