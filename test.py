import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.integrate import simps

# Définir la configuration de la page
st.set_page_config(
    page_title="DATABASE",  # Titre de l'onglet
    page_icon="https://shoplineimg.com/5ba2f18f88891600051d9b67/5c440c8d467a760ff5239042/800x.webp?source_format=png",  # URL de l'icône
    layout="centered"  # Choix de mise en page (par défaut "centered" ou "wide")
)
st.title("Visualisation des Courbes d'Absorption Acoustique")
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
                
                    st.header("Analyse Statistique Avancée")
                    
                    # Onglets pour organiser les différentes analyses
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Statistiques Clés",
                        "📈 Corrélations",
                        "🔍 Analyse Fréquentielle",
                        "🧮 Modélisation"
                    ])
                
                    with tab1:
                        st.subheader("Indicateurs de Performance")
                        stats_df = filtered_data.groupby('sample_info').agg({
                            'alpha_cabin': ['mean', 'max', 'min', 'std'],
                            'alpha_kundt': ['mean', 'max', 'min', 'std']
                        })
                        st.dataframe(stats_df.style.background_gradient(cmap='Blues'), height=300)
                
                    with tab2:
                        st.subheader("Matrice de Corrélation")
                        numerical_cols = [surface_mass_column, 'thickness_mm', 'alpha_cabin', 'alpha_kundt']
                        corr_matrix = filtered_data[numerical_cols].corr()
                        
                        fig, ax = plt.subplots(figsize=(10,6))
                        sns.heatmap(corr_matrix, 
                                    annot=True, 
                                    cmap='coolwarm', 
                                    center=0,
                                    ax=ax)
                        st.pyplot(fig)
                
                   # Remplacer la section "Analyse Fréquentielle" par :
                    with tab3:
                        st.subheader("Analyse par Bande Fréquentielle")
                        
                        # Calcul automatique des plages fréquentielles significatives
                        relevant_freq_ranges = {
                            "Basses fréquences (100-500 Hz)": (100, 500),
                            "Moyennes fréquences (500-2000 Hz)": (500, 2000),
                            "Hautes fréquences (2000-6000 Hz)": (2000, 6000)
                        }
                        
                        selected_range = st.selectbox(
                            "Plage fréquentielle cible:",
                            options=list(relevant_freq_ranges.keys())
                        )
                        fmin, fmax = relevant_freq_ranges[selected_range]
                        
                        # Calcul du NRC (Noise Reduction Coefficient)
                        def calculate_nrc(data):
                            freqs = np.array(data['frequency'])
                            alpha = np.array(data[absorption_type])
                            mask = (freqs >= 250) & (freqs <= 2000)
                            return round(np.mean(alpha[mask]).mean() * 2, 2) / 2
                        
                        # Calcul des indicateurs acoustiques
                        filtered_range = filtered_data[
                            (filtered_data['frequency'] >= fmin) & 
                            (filtered_data['frequency'] <= fmax)
                        ]
                        
                        if not filtered_range.empty:
                            # Visualisation de l'absorption intégrée
                            fig = plt.figure(figsize=(12,5))
                            
                            # Graphique d'absorption pondérée
                            plt.subplot(121)
                            for material in filtered_range['material_family'].unique():
                                material_data = filtered_range[filtered_range['material_family'] == material]
                                plt.plot(material_data['frequency'], material_data[absorption_type], 
                                        label=material, alpha=0.7)
                            plt.title("Réponse fréquentielle par matériau")
                            plt.xscale('log')
                            
                            # Calcul des indicateurs
                            nrc_values = filtered_range.groupby('material_family').apply(calculate_nrc)
                            peak_alpha = filtered_range.groupby('material_family')[absorption_type].max()
                            
                            # Affichage des indicateurs
                            plt.subplot(122)
                            plt.bar(nrc_values.index, nrc_values.values, alpha=0.6, label='NRC')
                            plt.bar(peak_alpha.index, peak_alpha.values, alpha=0.6, label='Alpha max')
                            plt.xticks(rotation=45)
                            plt.legend()
                            plt.title("Indicateurs de performance")
                            
                            st.pyplot(fig)
                            
                            # Avertissement scientifique
                            st.info("""
                            **Interprétation acoustique :**
                            - NRC (Noise Reduction Coefficient) : Moyenne des α sur 250-2000 Hz
                            - Alpha max : Valeur maximale dans la plage sélectionnée
                            """)
                        else:
                            st.warning("Aucune donnée dans cette plage fréquentielle")
                
                    with tab4:
                        st.subheader("Modélisation Linéaire")
                        freq_range = st.slider(
                            "Fréquence cible pour la modélisation:",
                            min_value=int(filtered_data['frequency'].min()),
                            max_value=int(filtered_data['frequency'].max()),
                            value=1000
                        )
                        
                        # Préparation des données
                        model_data = filtered_data[filtered_data['frequency'] == freq_range]
                        X = model_data[[surface_mass_column, 'thickness_mm']]
                        y = model_data[absorption_type]
                        
                        # Entraînement du modèle
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Affichage des résultats
                        st.write(f"#### Coefficients @ {freq_range}Hz")
                        st.write(f"- Masse surfacique: {model.coef_[0]:.4f}")
                        st.write(f"- Épaisseur: {model.coef_[1]:.4f}")
                        st.write(f"- Intercept: {model.intercept_:.4f}")
                        st.write(f"R²: {model.score(X, y):.2f}")


if uploaded_file and df is not None:  # Vérification de l'existence de df
    with st.expander("🔬 Analyse Comparative par Matériau"):
        if 'material_family' in filtered_df.columns:
            material_choice = st.selectbox(
                "Choisir une famille de matériaux:",
                options=filtered_df['material_family'].unique()
            )
            
            # Calcul des paramètres acoustiques
            material_data = filtered_df[filtered_df['material_family'] == material_choice]
            freq_response = material_data.groupby('frequency')[absorption_type].mean()
            
            # Création du graphique
            fig, ax = plt.subplots(figsize=(10,6))
            ax.semilogx(freq_response.index, freq_response.values, '-o', lw=2)
            
            # Annotation des points clés
            peak_freq = freq_response.idxmax()
            ax.annotate(f'α={freq_response.max():.2f} @ {peak_freq}Hz',
                        xy=(peak_freq, freq_response.max()),
                        xytext=(peak_freq*1.5, freq_response.max()*0.8),
                        arrowprops=dict(facecolor='black', shrink=0.05))
            
            ax.set_title(f"Réponse acoustique - {material_choice}")
            st.pyplot(fig)
        else:
            st.error("Données matériaux non disponibles")

# Display the Git URL with the new formatting
st.markdown(
    '<p style="color: blue; font-size: 14px; text-align: center;">'
    '<br><br><br>'  # Adds spacing between the elements
    '<br><br><br>'
    '<br><br><br>'
    '<br><br><br>'
    '<br><br><br>'
    'GitHub Link: <a href="https://github.com/LinoVation1312/database" style="color: blue; text-decoration: none;" target="_blank">'
    'https://github.com/LinoVation1312/database</a>'
    '<br><br><br>'
    'Lino CONORD, Déc. 2024'
    '</p>',
    unsafe_allow_html=True
)
