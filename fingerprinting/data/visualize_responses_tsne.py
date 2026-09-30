import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

import sys
# Ajoute le répertoire src au path pour permettre l'importation de modules personnalisés
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
try:
    from fingerprinter import get_model_family
except ImportError:
    # Fonction de secours si l'importation échoue
    def get_model_family(model_name: str) -> str:
        model_name_lower = model_name.lower()
        if 'llama' in model_name_lower: return 'Llama-3'
        if 'gemma' in model_name_lower: return 'Gemma'
        if 'qwen' in model_name_lower: return 'Qwen2'
        if 'mistral' in model_name_lower: return 'Mistral'
        if 'phi-3' in model_name_lower: return 'Phi-3'
        if 'deepseek' in model_name_lower: return 'Deepseek'
        if 'gpt-4o' in model_name_lower or 'gpt-5' in model_name_lower: return 'GPT'
        return 'Other'

def visualize_classifier_probabilities(fingerprints_csv_path: str, model_path: str, scaler_path: str, output_path: str):
    """
    Génère une visualisation t-SNE statique des probabilités du classificateur de famille.
    Chaque point représente une réponse de modèle, coloré par sa famille.
    """
    print(f"1. Chargement des empreintes depuis '{fingerprints_csv_path}'...")
    try:
        df = pd.read_csv(fingerprints_csv_path)
        df.fillna(0, inplace=True)
    except FileNotFoundError:
        print(f"Erreur: Fichier d'empreintes '{fingerprints_csv_path}' non trouvé.")
        return

    print(f"2. Chargement du classificateur depuis '{model_path}' et du scaler depuis '{scaler_path}'...")
    try:
        classifier = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        print(f"Erreur: Impossible de charger le modèle ou le scaler. {e}")
        print("Veuillez exécuter 'train_family_classifier.py' d'abord.")
        return

    print("3. Préparation des données et application du scaler...")
    df['family'] = df['model_name'].apply(get_model_family)
    
    # S'assure que les clonnes du DataFrame correspondent à celles attendues par le scaler
    X = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(X)

    print("4. Prédiction des probabilités avec le classificateur de famille...")
    probabilities = classifier.predict_proba(X_scaled)

    print("5. Application de t-SNE sur les vecteurs de probabilités...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(probabilities)

    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]

    print(f"6. Génération du graphique et sauvegarde dans '{output_path}'...")

    family_colors = {
        'Llama-3': '#1f77b4',
        'Gemma': '#ff7f0e',
        'Qwen2': '#2ca02c',
        'Mistral': '#d62728',
        'Phi-3': '#9467bd',
        'Deepseek': '#8c564b',
        'GPT': '#ff69b4',
        'Other': '#7f7f7f'
    }

    plt.figure(figsize=(16, 12))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="family",
        palette=family_colors,
        data=df,
        legend="full",
        alpha=0.8
    )
    plt.title('Visualisation t-SNE des probabilités de classification par famille', fontsize=18)
    plt.xlabel('Composante t-SNE 1', fontsize=14)
    plt.ylabel('Composante t-SNE 2', fontsize=14)
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    analyze_confusion_zone(df, scaler.feature_names_in_)

    plt.savefig(output_path)
    print(f"   Graphique sauvegardé dans '{output_path}'.")
    plt.show()


def analyze_confusion_zone(df: pd.DataFrame, feature_names: list[str]):
    """
    Analyse les métriques des points situés dans une zone de confusion définie.
    """
    x_min, x_max = -14, 0
    y_min, y_max = -3, 5

    confusion_df = df[
        (df['tsne-one'].between(x_min, x_max)) &
        (df['tsne-two'].between(y_min, y_max))
    ]
    clear_df = df[~df.index.isin(confusion_df.index)]

    if confusion_df.empty:
        print("\nAucun point trouvé dans la zone de confusion définie. Ajustez les coordonnées.")
        return
    
    print(f"\n--- ANALYSE DE LA ZONE DE CONFUSION ({len(confusion_df)} points) ---")

    confused_families = confusion_df['family'].unique()
    print(f"Familles présentes dans la zone : {confused_families}")

    for family in confused_families:
        print(f"\n--- Famille : {family} ---")

        in_zone_metrics = confusion_df[confusion_df['family'] == family][feature_names].mean()

        out_zone_metrics = clear_df[clear_df['family'] == family][feature_names].mean()

        metric_diff = (in_zone_metrics - out_zone_metrics) / (out_zone_metrics.abs() + 1e-6)

        print("Top 5 des métriques les plus différentes (points confus vs clairs) :")
        print(metric_diff.abs().nlargest(5))


if __name__ == "__main__":
    FINGERPRINTS_PATH = os.path.join('..', 'data', 'fingerprints_for_classification.csv')
    MODEL_PATH = os.path.join('..', 'models', 'family_classifier.joblib')
    SCALER_PATH = os.path.join('..', 'models', 'family_classifier_scaler.joblib')
    OUTPUT_PLOT_PATH = 'classifier_probabilities_tsne_visualization2.png'

    visualize_classifier_probabilities(
        fingerprints_csv_path=FINGERPRINTS_PATH,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        output_path=OUTPUT_PLOT_PATH
    )