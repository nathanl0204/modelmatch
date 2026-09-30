import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance(model_path: str, csv_path: str):
    """
    Analyse l'importance des caractéristiques (métriques) d'un classificateur entraîné.

    Cette fonction charge un modèle de classification (ex: RandomForest) et un fichier CSV
    contenant les données utilisées pour l'entraînement. Elle extrait les importances
    des caractéristiques du modèle, affiche les plus importantes, sauvegarde un rapport
    complet au format CSV, et génère un graphique à barres des 30 caractéristiques
    les plus importantes.

    Args:
        model_path (str): Le chemin vers le fichier du modèle sérialisé (ex: .joblib).
        csv_path (str): Le chemin vers le fichier CSV des empreintes.
    """
    print(f"Chargement du modèle depuis {model_path}...")
    try:
        classifier = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier du modèle '{model_path}' n'a pas été trouvé.")
        return
    
    print(f"Chargement des données depuis {csv_path} pour récupérer les noms des caractéristiques...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier CSV '{csv_path}' n'a pas été trouvé.")
        return
    
    features_to_drop = ['model_name', 'conversation_id', 'prompt_index']
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    feature_names = X.columns

    if len(feature_names) != len(classifier.feature_importances_):
        print("Erreur : Le nombre de caractéristiques dans le CSV ne correspond pas à celui du modèle entraîné.")
        return
    
    print("\nAnalyse de l'importance des caractéristiques...")
    importances = classifier.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("\nTop 20 des métriques les plus importantes :")
    print(feature_importance_df.head(20))

    feature_importance_report_path = "feature_importance_report_family.csv"
    feature_importance_df.to_csv(feature_importance_report_path, index=False)
    print(f"\nRapport d'importance sauvegardé dans : {feature_importance_report_path}")

    plt.figure(figsize=(12, 15))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(30), palette='viridis')
    plt.title('Top 30 des caractéristiques les plus importantes')
    plt.xlabel('Importance')
    plt.ylabel('Caractéristique (métrique)')
    plt.tight_layout()

    plot_path = "feature_importance_plot_family.png"
    plt.savefig(plot_path)
    print(f"Graphique de l'importance sauvegardé dans : {plot_path}")
    plt.show()

if __name__ == "__main__":
    MODEL_SAVE_PATH = "../models/family_classifier.joblib"
    FINGERPRINTS_CSV_PATH = "../data/fingerprints_for_classification.csv"
    analyze_feature_importance(MODEL_SAVE_PATH, FINGERPRINTS_CSV_PATH)