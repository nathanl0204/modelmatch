import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def analyze_pair_distinction(csv_path: str, family1: str, family2: str, top_n_features: int = 20):
    """
    Analyse les caractéristiques les plus discriminantes entre deux familles de modèles.
    Entraîne un classificateur binaire et visualise l'importance des caractéristiques.
    """
    print(f"\n--- Analyse de la distinction entre '{family1}' et '{family2}' ---")

    try:
        df = pd.read_csv(csv_path)
        df.fillna(0, inplace=True)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {csv_path} n'a pas été trouvé.")
        return
    
    def get_model_family(model_name: str) -> str:
        model_name = model_name.lower()
        if 'llama-3' in model_name: return 'Llama-3'
        if 'gemma' in model_name: return 'Gemma'
        if 'qwen2' in model_name: return 'Qwen2'
        if 'phi-3' in model_name: return 'Phi-3'
        if 'mistral' in model_name: return 'Mistral'
        if 'deepseek' in model_name: return 'Deepseek'
        if 'gpt-4o' in model_name: return 'GPT-4o'
        return model_name.split('/')[0]
    
    df['family'] = df['model_name'].apply(get_model_family)

    df_pair = df[df['family'].isin([family1, family2])].copy()

    if len(df_pair) < 20:
        print(f"Pas assez de données pour la paire '{family1}'/'{family2}'. Trouvé : {len(df_pair)} échantillons.")
        return
    
    print(f"Données trouvées: {len(df_pair[df_pair['family']==family1])} pour {family1}, {len(df_pair[df_pair['family']==family2])} pour {family2}")

    features_to_drop = ['model_name', 'conversation_id', 'prompt_index', 'family']
    X = df_pair.drop(columns=[col for col in features_to_drop if col in df_pair.columns])
    y = df_pair['family']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    classifier.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, classifier.predict(X_test_scaled))
    print(f"\nPrécision du classificateur binaire ({family1} vs {family2}): {accuracy:.2f}")
    
    importances = classifier.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print(f"\nTop {top_n_features} des caractéristiques les plus discriminantes pour '{family1}' vs '{family2}':")
    print(feature_importance_df.head(top_n_features))

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n_features), palette='viridis')
    plt.title(f"Importance des caractéristiques pour distinguer\n{family1} vs {family2}")
    plt.xlabel("Importance")
    plt.ylabel("Caractéristique")
    plt.tight_layout()

    plot_path = f"importance_{family1}_vs_{family2}.png"
    plt.savefig(plot_path)
    print(f"Graphique sauvegardé dans : {plot_path}")
    plt.show()


if __name__ == '__main__':
    CSV_DATA_PATH = "../data/fingerprints_for_classification.csv"

    analyze_pair_distinction(CSV_DATA_PATH, 'Deepseek', 'Mistral')
    analyze_pair_distinction(CSV_DATA_PATH, 'GPT-4o', 'Qwen2')
    analyze_pair_distinction(CSV_DATA_PATH, 'Mistral', 'Llama-3')
    analyze_pair_distinction(CSV_DATA_PATH, 'Phi-3', 'Qwen2')