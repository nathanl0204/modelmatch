import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def get_model_family(model_name: str) -> str:
    """Détermine la famille d'un modèle à partir de son nom."""
    model_name_lower = model_name.lower()
    if 'llama-3' in model_name_lower or 'llama3' in model_name_lower:
        return 'Llama-3'
    if 'gemma' in model_name_lower:
        return 'Gemma'
    if 'qwen' in model_name_lower:
        return 'Qwen'
    if 'mistral' in model_name_lower:
        return 'Mistral'
    if 'phi-3' in model_name_lower:
        return 'Phi-3'
    if 'gpt-4o' in model_name_lower or 'gpt-5' in model_name_lower:
        return 'GPT'
    if 'deepseek' in model_name_lower:
        return 'Deepseek'
    if 'nemotron' in model_name_lower:
        return 'Nemotron'
    return 'Other'

def train_binary_classifier_for_pair(df: pd.DataFrame, family1: str, family2: str, models_dir: str):
    """Entraîne et sauvegarde un classificateur binaire pour distinguer deux familles de modèles."""
    print(f"\n--- Génération du classificateur pour la paire : '{family1}' vs '{family2}' ---")

    df_pair = df[df['family'].isin([family1, family2])].copy()

    if len(df_pair) < 50:
        print(f"Pas assez de données pour la pire '{family1}'/'{family2}'. Trouvé : {len(df_pair)} échantillons. Annulation.")
        return
    
    features_to_drop = ['model_name', 'conversation_id', 'prompt_index', 'family']
    X = df_pair.drop(columns=[col for col in features_to_drop if col in df_pair.columns])
    y = df_pair['family']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }

    xgb_estimator = XGBClassifier(random_state=42, eval_metric='logloss')

    n_splits = 3
    param_combinations = list(itertools.product(*param_grid.values()))
    total_fits = len(param_combinations) * n_splits

    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        scoring='accuracy',
        cv=StratifiedKFold(n_splits),
        n_jobs=-1,
        verbose=2
    )

    print(f"Entraînement avec GridSearchCV pour '{family1}' vs '{family2}' (n_jobs=-1)...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Récupère le meilleur classificateur trouvé pour la recherche
    final_classifier = grid_search.best_estimator_

    # Évalue le classificateur sur l'ensemble de test
    y_pred = final_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision pour '{family1}' vs '{family2}': {accuracy:.4f}")

    # --- Sauvegarde des résultats d'entraînement ---
    pair_dir = os.path.join(models_dir, "..", "training_results", f"binary_{family1.lower()}_vs_{family2.lower()}")
    os.makedirs(pair_dir, exist_ok=True)

    # Rapport de classification
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    report_path = os.path.join(pair_dir, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Classificateur binaire : {family1} vs {family2}\n")
        f.write(f"Précision (Accuracy) : {accuracy:.4f}\n")
        f.write(f"Meilleurs hyperparamètres : {grid_search.best_params_}\n\n")
        f.write(report)
    print(f"  Rapport de classification sauvegardé : {report_path}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_labels = (np.asarray(["{0:d}\n({1:.1%})".format(value, cm_normalized[i, j])
                                for i, row in enumerate(cm)
                                for j, value in enumerate(row)])
                    ).reshape(cm.shape)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Matrice de confusion — {family1} vs {family2}')
    plt.ylabel('Vraie famille')
    plt.xlabel('Famille prédite')
    plt.tight_layout()
    cm_path = os.path.join(pair_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Matrice de confusion sauvegardée : {cm_path}")

    # Prédictions du jeu de test
    pred_df = pd.DataFrame(X_test, columns=[c for c in df_pair.columns if c not in features_to_drop])
    pred_df['true_family'] = label_encoder.inverse_transform(y_test)
    pred_df['predicted_family'] = label_encoder.inverse_transform(y_pred)
    pred_path = os.path.join(pair_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Prédictions du jeu de test sauvegardées : {pred_path}")

    # Sauvegarde le modèle, le scaler, et l'encodeur
    os.makedirs(models_dir, exist_ok=True)

    model_filename = f"binary_{family1.lower()}_vs_{family2.lower()}_classifier.joblib"
    scaler_filename = f"binary_{family1.lower()}_vs_{family2.lower()}_scaler.joblib"
    encoder_filename = f"binary_{family1.lower()}_vs_{family2.lower()}_encoder.joblib"

    joblib.dump(final_classifier, os.path.join(models_dir, model_filename))
    joblib.dump(scaler, os.path.join(models_dir, scaler_filename))
    joblib.dump(label_encoder, os.path.join(models_dir, encoder_filename))
    
    return (family1, family2, accuracy)


def main():
    """Fonction principale pour orchestrer l'entraînement des classificateurs binaires."""
    parser = argparse.ArgumentParser(description="Entraîne les classificateurs binaires de départage entre familles.")
    parser.add_argument(
        "--csv",
        default="../data/fingerprints_for_classification.csv",
        help="Chemin du CSV d'empreintes à utiliser pour l'entraînement.",
    )
    args = parser.parse_args()

    CSV_PATH = args.csv
    MODELS_SAVE_DIR = "../models/binary_family_classifiers"

    # Définit les paires de familles pour lesquelles entraîner un classificateur
    PAIRS_TO_TRAIN = [
        ("Deepseek", "Mistral"),
        ("GPT", "Qwen"),
        ("Mistral", "Llama-3"),
        ("Phi-3", "Qwen"),
        ("Nemotron", "Qwen"),
    ]

    os.makedirs(MODELS_SAVE_DIR, exist_ok=True)

    print(f"Chargement des données depuis {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        df.fillna(0, inplace=True)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {CSV_PATH} n'a pas été trouvé.")
        return
    
    df['family'] = df['model_name'].apply(get_model_family)

    results = []
    for family1, family2 in PAIRS_TO_TRAIN:
        result = train_binary_classifier_for_pair(df, family1, family2, MODELS_SAVE_DIR)
        if result:
            results.append(result)
    
    print("\n\n" + "="*60)
    print("--- RÉSUMÉ DES PRÉCISIONS DES CLASSIFICATEURS BINAIRES ---")
    print("="*60)
    if not results:
        print("Aucun classificateur n'a été entraîné.")
    else:
        for f1, f2, acc in results:
            print(f"Paire : {f1:<10} vs {f2:<10} | Précision : {acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()