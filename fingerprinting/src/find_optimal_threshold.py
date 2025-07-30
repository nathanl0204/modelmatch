import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
from xgboost import XGBClassifier

# Ajoute le répertoire src au path pour permettre l'importation de modules personnalisés
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
try:
    from fingerprinter import get_model_family
except ImportError:
    print("Avertissement: Impossible d'importer get_model_family. Utilisation d'une copie locale.")
    def get_model_family(model_name: str) -> str:
        return model_name.split('/')[1].split('-')[0]

def find_optimal_confidence_threshold():
    """
    Entraîne le classificateur de famille, évalue les probabilités de prédiction sur un jeu 
    de validation, et détermine le seuil de confiance optimal qui maximise le F1-score
    pour décider quand une prédicition est "fiable".
    """
    CSV_PATH = "../data/fingerprints_for_classification.csv"
    print(f"Chargement des données depuis {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        df.fillna(0, inplace=True)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {CSV_PATH} n'a pas été trouvé.")
        return
    
    df['family'] = df['model_name'].apply(get_model_family)

    X = df.drop(columns=['model_name', 'conversation_id', 'prompt_index', 'family'])
    y = df['family']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 60% entraînement, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Taille des ensembles: Entraînement={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    print("\nEntraînement du classificateur de famille...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
    classifier.fit(X_train_scaled, y_train)

    print("\nDétermination du seuil de confiance optimal sur le jeu de validation...")
    X_val_scaled = scaler.transform(X_val)

    y_val_pred = classifier.predict(X_val_scaled)
    y_val_probs = classifier.predict_proba(X_val_scaled)

    # Label binaire : 1 si la prédiction est correcte, 0 sinon
    is_correct_prediction = (y_val_pred == y_val).astype(int)

    # Score = probabilité maximale de la classe prédite
    max_probabilities = np.max(y_val_probs, axis=1)

    precision, recall, thresholds = precision_recall_curve(is_correct_prediction, max_probabilities)

    # Ajout d'un epsilon pour éviter la division par zéro
    f1_scores = np.divide(2 * precision * recall, precision + recall + 1e-9)

    # f1_scores a une longueur de len(thresholds) + 1. On ignore la dernière valeur
    # qui ne correspond à aucun seuil, pour aligner les tailles des tableaux.
    f1_scores_for_thresholds = f1_scores[:-1]

    valid_indices = np.isfinite(f1_scores_for_thresholds)
    valid_thresholds = thresholds[valid_indices]
    valid_f1_scores = f1_scores_for_thresholds[valid_indices]

    if len(valid_f1_scores) == 0:
        print("Impossible de déterminer un seuil F1-score valide.")
        optimal_threshold = 0.65 # Valeur par défaut
    else:
        optimal_idx = np.argmax(valid_f1_scores)
        optimal_threshold = valid_thresholds[optimal_idx]
    
    print("\n" + "="*50)
    print("--- RÉSULTAT DE L'ANALYSE DU SEUIL ---")
    print(f"Le seuil de confiance optimal est : {optimal_threshold:.4f}")
    print("Ce seuil a été choisi car il maximise le F1-score sur le jeu de données de validation.")
    print(f"À ce seuil, le F1-score est de {valid_f1_scores[optimal_idx]:.4f}.")
    print("\nACTION REQUISE:")
    print(f"Veuillez mettre à jour la constante 'FAMILY_CONFIDENCE_THRESHOLD' dans le fichier 'plugin/classifier_pipeline.py' avec la valeur {optimal_threshold:.4f}.")
    print("="*50)

    X_test_scaled = scaler.transform(X_test)
    y_test_pred = classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nPrécision du classificateur de famille sur le jeu de test final (non vu): {test_accuracy:.2%}")


if __name__ == "__main__":
    find_optimal_confidence_threshold()