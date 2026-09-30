import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TOP_N_FEATURES = 50
FEATURE_IMPORTANCE_REPORT_PATH = "feature_importance_report2.csv"

def train_model_fingerprint_classifier(csv_path: str, model_save_path: str = "../models/fingerprint_classifier.joblib", encoder_save_path: str = "../models/model_name_encoder.joblib", feature_importance_path: str = FEATURE_IMPORTANCE_REPORT_PATH):
    """
    Entraîne un classificateur pour identifier des modèles spécifiques à partir de leurs empreintes stylistiques.
    """
    print(f"Chargement des données depuis {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {csv_path} n'a pas été trouvé. Veuillez d'abord exécuter fingerprinter.py.")
        return
    
    print("Préparation des données...")
    df.fillna(0, inplace=True)

    if df.empty:
        print("Le DataFrame est vide. Impossible de continuer.")
        return
    
    features_to_drop = ['model_name', 'conversation_id', 'prompt_index']
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    y = df['model_name']

    # Bloc commenté pour utiliser uniquement les N caractéristiques les plus importantes
    # Décommenter pour activer la sélection de caractéristiques basée sur un rapport pré-généré
    """ try:
        print(f"\nChargement des caractéristiques depuis '{feature_importance_path}'...")
        feature_importance_df = pd.read_csv(feature_importance_path)
        top_features = feature_importance_df['feature'].head(TOP_N_FEATURES).tolist()
        print(f"Utilisation des {len(top_features)} caractéristiques les plus importantes.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier d'importance des caractéristiques '{feature_importance_path}' n'a pas été trouvé.")
        print("Veuillez d'abord exécuter analyze_importance.py pour le générer.")
        return

    print(f"\nSélection des {len(top_features)} caractéristiques les plus pertinentes...")
    X = X[top_features] """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nClasses de modèles détectées :")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"- {class_name} -> {i}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Bloc commenté pour l'optimisation des hyperparamètres avec GridSearchCV
    # Décommenter pour rechercher les meilleurs paramètres au lieu d'utiliser ceux par défaut
    """ print("\nOptimisation des hyperparamètres pour RandomForestClassifier avec GridSearchCV...")

    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    print("\nEntraînement avec GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)
    print("Entraînement terminé.")

    print("\nMeilleurs hyperparamètres trouvés :")
    print(grid_search.best_params_)

    classifier = grid_search.best_estimator_ """

    print("\nEntraînement du RandomForestClassifier avec les paramètres par défaut...")
    classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    classifier.fit(X_train_scaled, y_train)
    print("Entraînement terminé.")

    print("\nÉvaluation du modèle sur l'ensemble de test...")
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrécision (Accuracy): {accuracy:.4f}")

    print("\nRapport de classification :")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print(report)

    classification_report_path = "training_results/classification_report_top_40_features2.txt"
    with open(classification_report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Rapport de classification sauvegardé dans {classification_report_path}")

    print("Génération de la matrice de confusion...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    confusion_matrix_path = "training_results/confusion_matrix_top_40_features2.png"
    plt.savefig(confusion_matrix_path)
    print(f"Matrice de confusion sauvegardée dans {confusion_matrix_path}")
    plt.show()

    print("\nSauvegarde du modèle, du scaler et de l'encodeur...")
    joblib.dump(classifier, model_save_path)
    joblib.dump(scaler, model_save_path.replace(".joblib", "_scaler.joblib"))
    joblib.dump(label_encoder, encoder_save_path)
    print(f"Modèle sauvegardé dans : {model_save_path}")
    print(f"Scaler sauvegardé dans : {model_save_path.replace('.joblib', '_scaler.joblib')}")
    print(f"Encodeur de lables sauvegardé dans : {encoder_save_path}")


if __name__ == "__main__":
    import os
    if not os.path.exists("../models"):
        os.makedirs("../models")

    FINGERPRINTS_CSV_PATH = "../data/fingerprints_for_classification.csv"
    train_model_fingerprint_classifier(FINGERPRINTS_CSV_PATH)
