import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm
import itertools
import os

def get_model_family(model_name: str) -> str:
    """Détermine la famille d'un modèle à partir de son nom."""
    model_name = model_name.lower()
    if 'llama-3' in model_name:
        return 'Llama-3'
    if 'gemma' in model_name:
        return 'Gemma'
    if 'qwen2' in model_name:
        return 'Qwen2'
    if 'phi-3' in model_name:
        return 'Phi-3'
    if 'mistral' in model_name:
        return 'Mistral'
    if 'deepseek' in model_name:
        return 'Deepseek'
    if 'gpt-4o' in model_name:
        return 'GPT-4o'
    return model_name.split('/')[0]

def train_family_classifier(csv_path: str, model_save_path: str = "../models/family_classifier.joblib", encoder_save_path: str = "../models/family_name_encoder.joblib"):
    """
    Entraîne un classificateur pour identifier la famille d'un modèle (ex: Llama, Gemma)
    à partir de ses empreintes stylistiques.
    """
    print(f"Chargement des données depuis {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {csv_path} n'a pas été trouvé.")
        return
    
    print("Préparation des données...")
    df.fillna(0, inplace=True)

    if df.empty:
        print("Le DataFrame est vide. Impossible de continuer.")
        return
    
    features_to_drop = ['model_name', 'conversation_id', 'prompt_index']
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    y = df['model_name']

    y_family = y.apply(get_model_family)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_family)

    print("\nFamilles de modèles détectées :")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"- {class_name} -> {i}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nOptimisation des hyperparamètres pour RandomForestClassifier avec GridSearchCV...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0]
    }

    xgb_estimator = XGBClassifier(random_state=42, eval_metric='mlogloss')

    n_splits = 3
    param_combinations = list(itertools.product(*param_grid.values()))
    total_fits = len(param_combinations) * n_splits

    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits),
        n_jobs=-1,
        verbose=0,
        scoring='accuracy'
    )

    print("\nEntraînement avec GridSearchCV...")
    # Utilise tqdm poour afficher une barre de progression pour le GridSearchCV
    with tqdm(total=total_fits, desc="GridSearch progress") as pbar:
        def on_step(x):
            pbar.update(1)

        original_fit = grid_search._run_search
        def new_fit(self, *args, **kwargs):
            original_fit(*args, **kwargs)
            on_step(None)
        
        grid_search._run_search = lambda x: new_fit(grid_search, x)

        grid_search.fit(X_train_scaled, y_train)

    print("Entraînement terminé.")

    print("\nMeilleurs hyperparamètres trouvés :")
    print(grid_search.best_params_)
    classifier = grid_search.best_estimator_

    print("\nÉvaluation du modèle sur l'ensemble de test...")
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrécision (Accuracy): {accuracy:.4f}")

    print("\nRapport de classification par famille :")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print(report)

    classification_report_path = "training_results/family_classification_report.txt"
    with open(classification_report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Rapport de classification sauvegardé dans {classification_report_path}")

    print("\nGénération de la matrice de confusion pour les familles...")
    cm = confusion_matrix(y_test, y_pred)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Crée des étiquettes combinant le nombre absolu et le pourcentage
    annot_labels = (np.asarray(["{0:d}\n({1:.1%})".format(value, cm_normalized[i, j])
                                for i, row in enumerate(cm)
                                for j, value in enumerate(row)])
                    ).reshape(cm.shape)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matrice de confusion - Classification par famille')
    plt.ylabel('Vraie famille')
    plt.xlabel('Famille prédite')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    confusion_matrix_path = "training_results/family_confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    print(f"Matrice de confusion sauvegardée dans {confusion_matrix_path}")
    plt.show()

    print("\nSauvegarde du modèle, du scaler et de l'encodeur de famille...")

    model_dir = os.path.dirname(model_save_path)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(classifier, model_save_path)
    joblib.dump(scaler, model_save_path.replace(".joblib", "_scaler.joblib"))
    joblib.dump(label_encoder, encoder_save_path)
    print(f"Modèle de famille sauvegardé dans : {model_save_path}")
    print(f"Scaler sauvegardé dans : {model_save_path.replace('.joblib', '_scaler.joblib')}")
    print(f"Encodeur de famille sauvegardé dans : {encoder_save_path}")

if __name__ == '__main__':
    CSV_DATA_PATH = "../data/fingerprints_for_classification.csv"
    train_family_classifier(CSV_DATA_PATH)