import argparse
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

def train_intra_family_classifier(
        csv_path: str, 
        models_to_include: list,
        model_save_path: str,
        encoder_save_path: str,
        report_save_path: str,
        confusion_matrix_save_path: str,
        classifier_name: str
    ):
    """
    Entraîne un classificateur XGBoost pour distinguer les modèles au sein d'une même famille.
    La fonction filtre le dataset pour n'inclure que les modèles spécifiés, optimise les
    hyperparamètres avec GridSearchCV, évalue le modèle, et sauvegarde le classificateur,
    le scaler, l'encodeur, un rapport de classification et une matrice de confusion.
    """
    print(f"\n--- Entraînement du classificateur intra-famille : {classifier_name} ---")
    print(f"Chargement des données depuis {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {csv_path} n'a pas été trouvé.")
        return
    
    print("Préparation et filtrage des données...")
    df.fillna(0, inplace=True)

    df_filtered = df[df['model_name'].isin(models_to_include)].copy()

    if df_filtered.empty or len(df_filtered['model_name'].unique()) < 2:
        print(f"Le DataFrame filtré est vide ou contient moins de 2 classes pour '{classifier_name}'. Vérifiez les noms des modèles.")
        return
    
    print(f"Nombre d'échantillons conservés : {len(df_filtered)}")
    print(f"Distribution des classes pour '{classifier_name}':")
    print(df_filtered['model_name'].value_counts())

    features_to_drop = ['model_name', 'conversation_id', 'prompt_index']
    X = df_filtered.drop(columns=[col for col in features_to_drop if col in df_filtered.columns])
    y = df_filtered['model_name']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nOptimisation des hyperparamètres pour XGBClassifier ({classifier_name})...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_estimator = XGBClassifier(random_state=42, eval_metric='mlogloss')

    n_splits = 3

    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits),
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    print(f"\nEntraînement avec GridSearchCV pour '{classifier_name}' (n_jobs=-1)...")
    grid_search.fit(X_train_scaled, y_train)

    print("Entraînement terminé.")

    print("\nMeilleurs hyperparamètres trouvés :")
    print(grid_search.best_params_)
    classifier = grid_search.best_estimator_

    print("\nÉvaluation du modèle sur l'ensemble de test...")
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrécision (Accuracy) pour '{classifier_name}': {accuracy:.4f}")

    print("\nRapport de classification :")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print(report)

    with open(report_save_path, 'w', encoding='utf-8') as f:
        f.write(f"Classificateur intra-famille : {classifier_name}\n")
        f.write(f"Précision (Accuracy) : {accuracy:.4f}\n")
        f.write(f"Meilleurs hyperparamètres : {grid_search.best_params_}\n\n")
        f.write(report)
    print(f"Rapport de classification sauvegardé dans {report_save_path}")

    print(f"\nGénération de la matrice de confusion ({classifier_name})...")
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_labels = (np.asarray(["{0:d}\n({1:.1%})".format(value, cm_normalized[i, j])
                                for i, row in enumerate(cm)
                                for j, value in enumerate(row)])
                    ).reshape(cm.shape)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Matrice de confusion — {classifier_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(confusion_matrix_save_path, dpi=150)
    plt.close()
    print(f"Matrice de confusion sauvegardée dans {confusion_matrix_save_path}")

    print("\nSauvegarde du modèle, du scaler et de l'encodeur...")
 
    model_dir = os.path.dirname(model_save_path)
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(classifier, model_save_path)
    joblib.dump(scaler, model_save_path.replace(".joblib", "_scaler.joblib"))
    joblib.dump(label_encoder, encoder_save_path)
    print(f"Modèle '{classifier_name}' sauvegardé dans : {model_save_path}")

def main():
    """
    Fonction principale pour orchestrer l'entraînement des classificateurs intra-famille.
    Définit les groupes de modèles à comparer et appelle la fonction d'entraînement pour chacun.
    """
    parser = argparse.ArgumentParser(description="Entraîne les classificateurs intra-famille.")
    parser.add_argument(
        "--csv",
        default="../data/fingerprints_for_classification.csv",
        help="Chemin du CSV d'empreintes à utiliser pour l'entraînement.",
    )
    args = parser.parse_args()

    CSV_PATH = args.csv
    os.makedirs("training_results", exist_ok=True)

    # Classificateur intra-famille Gemma (3 modèles)
    gemma_models = [
        'google/gemma-7b-it',
        'unsloth/gemma-7b-it-bnb-4bit',
        'gemma-4-31b-it',
    ]
    train_intra_family_classifier(
        csv_path=CSV_PATH,
        models_to_include=gemma_models,
        model_save_path="../models/gemma_quantization_classifier.joblib",
        encoder_save_path="../models/gemma_quantization_encoder.joblib",
        report_save_path="training_results/gemma_intra_family_report.txt",
        confusion_matrix_save_path="training_results/gemma_intra_family_confusion_matrix.png",
        classifier_name="Gemma (7B vs 7B-4bit vs Gemma-4-31B)"
    )
    
    # Classificateur intra-famille Qwen (3 modèles)
    qwen_models = [
        'Qwen/Qwen2-7B-Instruct',
        'RedHatAI/Qwen2-7B-Instruct-quantized.w8a16',
        'qwen3.8-27b',
    ]
    train_intra_family_classifier(
        csv_path=CSV_PATH,
        models_to_include=qwen_models,
        model_save_path="../models/qwen2_quantization_classifier.joblib",
        encoder_save_path="../models/qwen2_quantization_encoder.joblib",
        report_save_path="training_results/qwen2_intra_family_report.txt",
        confusion_matrix_save_path="training_results/qwen2_intra_family_confusion_matrix.png",
        classifier_name="Qwen (2-7B vs 2-7B-w8a16 vs 3.8-27B)"
    )

    # Classificateur intra-famille Llama-3 (2 modèles)
    llama_models = [
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'elinas/Llama-3-13B-Instruct'
    ]
    train_intra_family_classifier(
        csv_path=CSV_PATH,
        models_to_include=llama_models,
        model_save_path="../models/parameter_variant_classifier.joblib",
        encoder_save_path="../models/parameter_variant_encoder.joblib",
        report_save_path="training_results/llama3_intra_family_report.txt",
        confusion_matrix_save_path="training_results/llama3_intra_family_confusion_matrix.png",
        classifier_name="Llama-3 (8B vs 13B)"
    )

    # Classificateur intra-famille GPT (2 modèles)
    gpt_models = [
        'gpt-4o',
        'openai/gpt-5.6-luna',
    ]
    train_intra_family_classifier(
        csv_path=CSV_PATH,
        models_to_include=gpt_models,
        model_save_path="../models/gpt_variant_classifier.joblib",
        encoder_save_path="../models/gpt_variant_encoder.joblib",
        report_save_path="training_results/gpt_intra_family_report.txt",
        confusion_matrix_save_path="training_results/gpt_intra_family_confusion_matrix.png",
        classifier_name="GPT (4o vs 5.6-luna)"
    )

if __name__ == "__main__":
    main()
