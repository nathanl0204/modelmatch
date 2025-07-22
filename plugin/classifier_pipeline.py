import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ajoute le répertoire src du module fingerprinting au path système pour permettre l'importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../fingerprinting/src')))
try:
    from fingerprinter import Fingerprinter, get_text_metrics, get_code_metrics, get_model_family
except ImportError:
    print("Erreur: Impossible d'importer depuis fingerprinter.py. Assurez-vous que le chemin est correct.")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "../fingerprinting/models/")
FAMILY_CONFIDENCE_THRESHOLD = 0.4564

# Mapping pour faire correspondre le couple (famille, variante) à un nom de modèle exact
MODEL_MAPPING = {
    ("Gemma", "Original"): "google/gemma-7b-it",
    ("Gemma", "Variante quantifiée"): "unsloth/gemma-7b-it-bnb-4bit",
    ("Qwen2", "Original"): "Qwen/Qwen2-7B-Instruct",
    ("Qwen2", "Variante quantifiée"): "RedHatAI/Qwen2-7B-Instruct-quantized.w8a16",
    ("Llama-3", "Original"): "meta-llama/Meta-Llama-3-8B-Instruct",
    ("Llama-3", "Variante de paramètres"): "elinas/Llama-3-13B-Instruct",
    # Modèles sans classificateur de variante, la famille est le nom du modèle
    ("Deepseek", "Original / Inconnu"): "deepseek-ai/deepseek-llm-7b-chat",
    ("GPT-4o", "Original / Inconnu"): "gpt-4o",
    ("Phi-3", "Original / Inconnu"): "microsoft/Phi-3-mini-128k-instruct",
    ("Mistral", "Original / Inconnu"): "mistralai/Mistral-7B-Instruct-v0.1"
}

class ClassifierPipeline:
    """
    Orchestre le processus de classification hiérarchique pour identifier un modèle de langage.
    Charge tous les classificateurs nécessaires (famille, binaire, variante) et exécute
    la séquence de prédiction pour déterminer la famille et le type de variante d'un modèle.
    """
    def __init__(self):
        """Initialise le pipeline en chargeant tous les assets et le fingerprinter."""
        self.assets = self._load_all_assets()
        self.fingerprinter = Fingerprinter(
            text_metrics=get_text_metrics(),
            code_metrics=get_code_metrics()
        )
    
    def get_metric_names(self):
        """Retourne la liste des noms de toutes les métriques disponibles."""
        return self.fingerprinter.get_all_metric_names()
    
    def _load_asset(self, path):
        """Charge un asset sérialisé (modèle, scaler, encodeur) depuis un fichier."""
        try:
            return joblib.load(path)
        except FileNotFoundError:
            print(f"Avertissement: Fichier non trouvé à l'adresse {path}, l'asset ne sera pas chargé.")
            return None
    
    def _load_all_assets(self):
        """Charge tous les classificateurs, scalers et encodeurs nécessaires pour la pipeline."""
        print("Chargement des modèles et des assets de classification...")
        assets = {
            "family_clf": self._load_asset(os.path.join(MODEL_DIR, "family_classifier.joblib")),
            "family_scaler": self._load_asset(os.path.join(MODEL_DIR, "family_classifier_scaler.joblib")),
            "family_encoder": self._load_asset(os.path.join(MODEL_DIR, "family_name_encoder.joblib")),

            "binary_deepseek_mistral": self._load_binary_assets("Deepseek", "Mistral"),
            "binary_gpt-4o_qwen2": self._load_binary_assets("GPT-4o", "Qwen2"),
            "binary_mistral_llama-3": self._load_binary_assets("Mistral", "Llama-3"),
            "binary_phi-3_qwen2": self._load_binary_assets("Phi-3", "Qwen2"),

            "gemma_quantization_classifier": self._load_variant_assets("gemma_quantization_classifier"),
            "qwen2_quantization_classifier": self._load_variant_assets("qwen2_quantization_classifier"),
            "parameter_variant_classifier": self._load_variant_assets("parameter_variant_classifier"),
        }
        print("Chargement terminé.")
        return assets
    
    def _load_binary_assets(self, family1, family2):
        """Charge les assets pour un classificateur binaire spécifique, en gérant les ordres de noms."""
        f1_l, f2_l = family1.lower(), family2.lower()
        
        possible_names = [
            f"binary_{f1_l}_vs_{f2_l}",
            f"binary_{f2_l}_vs_{f1_l}"
        ]

        for model_name in possible_names:
            path_prefix = os.path.join(MODEL_DIR, "binary_family_classifiers", model_name)
            clf = self._load_asset(f"{path_prefix}_classifier.joblib")
            if clf:
                return {
                    "clf": clf,
                    "scaler": self._load_asset(f"{path_prefix}_scaler.joblib"),
                    "encoder": self._load_asset(f"{path_prefix}_encoder.joblib")
                }
            
        return {"clf": None, "scaler": None, "encoder": None}
    
    def _load_variant_assets(self, classifier_name):
        """Charge les assets pour un classificateur de variante (quantification, taille)."""
        encoder_name = classifier_name.replace('_classifier', '_encoder')
        return {
            "clf": self._load_asset(os.path.join(MODEL_DIR, f"{classifier_name}.joblib")),
            "scaler": self._load_asset(os.path.join(MODEL_DIR, f"{classifier_name}_scaler.joblib")),
            "encoder": self._load_asset(os.path.join(MODEL_DIR, f"{encoder_name}.joblib"))
        }
    
    def _create_fingerprints(self, conversation_history):
        """Génère les empreintes stylistiques pour chaque réponse dans l'historique."""
        fingerprints = []
        for i, turn in enumerate(conversation_history):
            fp = self.fingerprinter.create_fingerprint(
                text=turn['response'],
                prompt_text=turn['prompt']
            )
            fp['conversation_id'] = 'live_session'
            fp['prompt_index'] = i
            fingerprints.append(fp)
        
        df = pd.DataFrame(fingerprints)
        if self.assets['family_scaler']:
            df = df.reindex(columns=self.assets['family_scaler'].feature_names_in_, fill_value=0)
        return df
    
    def _run_classification(self, df, clf, scaler, encoder):
        """Exécute une classification sur un DataFrame d'empreintes avec les assets fournis."""
        if df.empty or not all([clf, scaler, encoder]):
            return [], []
        X_scaled = scaler.transform(df)
        predictions = clf.predict(X_scaled)
        probabilities = clf.predict_proba(X_scaled)
        predicted_labels = encoder.inverse_transform(predictions)
        return predicted_labels, probabilities
    
    def verify(self, conversation_history):
        """
        Exécute le pipeline de classification hiérarchique complet.
        1. Classifie la famille du modèle.
        2. Si la confiance est faible, utilise un classificateur binaire pour départager.
        3. Si applicable, classifie le type de variante (quantification, taille).
        """
        if len(conversation_history) < 1:
            return "Pas assez d'échanges pour une vérification.", "N/A"
        
        fingerprints_df = self._create_fingerprints(conversation_history)

        # --- NIVEAU 1: Classification par famille ---
        family_preds, family_probs = self._run_classification(
            fingerprints_df, self.assets['family_clf'], self.assets['family_scaler'], self.assets['family_encoder']
        )
        if not family_preds.any():
            return "Erreur: Classificateur de famille non disponible.", "N/A"
        
        final_family = pd.Series(family_preds).mode()[0]
        avg_max_prob = np.mean([prob.max() for prob in family_probs])

        # --- NIVEAU 2: Classification binaire de départage (si confiance faible) ---
        if avg_max_prob < FAMILY_CONFIDENCE_THRESHOLD:
            avg_probs_per_class = np.mean(family_probs, axis=0)
            top_two_indices = np.argsort(avg_probs_per_class)[-2:]
            family1 = self.assets['family_encoder'].classes_[top_two_indices[1]]
            family2 = self.assets['family_encoder'].classes_[top_two_indices[0]]

            binary_assets_key = f"binary_{'_'.join(sorted([family1.lower(), family2.lower()]))}"
            binary_assets = self.assets.get(binary_assets_key)

            if binary_assets and all(binary_assets.values()):
                binary_preds, _ = self._run_classification(fingerprints_df, **binary_assets)
                final_family = pd.Series(binary_preds).mode()[0]
        
        # --- NIVEAU 3: Détection de variante ---
        variant_type = "Original / Inconnu"
        variant_map = {
            "Gemma": "gemma_quantization_classifier",
            "Qwen2": "qwen2_quantization_classifier",
            "Llama-3": "parameter_variant_classifier"
        }

        if final_family in variant_map:
            classifier_name = variant_map[final_family]
            variant_assets = self.assets.get(classifier_name)
            if variant_assets and all(variant_assets.values()):
                variant_preds, _ = self._run_classification(fingerprints_df, **variant_assets)
                final_variant_name = pd.Series(variant_preds).mode()[0]

                if "4bit" in final_variant_name or "w8a16" in final_variant_name:
                    variant_type = "Variante quantifiée"
                elif "8B" in final_variant_name or "13B" in final_variant_name:
                    variant_type = "Variante de paramètres"
                else:
                    variant_type = "Original"
        
        predicted_model_name = MODEL_MAPPING.get((final_family, variant_type), f"{final_family} ({variant_type})")
        
        return predicted_model_name, f"Famille: {final_family}, Type: {variant_type}"