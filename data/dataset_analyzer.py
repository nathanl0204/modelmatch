import json
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

class DatasetAnalyzer:
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.conversations = self.data['conversations']
    
    def detailed_analysis(self):
        """
        Effectue une analyse détaillée du dataset.
        """
        print("=== ANALYSE DÉTAILLÉE DU DATASET ===\n")

        # Informations générales
        print(f"Nombre total de conversations: {len(self.conversations)}")
        print(f"Conversations avec changement de modèle: {sum(1 for c in self.conversations if c['has_model_change'])}")
        print(f"Pourcentage avec changement de modèle: {sum(1 for c in self.conversations if c['has_model_change'])/len(self.conversations)*100:.1f}%\n")

        # Distribution des longueurs de conversation
        lengths = [len(c['user_prompts']) for c in self.conversations]
        print(f"Longueur moyenne des conversations: {sum(lengths)/len(lengths):.1f} prompts")
        print(f"Longueur min/max: {min(lengths)}/{max(lengths)} prompts")

        # Distribution des modèles
        models = [f"{c['target_model_name']}:{c['target_model_version']}" for c in self.conversations]
        model_counts = Counter(models)
        print(f"\nDistribution des modèles cibles:")
        for model, count in model_counts.most_common():
            print(f"  {model}: {count} ({count/len(self.conversations)*100:.1f}%)")
        
        # Distribution des quantifications
        quant_counts = Counter(c['quantization_type'] or 'none' for c in self.conversations)
        print(f"\nDistribution des quantifications:")
        for quant, count in quant_counts.items():
            print(f"  {quant}: {count} ({count/len(self.conversations)*100:.1f}%)")
        
        # Analyse des changements de modèle
        changes = [c for c in self.conversations if c['has_model_change']]
        if changes:
            # Filtrer les indices None
            change_indices = [c['model_change_index'] for c in changes if c['model_change_index'] is not None]
            if change_indices:
                print(f"\nIndex moyen des changements de modèle: {sum(change_indices)/len(change_indices):.1f}")
            else:
                print(f"\nAucun index de changement valide trouvé")
        
    def validate_dataset(self):
        """
        Valide la structure et la cohérence du dataset.
        """
        print("\n=== VALIDATION DU DATASET ===\n")

        errors = []
        warnings = []

        for i, conv in enumerate(self.conversations):
            # Vérifications obligatoires
            required_fields = ['id', 'user_prompts', 'target_model_name', 'target_model_version',
                               'has_model_change', 'model_change_index']
            
            for field in required_fields:
                if field not in conv:
                    errors.append(f"Conversation {i}: Champ manquant '{field}")
            
            # Vérification des prompts utilisateur
            if len(conv.get('user_prompts', [])) < 2:
                warnings.append(f"Conversation {i}: Moins de 2 prompts utilisateur")
            
            # Vérification de la cohérence des changements de modèle
            if conv.get('has_model_change'):
                if conv.get('model_change_index') is None:
                    errors.append(f"Conversation {i}: has_model_change=True mais model_change_index=None")
                elif conv.get('model_change_index', 0) >= len(conv.get('user_prompts', [])):
                    errors.append(f"Conversation {i}: model_change_index hors limites")
            else:
                if conv.get('model_change_index') is not None:
                    warnings.append(f"Conversation {i}: has_model_change=False mais model_change_index défini")
        
        if errors:
            print("ERREURS TROUVÉES:")
            for error in errors:
                print(f"  ❌ {error}")
        else:
            print("✅ Aucune erreur structurelle trouvée")
        
        if warnings:
            print(f"\nAVERTISSEMENTS ({len(warnings)}):")
            for warning in warnings[:5]: # limiter l'affichage
                print(f"  ⚠️ {warning}")
            if len(warnings) > 5:
                print(f"  ... et {len(warnings)-5} autres avertissements")
        else:
            print("✅ Aucun avertissement")
        
    def export_summary(self, output_file: str = "dataset_summary.txt"):
        """
        Exporte un résumé du dataset.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RÉSUMÉ DU DATASET MODELMATCH\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Nombre total de conversations: {len(self.conversations)}\n")
            f.write(f"Source: {self.data['dataset_info']['source_datasets']}\n\n")

            # Exemples de conversations
            f.write("EXEMPLES DE CONVERSATIONS:\n")
            f.write("-" * 25 + "\n\n")

            for i, conv in enumerate(self.conversations[:3]):
                f.write(f"Conversation {i+1}:\n")
                f.write(f"  Modèle: {conv['target_model_name']}:{conv['target_model_version']}\n")
                f.write(f"  Quantification: {conv['quantization_type']}\n")
                f.write(f"  Changement: {conv['has_model_change']}\n")
                f.write(f"  Prompts ({len(conv['user_prompts'])}):\n")
                for j, prompt in enumerate(conv['user_prompts'][:2]):
                    f.write(f"    {j+1}. {prompt[:80]}...\n")
                f.write("\n")
        
        print(f"Résumé exporté dans {output_file}")

def main():
    # Analyser le dataset créé
    analyzer = DatasetAnalyzer("modelmatch_dataset.json")
    analyzer.detailed_analysis()
    analyzer.validate_dataset()
    analyzer.export_summary()

if __name__ == "__main__":
    main()