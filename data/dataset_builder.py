import json
import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import uuid

class ModelMatchDataset:
    def __init__(self):
        self.conversations = []
        self.models_pool = [
            {"name": "gpt-3.5-turbo", "version": "0613", "quantization": None},
            {"name": "gpt-4", "version": "0314", "quantization": None},
            {"name": "claude-3-sonnet", "version": "20240229", "quantization": None},
            {"name": "llama-2-7b-chat", "version": "hf", "quantization": None},
            {"name": "llama-2-13b-chat", "version": "hf", "quantization": None},
            {"name": "mistral-7b-instruct", "version": "v0.1", "quantization": None},
            # Versions quantifiées
            {"name": "llama-2-7b-chat", "version": "hf", "quantization": "4bit"},
            {"name": "llama-2-7b-chat", "version": "hf", "quantization": "8bit"},
            {"name": "mistral-7b-instruct", "version": "v0.1", "quantization": "4bit"},
        ]
    
    def process_no_robots_dataset(self) -> None:
        """
        Traite le dataset no_robots en gardant seulement les conversations 
        de catégorie 'Chat' et en supprimant les instructions système.
        """
        print("Chargement du dataset no_robots...")
        dataset = load_dataset("HuggingFaceH4/no_robots")

        for item in dataset['train']:
            if item['category'] == 'Chat':
                conversation = self._process_no_robots_conversation(item)
                if conversation:
                    self.conversations.append(conversation)

        print(f"Nombre de conversations Chat extraites: {len(self.conversations)}")

    def _process_no_robots_conversation(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Traite une conversation individuelle du dataset no_robots.
        """
        messages = item['messages']
        user_prompts = []

        for message in messages:
            # Ne garder que les messages utilisateur
            if message['role'] == 'user':
                user_prompts.append(message['content'])
        
        # Ne garder que les conversations avec au moins 2 prompts utilisateur
        if len(user_prompts) < 2:
            return None
        
        # Générer les métadonnées de la conversation
        target_model = random.choice(self.models_pool)

        # 30 % de chance qu'il y ait un changement de modèle
        has_model_change = random.random() < 0.3
        model_change_index = None

        if has_model_change and len(user_prompts) > 2:
            # Le changement se fait après au moins le 2ème prompt
            model_change_index = random.randint(2, len(user_prompts) - 1)
        elif has_model_change:
            # Si on a exactement 2 prompts, pas de changement possible
            has_model_change = False
        
        conversation_data = {
            "id": str(uuid.uuid4()),
            "source_dataset": "no_robots",
            "user_prompts": user_prompts,
            "target_model_name": target_model["name"],
            "target_model_version": target_model["version"],
            "quantization_type": target_model["quantization"],
            "has_model_change": has_model_change,
            "model_change_index": model_change_index,
            "theme": None, # À remplir plus tard avec classification automatique
            "metadata": {
                "original_category": item['category'],
                "conversation_length": len(user_prompts)
            }
        }

        return conversation_data
    
    def save_dataset(self, filename: str = "modelmatch_dataset.json") -> None:
        """
        Sauvegarde le dataset au format JSON.
        """
        output_data = {
            "dataset_info": {
                "name": "ModelMatch Verification Dataset",
                "version": "1.0",
                "description": "Dataset pour tester le plugin ModelMatch",
                "total_conversations": len(self.conversations),
                "source_datasets": ["HuggingFaceH4/no_robots"]
            },
            "conversations": self.conversations
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset sauvegardé dans {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le dataset créé.
        """
        if not self.conversations:
            return {"error": "Aucune conversation dans le dataset"}
        
        stats = {
            "total_conversations": len(self.conversations),
            "conversations_with_model_change": sum(1 for c in self.conversations if c["has_model_change"]),
            "average_prompts_per_conversation": sum(len(c["user_prompts"]) for c in self.conversations) / len(self.conversations),
            "models_distribution": {},
            "quantization_distribution": {}
        }

        # Distribution des modèles
        for conv in self.conversations:
            model_key = f"{conv['target_model_name']}:{conv['target_model_version']}"
            stats["models_distribution"][model_key] = stats["models_distribution"].get(model_key, 0) + 1

            quant = conv["quantization_type"] or "none"
            stats["quantization_distribution"][quant] = stats["quantization_distribution"].get(quant, 0) + 1
        
        return stats
    
    def preview_conversations(self, n: int = 3) -> None:
        """
        Affiche un aperçu de n conversations.
        """
        print(f"\n=== Aperçu de {min(n, len(self.conversations))} conversations ===")

        for i, conv in enumerate(self.conversations[:n]):
            print(f"\n--- Conversation {i+1} ---")
            print(f"ID: {conv['id']}")
            print(f"Modèle cible: {conv['target_model_name']}:{conv['target_model_version']}")
            print(f"Quantification: {conv['quantization_type']}")
            print(f"Changement de modèle: {conv['has_model_change']}")
            if conv['has_model_change']:
                print(f"Index du changement: {conv['model_change_index']}")
            print(f"Nombre de prompts: {len(conv['user_prompts'])}")
            print("Premiers prompts:")
            for j, prompt in enumerate(conv['user_prompts'][:3]):
                print(f"  {j+1}. {prompt[:100]}...")
    
def main():
    # Créer le dataset
    builder = ModelMatchDataset()

    # Traiter le dataset no_robots
    builder.process_no_robots_dataset()

    # Afficher les statistiques
    stats = builder.get_statistics()
    print("\n=== Statistiques du dataset ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # Aperçu des conversations
    builder.preview_conversations()

    # Sauvegarder
    builder.save_dataset()

if __name__ == "__main__":
    main()