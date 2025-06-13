import json
import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import uuid
import os
from together import Together

class ModelMatchDataset:
    def __init__(self):
        self.conversations = []
        """ self.models_pool = [
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
        ] """

        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set. Please set it to fetch models.")
        self.client = Together(api_key=api_key)

        try:
            print("Fetching models from Together API...")
            all_together_models = self.client.models.list()
            self.together_models_pool = [
                model for model in all_together_models if model.type == 'chat'
            ]
            if not self.together_models_pool:
                raise RuntimeError("No suitable 'chat' type models found from Together API. Dataset generation cannot proceed with API models.")
            print(f"Successfully fetched {len(self.together_models_pool)} chat models from Together API.")
            print("Example models from Together API:")
            for i, model in enumerate(self.together_models_pool[:min(3, len(self.together_models_pool))]):
                print(f"  - {model.id} (Type: {model.type})")
        except Exception as e:
            print(f"Error fetching or filtering models from Together API: {e}")
            raise RuntimeError(f"Could not initialize models from Together API: {e}")
    
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
        
        if not self.together_models_pool:
            raise RuntimeError("Together models pool is not initialized or empty.")
        
        # Générer les métadonnées de la conversation
        target_api_model = random.choice(self.together_models_pool)

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
            "target_model_name": target_api_model.id,
            "target_model_version": "N/A", # La version est souvent donnée dans l'ID des modèles API (attribut modifiable plus tard)
            "quantization_type": None, # Les modèles API ne sont généralement pas quantifiés (idem)
            "has_model_change": has_model_change,
            "model_change_index": model_change_index,
            "theme": None, # À remplir plus tard avec classification automatique
            "complexity": None, # Idem
            "clarity": None, # Idem
            "metadata": {
                "original_category": item['category'],
                "conversation_length": len(user_prompts)
            }
        }

        return conversation_data
    
    def process_puffin_dataset(self) -> None:
        """
        Traite le dataset Puffin en gardant seulement les conversations
        avec au moins 2 prompts utilisateur.
        """
        print("Chargement du dataset Puffin...")
        dataset = load_dataset("LDJnr/Puffin")

        for item in dataset['train']:
            conversation = self._process_puffin_conversation(item)
            if conversation:
                self.conversations.append(conversation)
        
        print(f"Nombre de conversations Puffin extraites: {len([c for c in self.conversations if c['source_dataset'] == 'puffin'])}")

    def _process_puffin_conversation(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Traite une conversation individuelle du dataset Puffin.
        """
        conversations = item['conversations']
        user_prompts = []

        for message in conversations:
            # Ne garder que les messages utilisateur (from: human)
            if message['from'] == 'human':
                user_prompts.append(message['value'])
        
        # Ne garder que les conversations avec au moins 2 prompts utilisateur
        if len(user_prompts) < 2:
            return None
        
        if not self.together_models_pool:
            raise RuntimeError("Together models pool is not initialized or empty.")

        # Générer les métadonnées de la conversation
        target_api_model = random.choice(self.together_models_pool)

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
            "source_dataset": "puffin",
            "user_prompts": user_prompts,
            "target_model_name": target_api_model.id,
            "target_model_version": "N/A", # La version est souvent donnée dans l'ID des modèles API (attribut modifiable plus tard)
            "quantization_type": None, # Les modèles API ne sont généralement pas quantifiés (idem)
            "has_model_change": has_model_change,
            "model_change_index": model_change_index,
            "theme": None, # À remplir plus tard avec classification automatique
            "complexity": None, # Idem
            "clarity": None, # Idem
            "metadata": {
                "original_source": item.get('source', 'unknown'),
                "conversation_length": len(user_prompts)
            }
        }

        return conversation_data
    
    def process_sharegpt_dataset(self) -> None:
        """
        traite le dataset ShareGPT en gardant seulement les conversations
        avec au moins 2 prompts utilisateur et commençant par un prompt utilisateur.
        """
        print("Chargement du dataset ShareGPT...")
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", data_files="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json")

        processed_count = 0
        for item in dataset['train']:
            conversation = self._process_sharegpt_conversation(item)
            if conversation:
                self.conversations.append(conversation)
                processed_count += 1
        
        print(f"Nombre de conversations ShareGPT extraites: {processed_count}")
    
    def _process_sharegpt_conversation(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Traite une conversation individuelle du dataset ShareGPT.
        """
        conversations = item.get('conversations', [])
        user_prompts = []

        # Vérifier que la conversation commence par un message utilisateur
        if not conversations or conversations[0].get('from') != 'human':
            return None
        
        for message in conversations:
            # Ne garder que les messages utilisateur
            if message.get('from') == 'human':
                user_prompts.append(message.get('value', ''))
        
        # Ne garder que les conversations avec au moins 2 prompts utilisateur
        if len(user_prompts) < 2:
            return None
        
        if not self.together_models_pool:
            raise RuntimeError("Together models pool is not initialized or empty.")
        
        # Générer les métadonnées de la conversation
        target_api_model = random.choice(self.models_pool)

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
            "source_dataset": "sharegpt",
            "user_prompts": user_prompts,
            "target_model_name": target_api_model.id,
            "target_model_version": "N/A", # La version est souvent donnée dans l'ID des modèles API (attribut modifiable plus tard)
            "quantization_type": None, # Les modèles API ne sont généralement pas quantifiés (idem)
            "has_model_change": has_model_change,
            "model_change_index": model_change_index,
            "theme": None, # À remplir plus tard avec classification automatique
            "complexity": None, # Idem
            "clarity": None, # Idem
            "metadata": {
                "original_id": item.get('id', 'unknown'),
                "conversation_length": len(user_prompts)
            }
        }

        return conversation_data
    
    def process_oasst_dataset(self) -> None:
        """
        Traite le dataset OpenAssistant en reconstituant les conversations
        à partir de la structure d'arbre et en gardant seulement celles en anglais.
        """
        print("Chargement du dataset OpenAssistant...")
        dataset = load_dataset("OpenAssistant/oasst1")

        # Créer un dictionnaire pour organiser les messages par conversation
        conversations_data = {}

        for item in dataset['train']:
            if item['lang'] == 'en': # Garder seulement les conversations en anglais
                conversation_id = item['message_tree_id']
                if conversation_id not in conversations_data:
                    conversations_data[conversation_id] = []
                conversations_data[conversation_id].append(item)
        
        processed_count = 0
        for conv_id, messages in conversations_data.items():
            conversation = self._process_oasst_conversation(messages)
            if conversation:
                self.conversations.append(conversation)
                processed_count += 1
        
        print(f"Nombre de conversations OpenAssistant extraites: {processed_count}")
    
    def _process_oasst_conversation(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Traite une conversation individuelle du dataset OpenAssistant en reconstituant
        l'arbre de conversation et en extrayant les prompts utilisateur.
        """
        # Organiser les messages par ID pour faciliter la reconstruction de l'arbre
        messages_by_id = {msg['message_id']: msg for msg in messages}

        # Trouver le message racine (sans parent)
        root_message = None
        for msg in messages:
            if msg['parent_id'] is None:
                root_message = msg
                break
        
        if not root_message or root_message['role'] != 'prompter':
            return None
        
        # Reconstituer la conversation en suivant le chemin principal
        user_prompts = []
        current_message = root_message

        while current_message:
            if current_message['role'] == 'prompter':
                user_prompts.append(current_message['text'])
            
            # Trouver le prochain message dans le fil de conversation
            # On prend le premier enfant qui est une réponse d'assistant, puis son premier enfant prompter
            next_message = None
            for msg in messages:
                if msg['parent_id'] == current_message['message_id']:
                    if current_message['role'] == 'prompter' and msg['role'] == 'assistant':
                        # Chercher le prochain message prompter après cette réponse
                        for next_msg in messages:
                            if next_msg['parent_id'] == msg['message_id'] and next_msg['role'] == 'prompter':
                                next_message = next_msg
                                break
                        break
            current_message = next_message
        
        # Ne garder que les conversations avec au moins 2 prompts utilisateur
        if len(user_prompts) < 2:
            return None
        
        if not self.together_models_pool:
            raise RuntimeError("Together models pool is not initialized or empty.")

        # Générer les métadonnées de la conversation
        target_api_model = random.choice(self.models_pool)

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
            "source_dataset": "oasst",
            "user_prompts": user_prompts,
            "target_model_name": target_api_model.id,
            "target_model_version": "N/A", # La version est souvent donnée dans l'ID des modèles API (attribut modifiable plus tard)
            "quantization_type": None, # Les modèles API ne sont généralement pas quantifiés (idem)
            "has_model_change": has_model_change,
            "model_change_index": model_change_index,
            "theme": None, # À remplir plus tard avec classification automatique
            "complexity": None, # Idem
            "clarity": None, # Idem
            "metadata": {
                "original_tree_id": root_message['message_tree_id'],
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
                "source_datasets": ["HuggingFaceH4/no_robots", "LDJnr/Puffin", "anon8231489123/ShareGPT_Vicuna_unfiltered", "OpenAssistant/oasst1"]
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
    builder.process_puffin_dataset()
    builder.process_sharegpt_dataset()
    builder.process_oasst_dataset()

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