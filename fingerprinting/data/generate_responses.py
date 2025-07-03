import json
import os
from openai import OpenAI
from tqdm import tqdm

DATASET_PATH = 'modelmatch_dataset_reduced_no_change.json'
# Nom du modèle à utiliser pour la génération, tel que reconnu par l'API. Ce champ doit être modifié pour générer des réponses avec un autre modèle.
MODEL_TO_RUN = 'elinas/Llama-3-13B-Instruct'
# URL de base de l'API (ici, un serveur local via vLLM)
API_BASE_URL = "http://localhost:8000/v1"
# Clé API (non nécessaire pour l'API locale vLLM)
API_KEY = "not-needed"

def generate_responses_for_dataset():
    """
    Génère des réponses pour chaque prompt de chaque conversation dans le dataset
    en utilisant un modèle spécifique via une API compatible OpenAI.
    Le script est conçu pour être résilient : il sauvegarde la progression après chaque
    conversation et peut reprendre là où il s'est arrêté en cas d'erreur.
    """
    print(f"Chargement du dataset depuis {DATASET_PATH}...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier dataset '{DATASET_PATH}' n'a pas été trouvé.")
        return
    except json.JSONDecodeError:
        print(f"Erreur : Impossible de décoder le JSON depuis '{DATASET_PATH}'.")
        return
    
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    conversations = data.get('conversations', [])
    if not conversations:
        print("Aucune conversation trouvée dans le fichier dataset.")
        return
    
    print(f"Début de la génération des réponses pour le modèle : {MODEL_TO_RUN}")

    for conversation in tqdm(conversations, desc="Traitement des conversations"):
        if 'model_responses' not in conversation:
            conversation['model_responses'] = {}

        user_prompts = conversation.get('user_prompts', [])
        num_prompts = len(user_prompts)

        all_responses = conversation.get('model_responses', {}).get(MODEL_TO_RUN, [])
        start_index = 0
        # Cherche la première erreur pour reprendre juste avant
        for i, response in enumerate(all_responses):
            if response.startswith("ERREUR_API:"):
                start_index = i
                break
        else:
            # Si aucune erreur n'est trouvée, on reprend après la dernière réponse générée
            start_index = len(all_responses)
        
        if start_index == num_prompts:
            continue
        
        generated_responses = all_responses[:start_index]
        history_for_api = []

        # Reconstruit l'historique de la conversation jusqu'au point de reprise
        if start_index > 0:
            tqdm.write(f"\nReprise de la conversation {conversation.get('id')} à partir du prompt {start_index + 1}/{num_prompts}")
            for j in range(start_index):
                history_for_api.append({"role": "user", "content": user_prompts[j].get('text', '')})
                history_for_api.append({"role": "assistant", "content": generated_responses[j]})

        for prompt_info in user_prompts[start_index:]:
            prompt_text = prompt_info.get('text', '')

            current_turn_messages = history_for_api + [{"role": "user", "content": prompt_text}]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_TO_RUN,
                    messages=current_turn_messages,
                    temperature=0.7,
                    max_tokens=1500
                )
                response_text = completion.choices[0].message.content
            except Exception as e:
                response_text = f"ERREUR_API: {str(e)}"
                print(f"\nErreur lors de l'appel API pour la conversation {conversation.get('id')}")
            
            generated_responses.append(response_text)

            history_for_api.append({"role": "user", "content": prompt_text})
            history_for_api.append({"role": "assistant", "content": response_text})

        conversation['model_responses'][MODEL_TO_RUN] = generated_responses

        try:
            with open(DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tqdm.write(f"\nErreur lors de la sauvegarde de la progression : {e}")

    print("\nToutes les conversations ont été traitées.")

if __name__ == "__main__":
    generate_responses_for_dataset()
