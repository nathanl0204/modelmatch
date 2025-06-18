import os
import json
import time
import together
from tqdm import tqdm

def evaluate_clarity():
    """
    Évalue la clarté de chaque prompt utilisateur dans le dataset qui n'a pas encore été évalué.
    Utilise l'API Together pour classifier la clarté en quatre catégories :
    "clear", "acceptable", "ambiguous", "unclear".
    Les résultats sont ensuite sauvegardés directement dans le fichier JSON du dataset.
    """
    api_key = "547ae2cf346fd48fdb202b8f0ccb913ae67a104231ad98132206ecadbc04ad2a"
    if not api_key:
        print("Erreur : La variable d'environnement TOGETHER_API_KEY n'est pas définie.")
        print("Veuillez la définir avant de lancer le script.")
        return
    
    client = together.Together(api_key=api_key)

    dataset_path = 'modelmatch_dataset_reduced_no_change.json'
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

    system_prompt = (
        'You are an expert AI assistant specializing in prompt engineering. Your task is to evaluate the clarity of a user prompt. '
        'Classify the following user prompt into one of four categories: "clear", "acceptable", "ambiguous", or "unclear".\n\n'
        '- "clear": The prompt is specific, unambiguous, and provides all necessary context for the AI to generate a relevant response. '
        'Example: "Write a Python function that takes a list of integers and returns their sum."\n'
        '- "acceptable": The prompt is generally understandable but may lack some minor context, be slightly informal, or require the AI to make reasonable assumptions. '
        'Example: "Can you summarize the main points of the last text?"\n'
        '- "ambiguous": The prompt is vague, lacks critical information, or could be interpreted in multiple ways, making it difficult to generate a specific response. '
        'Example: "Fix this." or "Tell me more."\n'
        '- "unclear": The prompt is grammatically incorrect, nonsensical, or its intent is impossible to understand. '
        'Example: "how code python do?"\n\n'
        'IMPORTANT: You must respond with a JSON object containing a single key "clarity" whose value is ONE of the following strings: "clear", "acceptable", "ambiguous", "unclear". '
        'Example response: {"clarity": "clear"}'
    )

    print(f"Chargement du dataset depuis {dataset_path}...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {dataset_path} n'a pas été trouvé.")
        return
    except json.JSONDecodeError:
        print(f"Erreur : Impossible de décoder le JSON du fichier {dataset_path}.")
        return
    
    prompts_to_evaluate = []
    for conversation in data.get('conversations', []):
        for prompt in conversation.get('user_prompts', []):
            if prompt.get('clarity') is None:
                prompts_to_evaluate.append(prompt)
    
    if not prompts_to_evaluate:
        print("Aucun prompt à évaluer (le champ 'clarity' est déjà rempli pour tous les prompts).")
        return
    
    print(f"Début de l'évaluation de la clarté de {len(prompts_to_evaluate)} prompts...")

    valid_labels = {"clear", "acceptable", "ambiguous", "unclear"}

    with tqdm(total=len(prompts_to_evaluate), desc="Évaluation de la clarté") as pbar:
        for prompt in prompts_to_evaluate:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt.get('text', '')}
                    ],
                    max_tokens=30,
                    temperature=0.0,
                    response_format={"type": "json_object"} # Assure une réponse en format JSON
                )

                response_content = response.choices[0].message.content
                try:
                    clarity_data = json.loads(response_content)
                    clarity_label = clarity_data.get("clarity", "").strip().lower()

                    if clarity_label in valid_labels:
                        prompt['clarity'] = clarity_label
                    else:
                        prompt['clarity'] = 'unclear' # Valeur par défaut en cas de label invalide
                        pbar.write(f"Avertissement : Label JSON invalide reçu '{clarity_label}'. Assignation de 'unclear'.")

                except json.JSONDecodeError:
                    prompt['clarity'] = 'unclear'
                    pbar.write(f"Avertissement : Réponse JSON malformée reçue '{response_content[:50]}...'. Assignation de 'unclear'.")
                    
            except Exception as e:
                pbar.write(f"Une erreur est survenue pour le prompt: {prompt.get('text', '')[:50]}... Erreur: {e}")
                prompt['clarity'] = 'evaluation_error'
                time.sleep(1) # Pause pour éviter de surcharger l'API en cas d'erreurs répétées
            
            pbar.update(1)
    
    print("\nÉvaluation terminée. Sauvegarde des résultats dans le fichier...")
    try:
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Le fichier {dataset_path} a été mis à jour avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier : {e}")

if __name__ == '__main__':
    evaluate_clarity()
