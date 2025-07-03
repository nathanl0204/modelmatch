import os
import json
from tqdm import tqdm
import together

def evaluate_complexity():
    """
    Évalue la complexité de chaque prompt utilisateur dans le dataset qui n'a pas encore été évalue.
    Utilise l'API Together pour classifier la complexité en trois catégories : "low", "medium", "high".
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
        'You are an expert AI assistant whose role is to evaluate the complexity of tasks given in user prompts. '
        'Classify the following user prompt into one of three categories: "low", "medium", or "high".\n\n'
        '- "low": Simple requests for facts, short definitions, or straightforward commands. '
        'Example: "What is the capital of France?"\n'
        '- "medium": Requests that require some reasoning, summarization, translation, or formatting. '
        'Example: "Summarize the main plot points of Hamlet in three paragraphs."\n'
        '- "high": Requests that require creativity, deep analysis, code generation, problem-solving, '
        'or synthesis of multiple concepts. '
        'Example: "Write a Python script to perform a sentiment analysis on a CSV file and generate a report."\n\n'
        'Only respond with one word: "low", "medium", or "high".'
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
            if prompt.get('complexity') is None:
                prompts_to_evaluate.append(prompt)

    if not prompts_to_evaluate:
        print("Aucun prompt à évaluer (le champ 'complexity' est déjà rempli pour tous les prompts).")
        return
    
    print(f"Début de l'évaluation de {len(prompts_to_evaluate)} prompts...")

    with tqdm(total=len(prompts_to_evaluate), desc="Évaluation de la complexité") as pbar:
        for prompt in prompts_to_evaluate:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt.get('text', '')}
                    ],
                    max_tokens=10,
                    temperature=0.0,
                )

                result = response.choices[0].message.content.strip().lower()

                if result in ["low", "medium", "high"]:
                    prompt['complexity'] = result
                else:
                    prompt['complexity'] = "unknown_response" # Si la réponse n'est pas conforme
            
            except Exception as e:
                print(f"\nUne erreur est survenue lors de l'appel à l'API : {e}")
                prompt['complexity'] = "error"

            pbar.update(1)

    print("\nÉvaluation terminée. Sauvegarde du dataset mis à jour...")
    try:
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"le fichier {dataset_path} a été mis à jour avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier : {e}")


if __name__ == '__main__':
    evaluate_complexity()
