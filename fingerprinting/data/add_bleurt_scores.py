import pandas as pd
import json
from tqdm import tqdm
from bleurt import score as bleurt_scorer
import os

FINGERPRINTS_CSV_PATH = "fingerprints_for_classification_extended.csv"
OUTPUT_CSV_PATH = "fingerprints_with_bleurt.csv"
DATASET_PATH = "modelmatch_dataset_reduced_no_change.json"
BLEURT_CHECKPOINT = "BLEURT-20"

def load_fingerprints(filepath: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Erreur: Fichier d'empreintes non trouvé à '{filepath}'")
        return None

def load_responses_dataset(filepath: str) -> dict | None:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erreur: Fichier dataset non trouvé à '{filepath}'")
        return None

def main():
    script_dir = os.path.dirname(__file__)
    fingerprints_path = os.path.join(script_dir, FINGERPRINTS_CSV_PATH)
    output_path = os.path.join(script_dir, OUTPUT_CSV_PATH)
    dataset_path = os.path.join(script_dir, DATASET_PATH)

    df = load_fingerprints(fingerprints_path)
    if df is None:
        return
    
    dataset = load_responses_dataset(dataset_path)
    if dataset is None or "conversations" not in dataset:
        return
    
    print("Préparation des données de référence pour le calcul BLEURT...")
    responses_map = {}
    for conv in tqdm(dataset["conversations"], desc="Indexation des réponses"):
        conv_id = conv.get("id")
        if "model_responses" in conv:
            for model_name, responses in conv["model_responses"].items():
                for i, response_text in enumerate(responses):
                    if (conv_id, i) not in responses_map:
                        responses_map[(conv_id, i)] = {}
                    responses_map[(conv_id, i)][model_name] = response_text
    
    quantized_to_original_map = {
        'unsloth/gemma-7b-it-bnb-4bit': 'google/gemma-7b-it',
        'RedHatAI/Qwen2-7B-Instruct-quantized.w8a16': 'Qwen/Qwen2-7B-Instruct'
    }

    print(f"Initialisation du scorer BLEURT avec le checkpoint '{BLEURT_CHECKPOINT}'...")
    scorer = bleurt_scorer.BleurtScorer(BLEURT_CHECKPOINT)

    bleurt_scores = []
    print("Calcul des scores BLEURT pour les modèles quantifiés...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calcul BLEURT"):
        model_name = row['model_name']
        conv_id = row['conversation_id']
        prompt_idx = row['prompt_index']

        if model_name in quantized_to_original_map:
            original_model = quantized_to_original_map[model_name]
            lookup_key = (conv_id, prompt_idx)

            if lookup_key in responses_map:
                references = [responses_map[lookup_key].get(original_model, "")]
                candidates = [responses_map[lookup_key].get(model_name, "")]

                if references[0] and candidates[0]:
                    score = scorer.score(references=references, candidates=candidates)[0]
                    bleurt_scores.append(score)
                else:
                    bleurt_scores.append(0.0)
            else:
                bleurt_scores.append(0.0)
        else:
            bleurt_scores.append(0.0)

    df['bleurt_score'] = bleurt_scores
    print("\nDistribution des scores BLEURT (non-nuls) :")
    print(df[df['bleurt_score'] != 0.0]['bleurt_score'].describe())

    print(f"\nSauvegarde du nouveau dataset d'empreintes dans '{output_path}'...")
    df.to_csv(output_path, index=False)
    print("Terminé.")

if __name__ == "__main__":
    main()