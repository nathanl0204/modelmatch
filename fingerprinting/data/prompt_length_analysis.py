import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_prompt_length_distribution(dataset_path):
    """
    Charge un dataset, analyse la longueur des prompts à chaque tour de conversation
    et génère un boxplot pour visualiser la distribution de la longueur
    en fonction de l'index du prompt dans la conversation.
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {dataset_path} n'a pas été trouvé.")
        return
    except json.JSONDecodeError:
        print(f"Erreur: Impossible de décoder le JSON du fichier {dataset_path}.")
        return
    
    # Gère les différents formats de JSON (soit une liste de conversations, soit un dictionnaire les contenant)
    actual_conversations_list = []
    if isinstance(loaded_data, list):
        actual_conversations_list = loaded_data
    elif isinstance(loaded_data, dict):
        if "conversations" in loaded_data and isinstance(loaded_data["conversations"], list):
            actual_conversations_list = loaded_data["conversations"]
        else:
            keys_found = list(loaded_data.keys())
            print(f"Erreur: Le fichier JSON {dataset_path} est un dictionnaire mais ne contient pas de clé 'conversations' valide (liste) ou la clé est manquante. Clés trouvées: {keys_found}")
            return
    else:
        print(f"Erreur: Le format du dataset {dataset_path} n'est ni une liste ni un dictionnaire.")
        return
    
    if not actual_conversations_list:
        print("Aucune conversation trouvée dans le dataset.")
        return

    prompt_lengths_by_index = {}

    for conv in actual_conversations_list:
        if not isinstance(conv, dict):
            print(f"Avertissement: Élément inattendu ({type(conv)}) dans la liste des conversations, attendu: dict. Élément ignoré.")
            continue
        user_prompts = conv.get("user_prompts", [])
        for i, prompt in enumerate(user_prompts):
            prompt_len = len(prompt.get('text', ''))
            if i not in prompt_lengths_by_index:
                prompt_lengths_by_index[i] = []
            prompt_lengths_by_index[i].append(prompt_len)
    
    if not prompt_lengths_by_index:
        print("Aucun prompt trouvé dans le dataset.")
        return
    
    plot_data = []
    indices = sorted(prompt_lengths_by_index.keys())
    for index in indices:
        for length in prompt_lengths_by_index[index]:
            plot_data.append({'prompt_index': index + 1, 'length': length})
    
    df = pd.DataFrame(plot_data)

    if df.empty:
        print("Le DataFrame est vide, impossible de générer le graphique.")
        return
    
    plt.figure(figsize=(15, 8))

    sns.boxplot(x='prompt_index', y='length', data=df, color='skyblue', showfliers=False)

    plt.title('Distribution de la longueur des prompts par index dans les conversations', fontsize=16)
    plt.xlabel('Index du prompt dans la conversation', fontsize=14)
    plt.ylabel('Longueur du prompt (nombre de caractères)', fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    output_path = "exploratory_analysis_results/prompt_length_boxplot.png"
    plt.savefig(output_path)
    print(f"Graphique sauvegardé sous : {output_path}")
    plt.show()


if __name__ == "__main__":
    dataset_file = "modelmatch_dataset_reduced_no_change.json"
    plot_prompt_length_distribution(dataset_file)