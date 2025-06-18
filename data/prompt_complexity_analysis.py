import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_prompt_complexity_distribution(dataset_path):
    """
    Charge un dataset, analyse la complexité des prompts à chaque tour de conversation
    et génère un boxplot pour visualiser la distribution de la complexité
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
    elif isinstance(loaded_data, dict) and "conversations" in loaded_data:
        actual_conversations_list = loaded_data["conversations"]
    else:
        print(f"Erreur: Format de données non supporté dans {dataset_path}.")
        return
    
    if not actual_conversations_list:
        print("Aucune conversation trouvée dans le dataset.")
        return
    
    prompt_complexity_by_index = {}
    complexity_mapping = {"low": 1, "medium": 2, "high": 3}

    for conv in actual_conversations_list:
        if not isinstance(conv, dict):
            continue
        user_prompts = conv.get("user_prompts", [])
        for i, prompt in enumerate(user_prompts):
            complexity_str = prompt.get('complexity')
            if complexity_str and complexity_str in complexity_mapping:
                complexity_val = complexity_mapping[complexity_str]
                if i not in prompt_complexity_by_index:
                    prompt_complexity_by_index[i] = []
                prompt_complexity_by_index[i].append(complexity_val)
    
    if not prompt_complexity_by_index:
        print("Aucune donnée de complexité valide ('low', 'medium', 'high') n'a été trouvée dans le dataset.")
        return
    
    plot_data = []
    indices = sorted(prompt_complexity_by_index.keys())
    for index in indices:
        for complexity in prompt_complexity_by_index[index]:
            plot_data.append({'prompt_index': index + 1, 'complexity': complexity})
    
    df = pd.DataFrame(plot_data)

    if df.empty:
        print("Le DataFrame est vide, impossible de génére le graphique.")
        return
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='prompt_index', y='complexity', data=df, color='skyblue', showfliers=False)

    plt.title('Distribution de la complexité des prompts par index dans les conversations', fontsize=16)
    plt.xlabel('Index du prompt dans la conversation', fontsize=14)
    plt.ylabel('Complexité du prompt', fontsize=14)
    plt.yticks(list(complexity_mapping.values()), list(complexity_mapping.keys()))
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    output_path = "exploratory_analysis_results/prompt_complexity_boxplot.png"
    plt.savefig(output_path)
    print(f"Graphique de complexité sauvegardé sous : {output_path}")
    plt.show()

if __name__ == "__main__":
    dataset_file = "modelmatch_dataset_reduced_no_change.json"
    plot_prompt_complexity_distribution(dataset_file)