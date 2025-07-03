import json
import random
from collections import Counter, defaultdict
import math
import os


# --- CONFIGURATION ---
# Nom du fichier du dataset d'entrée complet
INPUT_DATASET_FILENAME = "modelmatch_dataset.json"
# Nom du fichier du dataset de sortie réduit
OUTPUT_DATASET_FILENAME = "modelmatch_dataset_reduced_no_change.json"
# Nombre cible de conversations à inclure dans le dataset réduit
# (Il n'y en a que 499 car une conversation a été supprimée à cause de sa longueur et des problèmes de réponse que cela a entraîné)
TARGET_CONVERSATIONS = 500

def load_dataset(filepath: str) -> dict | None:
    """
    Charge un dataset depuis un fichier JSON.

    Args:
        filepath: Le chemin vers le fichier JSON.
    
    Returns:
        Un dictionnaire représentant le dataset, ou None en cas d'erreur.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
    return None

def save_dataset(data: dict, filepath: str) -> None:
    """
    Sauvegarde un dataset dans un fichier JSON.

    Args:
        data: Le dictionnaire à sauvegarder.
        filepath: Le chemin du fichier de sortie.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Dataset successfully saved to {filepath}")
    except Exception as e:
        print(f"Erro saving dataset to {filepath}: {e}")

def print_source_distribution(conversations: list, dataset_name: str):
    """
    Affiche la distribution des conversations par source de dataset.

    Args:
        conversations: La liste des conversations à analyser.
        dataset_name: Le nom du dataset pour l'affichage.
    """
    if not conversations:
        print(f"{dataset_name}: No conversations to analyze.")
        return
    source_counts = Counter(conv.get("source_dataset", "unknown") for conv in conversations)
    total = len(conversations)
    print(f"\nSource distribution in {dataset_name} (Total: {total}):")
    for source, count in sorted(source_counts.items()):
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  - {source}: {count} ({percentage:.2f}%)")

def main():
    """
    Fonction principale du script.
    Charge un grand dataset, le filtre pour ne garder que les conversations sans changement de modèle,
    puis crée un sous-ensemble plus petit tout en préservant la distribution des sources.
    """
    script_dir = os.path.dirname(__file__)
    input_dataset_path = os.path.join(script_dir, INPUT_DATASET_FILENAME)
    output_dataset_path = os.path.join(script_dir, OUTPUT_DATASET_FILENAME)

    full_dataset = load_dataset(input_dataset_path)
    if not full_dataset or "conversations" not in full_dataset:
        print(f"Could not load conversations from {input_dataset_path}")
        return
    
    all_conversations = full_dataset["conversations"]
    print(f"Loaded {len(all_conversations)} total conversations from {input_dataset_path}")
    print_source_distribution(all_conversations, "Original full dataset")

    no_change_conversations = [
        conv for conv in all_conversations if not conv.get("has_model_change", False)
    ]

    if not no_change_conversations:
        print("No conversations without model change found.")
        empty_reduced_dataset = {
            "dataset_info": {
                "name": "ModelMatch verification dataset - reduced and no model change",
                "version": "1.0-reduced-no-change-empty",
                "description": "Reduced dataset (0 conversations) - no 'no model change' conversations found in source.",
                "total_conversations": 0,
                "source_datasets": []
            },
            "conversations": []
        }
        save_dataset(empty_reduced_dataset, output_dataset_path)
        return
    
    num_available_no_change = len(no_change_conversations)
    print(f"\nFound {num_available_no_change} conversations with no model change.")
    print_source_distribution(no_change_conversations, "No model change subset")

    actual_target_conversations = min(TARGET_CONVERSATIONS, num_available_no_change)
    print(f"Targeting {actual_target_conversations} for the reduced dataset.")

    reduced_conversations: list
    if num_available_no_change <= actual_target_conversations:
        reduced_conversations = random.sample(no_change_conversations, num_available_no_change)
        print(f"Available 'no change' conversations ({num_available_no_change}) is less than or equal to target. Using all available.")
    else:
        convs_by_source = defaultdict(list)
        for conv in no_change_conversations:
            source = conv.get("source_dataset", "unknown")
            convs_by_source[source].append(conv)
        
        for source in convs_by_source:
            random.shuffle(convs_by_source[source])
        
        source_counts = {source: len(convs) for source, convs in convs_by_source.items()}

        quotas = {
            source: (count / num_available_no_change) * actual_target_conversations
            for source, count in source_counts.items()
        }
        allocations = {source: math.floor(q) for source, q in quotas.items()}
        remainders = {source: quotas[source] - allocations[source] for source in quotas}

        current_total_allocated = sum(allocations.values())
        slots_to_distribute_count = actual_target_conversations - current_total_allocated

        sorted_sources_for_remainders = sorted(
            quotas.keys(),
            key=lambda s: (remainders[s], source_counts[s], s),
            reverse=True
        )

        for i in range(int(round(slots_to_distribute_count))):
            source_to_increment = sorted_sources_for_remainders[i % len(sorted_sources_for_remainders)]
            if allocations[source_to_increment] < source_counts[source_to_increment]:
                allocations[source_to_increment] += 1
            else:
                print(f"Warning: Source {source_to_increment} is full but was due a remainder slot. Trying next.")
                for j in range(len(sorted_sources_for_remainders)):
                    alt_source = sorted_sources_for_remainders[(i + j) % len(sorted_sources_for_remainders)]
                    if allocations[alt_source] < source_counts[alt_source]:
                        allocations[alt_source] += 1
                        break
                else:
                    print("Could not re-allocate a remainder slot, target might be undershot.")
        
        final_allocated_sum = sum(allocations.values())
        deficit = actual_target_conversations - final_allocated_sum
        if deficit > 0:
            print(f"Adjusting: Need to add {deficit} more conversations.")
            for source in sorted_sources_for_remainders:
                while deficit > 0 and allocations[source] < source_counts[source]:
                    allocations[source] += 1
                    deficit -= 1
        elif deficit < 0:
            print(f"Adjusting: Need to remove {-deficit} conversations.")
            sorted_alloc_desc = sorted(allocations.key(), key=lambda s: allocations[s], reverse=True)
            for source in sorted_alloc_desc:
                while deficit < 0 and allocations[source] > math.floor(quotas[source]):
                    allocations[source] -= 1
                    deficit += 1
            for source in sorted_alloc_desc:
                while deficit < 0 and allocations[source] > 0:
                    allocations[source] -= 1
                    deficit += 1

        reduced_conversations = []
        for source, num_to_take in allocations.items():
            reduced_conversations.extend(convs_by_source[source][:int(num_to_take)])

        # Vérifications finales du nombre de conversations
        if len(reduced_conversations) > actual_target_conversations:
            reduced_conversations = random.sample(reduced_conversations, actual_target_conversations)
        elif len(reduced_conversations) < actual_target_conversations:
            print(f"Warning: Final count {len(reduced_conversations)} is less than target {actual_target_conversations}. This may happen if sources are exhausted.")

        random.shuffle(reduced_conversations)
    
    new_dataset_info = full_dataset.get("dataset_info", {}).copy()
    new_dataset_info["name"] = "ModelMatch verification dataset - reduced and no model change"
    original_version = new_dataset_info.get("version", "1.0")
    new_dataset_info["version"] = f"{original_version}-reduced-no-change-{len(reduced_conversations)}"
    new_dataset_info["description"] = (
        f"Reduced dataset ({len(reduced_conversations)} conversations) for testing ModelMatch, "
        "containing only conversations with no model change. "
        "Proportions from original_sources (among 'no change' conversations) are approximately preserved."
    )
    new_dataset_info["total_conversations"] = len(reduced_conversations)

    present_sources_in_reduced = sorted(list(set(
        conv.get("source_dataset", "unknown") for conv in reduced_conversations
    )))
    new_dataset_info["source_datasets"] = [s for s in present_sources_in_reduced if s != "unknown"]
    if "unknown" in present_sources_in_reduced:
        new_dataset_info["source_datasets"].append("unknown")
    
    reduced_dataset = {
        "dataset_info": new_dataset_info,
        "conversations": reduced_conversations
    }

    save_dataset(reduced_dataset, output_dataset_path)
    print_source_distribution(reduced_conversations, "Final reduced dataset")

if __name__ == "__main__":
    main()