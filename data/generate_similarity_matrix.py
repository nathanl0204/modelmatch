import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

def load_dataset(file_path: str) -> list:
    """
    Charge les conversations depuis un fichier JSON.

    Args:
        file_path: Le chemin vers le fichier JSON.
    
    Returns:
        Une liste de conversations, ou une liste vide en cas d'erreur.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("conversations", data)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {file_path} n'a pas été trouvé.")
        return []
    except json.JSONDecodeError:
        print(f"Erreur: Impossible de décoder le JSON depuis {file_path}.")
        return []
    
def get_unique_models(conversations: list) -> list:
    """
    Extrait et retourne une liste triée des noms de modèles uniques
    qui ont des réponses dans le dataset.

    Args:
        conversations: La liste des conversations du dataset.
    
    Returns:
        Une liste triée de noms de modèles uniques.
    """
    models = set()
    for conv in conversations:
        if 'model_responses' in conv and conv['model_responses']:
            for model_name in conv['model_responses'].keys():
                models.add(model_name)
    return sorted(list(models))

def build_similarity_matrix(conversations: list, model_list: list, embedding_model) -> pd.DataFrame:
    """
    Construit une latrice de similarité cosinus entre les modèles en comparant
    leurs réponses pour chaque tour de chaque conversation.

    Args:
        conversations: La liste des conversations.
        model_list: La liste des modèles uniques à inclure dans la matrice.
        embedding_model: Le modèle SentenceTransformer pré-chargé pour générer les embeddings.
    
    Returns:
        Un DataFrame pandas représentant la matrice de similarité cumulative.
    """
    similarity_matrix = pd.DataFrame(0.0, index=model_list, columns=model_list)

    for conv in tqdm(conversations, desc="Processing conversations"):
        if 'model_responses' not in conv or not conv['model_responses']:
            continue
        
        model_pairs = combinations(conv['model_responses'].keys(), 2)

        for model1, model2 in model_pairs:
            if model1 not in model_list or model2 not in model_list:
                continue

            responses1 = conv['model_responses'][model1]
            responses2 = conv['model_responses'][model2]

            num_turns = min(len(responses1), len(responses2))
            for i in range(num_turns):
                text1 = responses1[i]
                text2 = responses2[i]

                if not text1 or not text2:
                    continue
                
                embeddings = embedding_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

                # Ajoute la similarité à la matrice (la matrice est symétrique)
                similarity_matrix.loc[model1, model2] += similarity
                similarity_matrix.loc[model2, model1] += similarity
        
    return similarity_matrix

if __name__ == "__main__":
    DATASET_PATH = 'modelmatch_dataset_reduced_no_change.json'

    print("1. Chargement du dataset...")
    conversations = load_dataset(DATASET_PATH)

    if not conversations:
        exit()
    
    print("2. Identification des modèles uniques...")
    unique_models = get_unique_models(conversations)
    print(f"   Trouvé {len(unique_models)} modèles uniques.")

    print("3. Initialisation du modèle d'embedding (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("4. Calcul de la matrice de similarité cumulative...")
    full_matrix = build_similarity_matrix(conversations, unique_models, embedding_model)

    print("4.5. Réordonnancement de la matrice par clustering hiérarchique...")
    dissimilarity_matrix = 1 - full_matrix.to_numpy()
    np.fill_diagonal(dissimilarity_matrix, 0)

    linked = linkage(dissimilarity_matrix, method='ward')

    # Obtient l'ordre des modèles à partir du dendrogramme pour une meilleure visualisation
    dendro = dendrogram(linked, no_plot=True)
    reordered_models = [unique_models[i] for i in dendro['leaves']]

    # Réorganise la matrice selon le nouvel ordre
    full_matrix = full_matrix.reindex(index=reordered_models, columns=reordered_models)

    print("5. Formatage de la matrice en triangulaire inférieure...")
    lower_triangular_matrix = pd.DataFrame(
        np.tril(full_matrix.values, k=-1),
        index=full_matrix.index,
        columns=full_matrix.columns
    )

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', len(unique_models))
    pd.set_option('display.float_format', '{:.2f}'.format)

    print("\n=== Matrice de similarité cposinus cumulative (triangulaire inférieure) ===")
    print(lower_triangular_matrix)

    print("6. Génération de la visualisation de la matrice...")
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(lower_triangular_matrix, dtype=bool))

    sns.heatmap(lower_triangular_matrix,
                annot=True,
                fmt=".2f",
                cmap='viridis',
                mask=mask,
                linewidths=.5)
    
    plt.title('Matrice de similarité cosinus entre les modèles', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = 'similarity_matrix2.png'
    plt.savefig(save_path)
    print(f"7. Matrice de similarité sauvegardée dans '{save_path}'")

    plt.show()