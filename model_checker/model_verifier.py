import requests
import json

VLLM_API_URL = "http://localhost:8000/v1"
EXPECTED_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

EXPECTED_MODEL_PARAMS = {
    "id": EXPECTED_MODEL_ID,
    "object": "model",
    "owned-by": "meta-llama"
}

def verify_model_parameters():
    print(f"--- Début de la vérification du modèle sur {VLLM_API_URL} ---")

    try:
        response = requests.get(f"{VLLM_API_URL}/models")
        response.raise_for_status()

        models_data = response.json()

        target_model_info = None
        for model in models_data.get("data", []):
            if model.get("id") == EXPECTED_MODEL_ID:
                target_model_info = model
                break
        
        if not target_model_info:
            print(f"ERREUR : Le modèle attendu '{EXPECTED_MODEL_ID}' n'a pas été trouvé sur le serveur.")
            return False
        
        print(f"SUCCÈS : Le modèle '{EXPECTED_MODEL_ID}' est bien disponible.")
        print("\nInformations reçues de l'API :")
        print(json.dumps(target_model_info, indent=2))

        print("\n--- Vérification des paramètres ---")
        all_params_match = True
        for key, expected_value in EXPECTED_MODEL_PARAMS.items():
            actual_value = target_model_info.get(key)
            if actual_value == expected_value:
                print(f"  [OK] '{key}': correspond ('{actual_value}')")
            else:
                print(f"  [ÉCHEC] '{key}': attendu '{expected_value}', obtenu '{actual_value}'")
                all_params_match = False
        
        print("\n--- Résultat de la vérification ---")
        if all_params_match:
            print("Le modèle correspond parfaitement aux paramètres attendus.")
            return True
        else:
            print("Le modèle ne correspond pas à tous les paramètres attendus.")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"ERREUR de connexion à l'API : {e}")
        print("Veuillez vérifier que votre serveur vLLM est bien démarré et accessible à l'adresse configurée.")
        return False

if __name__ == "__main__":
    verify_model_parameters()