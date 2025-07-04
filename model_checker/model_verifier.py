import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from huggingface_hub import HfApi, model_info, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

class LocalModelVerifier:
    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi(token=hf_token)
        self.model_id = None
        self.local_path = None
        self.hf_info = None
        self.local_config = None
    
    def _get_huggingface_info(self, model_id: str) -> bool:
        print(f"\n[INFO] Récupération des informations pour '{model_id}' depuis le Hub Hugging Face...")
        try:
            self.hf_info = model_info(model_id, token=self.api.token)
            print("[SUCCESS] Informations récupérées avec succès.")
            return True
        except HfHubHTTPError as e:
            print(f"[ERROR] Impossible de trouver le modèle '{model_id}' sur le Hub. Erreur: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Une erreur inattendue est survenue lors de la récupération des informations : {e}")
            return False
    
    def _load_local_config(self) -> bool:
        config_path = self.local_path / "config.json"
        print(f"[INFO] Chargement de la configuration locale depuis '{config_path}'...")
        if not config_path.exists():
            print(f"[ERROR] Le fichier 'config.json' est introuvable dans '{self.local_path}'.")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.local_config = json.load(f)
        
        self.model_id = self.local_config.get("_name_or_path")
        if not self.model_id:
            print("[ERROR] Le champ '_name_or_path' est manquant dans 'config.json'. Impossible d'identifier le modèle.")
            return False
        
        print(f"[SUCCESS] Configuration locale chargée. Modèle identifié : '{self.model_id}'.")
        return True
    
    def _verify_config_parameters(self) -> Dict[str, Any]:
        print("\n--- Vérification des paramètres de configuration ---")
        report = {"status": "SUCCESS", "discrepancies": []}

        try:
            hf_config_path = hf_hub_download(repo_id=self.model_id, filename="config.json", token=self.api.token)
            with open(hf_config_path, 'r', encoding='utf-8') as f:
                hf_config = json.load(f)
        except Exception as e:
            return {"status": "ERROR", "message": f"Impossible de télécharger le config.json de référence: {e}"}
        
        params_to_check = [
            "architectures", "model_type", "num_hidden_layers",
            "hidden_size", "intermediate_size", "num_attention_heads",
            "vocab_size", "hidden_act"
        ]

        for param in params_to_check:
            local_val = self.local_config.get(param)
            hf_val = hf_config.get(param)
            if local_val != hf_val:
                discrepancy = {
                    "parameter": param,
                    "local_value": local_val,
                    "expected_value": hf_val
                }
                report["discrepancies"].append(discrepancy)
                report["status"] = "FAIL"

        if report["status"] == "SUCCESS":
            print("[SUCCESS] Les paramètres de configuration correspondent à la référence.")
        else:
            print(f"[FAIL] Des divergences ont été trouvées dans la configuration : {report['discrepancies']}")
        
        return report
    
    def _verify_file_integrity(self) -> Dict[str, Any]:
        print("\n--- Vérification de l'intégrité des fichiers (taille et hash) ---")
        report = {"status": "SUCCESS", "checked_files": []}

        model_files = [f for f in self.hf_info.siblings if f.rfilename.endswith(('.safetensors', '.bin'))]
        if not model_files:
            return {"status": "WARNING", "message": "Aucun fichier de poids (.safetensors, .bin) trouvé sur le Hub."}
        
        for hf_file in model_files:
            filename = hf_file.rfilename
            local_file_path = self.local_path / filename
            file_report = {"file": filename, "status": "SUCCESS", "checks": []}

            if not local_file_path.exists():
                file_report["status"] = "FAIL"
                file_report["checks"].append({"type": "existence", "status": "FAIL", "message": "Fichier manquant localement."})
            else:
                local_size = local_file_path.stat().st_size
                hf_size = hf_file.size
                size_check = {"type": "size", "local": local_size, "expected": hf_size}
                if local_size != hf_size:
                    file_report["status"] = "FAIL"
                    size_check["status"] = "FAIL"
                else:
                    size_check["status"] = "SUCCESS"
                file_report["checks"].append(size_check)

                print(f"[INFO] Calcul du hash SHA256 pour '{filename}' (peut prendre du temps)...")
                sha256 = hashlib.sha256()
                with open(local_file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)
                local_hash = sha256.hexdigest()

                hf_hash = hf_file.sha256
                hash_check = {"type": "sha256", "local": local_hash, "expected": hf_hash}
                if local_hash != hf_hash:
                    file_report["status"] = "FAIL"
                    hash_check["status"] = "FAIL"
                else:
                    hash_check["status"] = "SUCCESS"
                file_report["checks"].append(hash_check)
            
            report["checked_files"].append(file_report)
            if file_report["status"] != "SUCCESS":
                print("[SUCCESS] L'intégrité de tous les fichiers a été vérifiée avec succès.")
            else:
                print("[FAIL] Des problèmes d'intégrité ont été détectés.")
            
            return report
        
        def verify_model(self, local_path: str) -> Dict[str, Any]:
            self.local_path = Path(local_path)
            if not self.local_path.is_dir():
                return {"status": "ERROR", "message": f"Le chemin spécifié n'est pas un répertoire valide : {local_path}"}
            
            if not self._load_local_config():
                return {"status": "ERROR", "message": "Échec de la lecture de la configuration locale."}
            
            if not self._get_huggingface_info(self.model_id):
                return {"status": "ERROR", "message": f"Impossible de récupérer les informations pour {self.model_id}."}
            
            config_report = self._verify_config_parameters()
            integrity_report = self._verify_file_integrity()

            final_status = "SUCCESS"
            if config_report["status"] != "SUCCESS" or integrity_report["status"] != "SUCCESS":
                final_status = "FAIL"
            
            return {
                "overall_status": final_status,
                "model_id": self.model_id,
                "local_path": str(self.local_path),
                "verification_reports": {
                    "config_parameters": config_report,
                    "file_integrity": integrity_report
                }
            }

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model.save_pretrained("./Mistral-7B-Instruct-v0.1")

    verifier = LocalModelVerifier()

    # Cas 1: Un modèle valide et non modifié
    valid_model_path = "./Mistral-7B-Instruct-v0.1"
    if os.path.exists(valid_model_path):
        print("="*50)
        print(f"VÉRIFICATION D'UN MODÈLE VALIDE : {valid_model_path}")
        print("="*50)
        report = verifier.verify_model(valid_model_path)
        print("\n--- RAPPORT FINAL ---")
        print(json.dumps(report, indent=2))
    else:
        print(f"\n[AVERTISSEMENT] Le répertoire de test '{valid_model_path}' n'existe pas. L'exemple de cas valide est ignoré.")
    
    # Cas 2: Un modèle quantifié (qui devrait échouer aux vérifications)
    quantized_model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

    # Création d'un faux modèle en modifiant un fichier
    invalid_model_path = "./Invalid-Mistral-7B-Instruct-v0.1"
    if os.path.exists(valid_model_path) and not os.path.exists(invalid_model_path):
        import shutil
        print(f"\n[SETUP] Création d'un modèle invalide de test à '{invalid_model_path}'...")
        shutil.copytree(valid_model_path, invalid_model_path)
        file_to_alter = os.path.join(invalid_model_path, "model-00001-of-00002.safetensors")
        if os.path.exists(file_to_alter):
            with open(file_to_alter, "ab") as f:
                f.write(b"modification")
            print("[SETUP] Fichier altéré avec succès.")
        else:
            print("[SETUP] Fichier de modèle à altérer non trouvé, la simulation pourrait ne pas être parfaite.")
    
    if os.path.exists(invalid_model_path):
        print("\n" + "="*50)
        print(f"VÉRIFICATION D'UN MODÈLE ALTÉRÉ : {invalid_model_path}")
        print("="*50)
        report_invalid = verifier.verify_model(invalid_model_path)
        print("\n--- RAPPORT FINAL (MODÈLE ALTÉRÉ) ---")
        print(json.dumps(report_invalid, indent=2))
    else:
        print(f"\n[AVERTISSEMENT] Le répertoire de test invalide '{invalid_model_path}' n'a pas pu être créé. L'exemple de cas invalide est ignoré.")