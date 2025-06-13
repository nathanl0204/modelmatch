import os
import re
import nltk
import json
import random
import numpy as np
from together import Together
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class ModelMatchPlugin:
    def __init__(self, together_api_key: str, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        if not together_api_key:
            raise ValueError("Together API key is required.")
        self.client = Together(api_key=together_api_key)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.politeness_words = ["please", "thank", "sorry", "excuse", "pardon", "appreciate", "grateful"]
        self.adverb_tags = ['RB', 'RBR', 'RBS']
        self.common_emojis_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        self.markdown_patterns = [
            re.compile(r"\*\*.*?\*\*"),  # Gras
            re.compile(r"\*.*?\*"),    # Italique/Gras
            re.compile(r"__.*?__"),  # Sous-ligné/Gras
            re.compile(r"_.*?_"),    # Italique/Sous-ligné
            re.compile(r"`.*?`"),    # Ligne de code
            re.compile(r"```.*?```", re.DOTALL),  # Bloc de code
            re.compile(r"^\s*[-*+]\s+"), # Liste d'items
            re.compile(r"^\s*\d+\.\s+")   # Liste d'items numérotée
        ]

    def _get_model_response(self, model_name: str, messages: list) -> str | None:
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=350,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Together API for {model_name}: {e}")
            return None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode([text])[0]
    
    def _analyze_response_features(self, response_text: str) -> dict:
        if not response_text:
            return {
                "text": "",
                "length": 0,
                "tokens": [],
                "politeness_score": 0,
                "adverb_score": 0,
                "punctuation_counts": Counter(),
                "markdown_score": 0,
                "emoji_count": 0,
                "embedding": self._get_embedding(""),
            }
        
        tokens = nltk.word_tokenize(response_text.lower())
        tagged_tokens = nltk.pos_tag(nltk.word_tokenize(response_text)) # Le POS tagger est sensible à la casse

        politeness_score = sum(1 for token in tokens if any(polite_word in token for polite_word in self.politeness_words))
        adverb_score = sum(1 for _, tag in tagged_tokens if tag in self.adverb_tags)

        punctuation_counts = Counter(char for char in response_text if char in "!?;:.")

        markdown_score = 0
        for pattern in self.markdown_patterns:
            if pattern.search(response_text):
                markdown_score += 1
        
        emoji_count = len(self.common_emojis_pattern.findall(response_text))

        embedding = self._get_embedding(response_text)

        return {
            "text": response_text,
            "length": len(response_text),
            "tokens": tokens,
            "politeness_score": politeness_score,
            "adverb_score": adverb_score,
            "punctuation_counts": punctuation_counts,
            "markdown_score": markdown_score,
            "emoji_count": emoji_count,
            "embedding": embedding,
        }
    
    def _compare_features(self, prev_feats: dict, curr_feats: dict, thresholds: dict) -> list[str]:
        deviations = []

        similarity = cosine_similarity(prev_feats["embedding"].reshape(1, -1), curr_feats["embedding"].reshape(1, -1))[0][0]
        if similarity < thresholds.get("cosine_similarity_drop", 0.85): # 0.85 est le seuil min de similarité
            deviations.append(f"cosine_similarity (val:{similarity:.2f} < thr:{thresholds.get('cosine_similarity_drop', 0.85)})")
        
        if prev_feats["length"] > 0:
            len_ratio = curr_feats["length"] / prev_feats["length"]
            if not (thresholds.get("length_ratio_min", 0.5) < len_ratio < thresholds.get("length_ratio_max", 2.0)):
                deviations.append(f"length_ratio (val:{len_ratio:.2f} not in ({thresholds.get('length_ratio_min', 0.5)}-{thresholds.get('length_ratio_max', 2.0)}))")
        elif curr_feats["length"] > thresholds.get("length_min_if_prev_empty", 20):
            deviations.append(f"length_appeared (val:{curr_feats['length']} > thr:{thresholds.get('length_min_if_prev_empty', 20)})")
        

        if abs(curr_feats["politeness_score"] - prev_feats["politeness_score"]) > thresholds.get("politeness_diff", 2):
            deviations.append(f"politeness_score (abs_diff:{abs(curr_feats['politeness_score'] - prev_feats['politeness_score'])} > thr:{thresholds.get('politeness_diff', 2)})")

        if abs(curr_feats["adverb_score"] - prev_feats["adverb_score"]) > thresholds.get("adverb_diff", 3):
            deviations.append(f"adverb_score (abs_diff:{abs(curr_feats['adverb_score'] - prev_feats['adverb_score'])} > thr:{thresholds.get('adverb_diff', 3)})")

        if abs(curr_feats["markdown_score"] - prev_feats["markdown_score"]) > thresholds.get("markdown_diff", 1):
            deviations.append(f"markdown_score (abs_diff:{abs(curr_feats['markdown_score'] - prev_feats['markdown_score'])} > thr:{thresholds.get('markdown_diff', 1)})")

        if (prev_feats["emoji_count"] == 0 and curr_feats["emoji_count"] > 0) or \
           (prev_feats["emoji_count"] > 0 and curr_feats["emoji_count"] == 0) or \
           (prev_feats["emoji_count"] > 0 and abs(curr_feats["emoji_count"] - prev_feats["emoji_count"]) > thresholds.get("emoji_diff", 1)):
            deviations.append(f"emoji_count (prev:{prev_feats['emoji_count']}, curr:{curr_feats['emoji_count']})")
        
        for punc in ['!', '?']:
            prev_punc_count = prev_feats["punctuation_counts"].get(punc, 0)
            curr_punc_count = curr_feats["punctuation_counts"].get(punc, 0)
            if abs(curr_punc_count - prev_punc_count) > thresholds.get(f"punc_{punc}_diff", 2):
                deviations.append(f"punc_{punc}_count (abs_diff:{abs(curr_punc_count - prev_punc_count)} > thr:{thresholds.get(f'punc_{punc}_diff', 2)})")

        return deviations
    
    def verify_conversation_for_change(
            self,
            user_prompts: list[str],
            initial_model_name: str,
            # Seuils pour la détection de changement. Ceux-ci devront être ajustés
            thresholds: dict | None = None,
            actual_model_change_index: int | None = None,
            actual_changed_model_name: str | None = None
    ) -> tuple[bool, int, list[str]]:
        if thresholds is None:
            thresholds = {
                "cosine_similarity_drop": 0.80, # La similarité doit être supérieure à 0.85
                "length_ratio_min": 0.4,
                "length_ratio_max": 2.5,
                "length_min_if_prev_empty": 10,
                "politeness_diff": 3,
                "adverb_diff": 4,
                "markdown_diff": 2,
                "emoji_diff": 1,
                "punc_!_diff": 2,
                "punc_?_diff": 2,
                "min_deviations_for_change": 3 # Combien de métriques doivent dévier pour considérer qu'il y a un changement
            }
        
        history = []
        previous_response_features = None

        print(f"Starting verification. Expecting model: {initial_model_name}")
        if actual_model_change_index is not None and actual_changed_model_name is not None:
            print(f"Simulating a switch to '{actual_changed_model_name}' after prompt index {actual_model_change_index - 1}.")

        current_model_for_api_call = initial_model_name

        for i, prompt_text in enumerate(user_prompts):
            print(f"\n--- Turn {i+1}/{len(user_prompts)} ---")
            print(f"User: {prompt_text[:100]}...")

            if actual_model_change_index is not None and \
               actual_changed_model_name is not None and \
               i >= actual_model_change_index:
                if current_model_for_api_call != actual_changed_model_name:
                    print(f"--- SIMULATING SWITCH: Now using {actual_changed_model_name} for API call (prompt index {i}) ---")
                current_model_for_api_call = actual_changed_model_name

            current_turn_messages = history + [{"role": "user", "content": prompt_text}]
            response_text = self._get_model_response(current_model_for_api_call, current_turn_messages)

            if response_text is None:
                print(f"Model {current_model_for_api_call}: No response or error for prompt {i+1}. Assuming critical failure/change.")
                return True, i - 1, ["api_error_or_no_response"]
            
            print(f"Model ({current_model_for_api_call}): {response_text[:100]}...")
            current_response_features = self._analyze_response_features(response_text)

            history.append({"role": "user", "content": prompt_text})
            history.append({"role": "assistant", "content": response_text})

            if i == 0: # On établit la baseline pour la première réponse
                previous_response_features = current_response_features
                print("Established baseline features from first response.")
                continue

            deviations = self._compare_features(previous_response_features, current_response_features, thresholds)

            if deviations:
                print(f"Potential change detected after prompt {i} (turn {i+1}) due to: {deviations}")
                # On vérifie si le nombre de déviations est suffisant
                if len(deviations) >= thresholds.get("min_deviations_for_change", 2):
                    return True, i - 1, deviations
                
            previous_response_features = current_response_features

        print("\nNo significant model change detected based on the analyzed metrics.")
        return False, -1, []

if __name__ == "__main__":
    API_KEY = os.environ.get("TOGETHER_API_KEY")
    if not API_KEY:
        print("Please set the TOGETHER_API_KEY environment variable to run this example.")
        exit()
    
    plugin = ModelMatchPlugin(together_api_key=API_KEY)

    dataset_file_path = "../data/modelmatch_dataset.json"

    try:
        with open(dataset_file_path, 'r', encoding='utf-8') as f:
            dataset_content = json.load(f)
        all_conversations = dataset_content.get("conversations")
        if not all_conversations:
            print(f"No conversations found in {dataset_file_path} under the 'conversations' key, or the list is empty.")
            exit()
    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_file_path}")
        exit()
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {dataset_file_path}")
        exit()
    
    conversations_with_expected_change = [
        conv for conv in all_conversations if conv.get("has_model_change") and conv.get("model_change_index") is not None
    ]

    if not conversations_with_expected_change:
        print("Warning: No conversations found with 'has_model_change': true. Cannot test model change detection.")
        if not all_conversations:
            print("No conversations at all in the dataset.")
        exit()
    else:
        conversation_to_test = random.choice(conversations_with_expected_change)
    
    test_prompts = conversation_to_test.get("user_prompts", [])
    initial_model_for_plugin = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_to_switch_to = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

    if initial_model_for_plugin == model_to_switch_to:
        print(f"Warning: Initial model and switch model are the same ({initial_model_for_plugin}). Pick a different 'model_to_switch_to'.")
        if initial_model_for_plugin == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            model_to_switch_to = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
        else:
            model_to_switch_to = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        print(f"Adjusted 'model_to_switch_to' to: {model_to_switch_to}")
    

    simulated_change_prompt_index = conversation_to_test.get("model_change_index")

    if not test_prompts:
        print(f"Selected conversation (ID: {conversation_to_test.get('id', 'N/A')}) has no user prompts.")
        exit()
    
    if simulated_change_prompt_index is None or not (0 <= simulated_change_prompt_index < len(test_prompts)):
        print(f"Invalid 'model_change_index' ({simulated_change_prompt_index}) for conversation ID {conversation_to_test.get('id', 'N/A')}. Cannot simulate change.")
        exit()

    print(f"Testing with initial model (plugin expectation): {initial_model_for_plugin}")
    print(f"Conversation ID: {conversation_to_test.get('id', 'N/A')}")
    print(f"Dataset 'target_model_name': {conversation_to_test.get('target_model_name', 'N/A')}")
    print(f"Dataset 'has_model_change': {conversation_to_test.get('has_model_change')}")
    print(f"Dataset 'model_change_index': {simulated_change_prompt_index} (0-indexed)")
    print(f"Number of prompts: {len(test_prompts)}")
    print(f"Simulating switch to model: {model_to_switch_to} at prompt index {simulated_change_prompt_index}")


    change_detected, change_idx, reasons = plugin.verify_conversation_for_change(
        user_prompts=test_prompts,
        initial_model_name=initial_model_for_plugin,
        thresholds=None,
        actual_model_change_index=simulated_change_prompt_index,
        actual_changed_model_name=model_to_switch_to
    )
    if change_detected:
        print(f"\nResult: Model change DETECTED after prompt index {change_idx} (0-indexed). Reasons: {reasons}")
        if change_idx == simulated_change_prompt_index - 1:
            print(f"SUCCESS: Detected change at the correct point (after prompt {simulated_change_prompt_index - 1}, before prompt {simulated_change_prompt_index}).")
        else:
            print(f"INFO: Change detected at index {change_idx}, but simulated change was at index {simulated_change_prompt_index - 1}.")
    else:
        print("\nResult: No model change detected by the plugin.")
        print(f"FAILURE: A model change was simulated at prompt index {simulated_change_prompt_index} but not detected.")