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
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict

# Télécharge les ressources NLTK nécessaires si elles ne sont pas déjà présentes
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

class BaseMetric(ABC):
    """
    Classe de base abstraite pour toutes les métriques de détection de changement.
    Définit l'interface que chaque métrique doit implémenter.
    """
    def __init__(self, name: str, default_thresholds: Optional[Dict[str, Any]] = None):
        self.name = name
        self.default_thresholds = default_thresholds if default_thresholds is not None else {}
    
    @abstractmethod
    def analyze(self, reponse_text: str, previous_response_metric_feature: Optional[Any] = None) -> Any:
        """Analyse une réponse textuelle pour extraire la valeur de la métrique."""
        pass

    @abstractmethod
    def compare(self, prev_feature_value: Any, curr_feature_value: Any, thresholds: Dict[str,Any]) -> List[str]:
        """Compare les valeurs de la métrique entre deux réponses et retourne les déviations."""
        pass

    def get_effective_thresholds(self, global_metric_specific_thresholds: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine les seuils par défaut avec des seuils globaux surchargés."""
        eff_thresholds = self.default_thresholds.copy()
        if global_metric_specific_thresholds:
            eff_thresholds.update(global_metric_specific_thresholds)
        return eff_thresholds

class EmbeddingMetric(BaseMetric):
    """Métrique basée sur la similarité cosinus des embeddings de phrases."""
    def __init__(self, embedding_model: SentenceTransformer, default_thresholds: Optional[Dict[str, Any]] = None):
        super().__init__("embedding", default_thresholds or {"cosine_similarity_drop": 0.70})
        self.embedding_model = embedding_model
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Génère un embedding pour un texte donné."""
        return self.embedding_model.encode([text])[0]
    
    def analyze(self, response_text: str, previous_resppnse_metric_feature: Optional[Any] = None) -> np.ndarray:
        """Analyse le texte pour retourner son embedding."""
        return self._get_embedding(response_text if response_text else "")
    
    def compare(self, prev_embedding: np.ndarray, curr_embedding: np.ndarray, thresholds: Dict[str, Any]) -> List[str]:
        """Compare deux embeddings et détecte une chute de similarité."""
        deviations = []
        if np.any(prev_embedding) and np.any(curr_embedding):
            similarity = cosine_similarity(prev_embedding.reshape(1, -1), curr_embedding.reshape(1, -1))[0][0]
            threshold_val = thresholds.get("cosine_similarity_drop", self.default_thresholds["cosine_similarity_drop"])
            if similarity < threshold_val:
                deviations.append(f"cosine_similarity (val:{similarity:.2f} < thr:{threshold_val})")
        elif np.any(prev_embedding) != np.any(curr_embedding):
            deviations.append(f"cosine_similarity (one embedding is zero, other is not)")
        return deviations

class LengthMetric(BaseMetric):
    """Métrique basée sur la longueur du texte de la réponse."""
    def __init__(self, default_thresholds: Optional[Dict[str, Any]] = None):
        super().__init__("length", default_thresholds or {
            "length_ratio_min": 0.4,
            "length_ratio_max": 2.5,
            "length_min_if_prev_empty": 10
        })
    
    def analyze(self, response_text: str, previous_reponse_metric_feature: Optional[Any] = None) -> int:
        """Retiurne la longueur du texte."""
        return len(response_text)
    
    def compare(self, prev_length: int, curr_length: int, thresholds: Dict[str, Any]) -> List[str]:
        """Compare les longueurs et détecte des changements de ratio importants."""
        deviations = []
        if prev_length > 0:
            len_ratio = curr_length / prev_length if prev_length != 0 else float('inf')
            min_ratio = thresholds.get("length_ratio_min", self.default_thresholds["length_ratio_min"])
            max_ratio = thresholds.get("length_ratio_max", self.default_thresholds["length_ratio_max"])
            if not(min_ratio <= len_ratio <= max_ratio):
                deviations.append(f"length_ratio (val:{len_ratio:.2f} not in [{min_ratio}-{max_ratio}])")
        elif curr_length > thresholds.get("length_min_if_prev_empty", self.default_thresholds["length_min_if_prev_empty"]):
            deviations.append(f"length_appeared (val:{curr_length} > thr:{thresholds.get('length_min_if_prev_empty', self.default_thresholds['length_min_if_prev_empty'])})")
        return deviations

class PolitenessMetric(BaseMetric):
    """Métrique basée sur la présence de mots de politesse."""
    def __init__(self, default_thresholds: Optional[Dict[str, Any]] = None):
        super().__init__("politeness_score", default_thresholds or {"politeness_diff": 3})
        self.politeness_words = ["please", "thank", "sorry", "excuse", "pardon", "appreciate", "grateful"]
    
    def analyze(self, response_text: str, previous_response_metric_feature: Optional[Any] = None) -> int:
        """Compte le nombre de mots de politesse dans le texte."""
        if not response_text: return 0
        tokens = nltk.word_tokenize(response_text.lower())
        return sum(1 for token in tokens if any(polite_word in token for polite_word in self.politeness_words))
    
    def compare(self, prev_score: int, curr_score: int, thresholds: Dict[str, Any]) -> List[str]:
        """Détecte une différence significative dans le score de politesse."""
        diff_threshold = thresholds.get("politeness_diff", self.default_thresholds["politeness_diff"])
        if abs(curr_score - prev_score) > diff_threshold:
            return [f"politeness_score (abs_diff:{abs(curr_score - prev_score)} > thr:{diff_threshold})"]
        return []

class AdverbMetric(BaseMetric):
    """Métrique basée sur le nombre d'adverbes utilisés."""
    def __init__(self, default_thresholds: Optional[Dict[str, Any]]  = None):
        super().__init__("adverb_score", default_thresholds or {"adverb_diff": 4})
        self.adverb_tags = ['RB', 'RBR', 'RBS']
    
    def analyze(self, response_text: str, previous_response_metric_feature: Optional[Any] = None) -> int:
        """Compte le nombre d'adverbes en utilisant le POS tagging."""
        if not response_text: return 0
        tagged_tokens = nltk.pos_tag(nltk.word_tokenize(response_text))
        return sum(1 for _, tag in tagged_tokens if tag in self.adverb_tags)
    
    def compare(self, prev_score: int, curr_score: int, thresholds: Dict[str, Any]) -> List[str]:
        """Détecte une différence significative dans le nombre d'adverbes."""
        diff_threshold = thresholds.get("adverb_diff", self.default_thresholds["adverb_diff"])
        if abs(curr_score - prev_score) > diff_threshold:
            return [f"adverb_score (abs_diff:{abs(curr_score - prev_score)} > thr:{diff_threshold})"]
        return []

class MarkdownMetric(BaseMetric):
    """Métrique basée sur l'utilisation de la syntaxe Markdown."""
    def __init__(self, default_thresholds: Optional[dict[str, Any]] = None):
        super().__init__("markdown_score", default_thresholds or {"markdown_diff": 2})
        self.markdown_patterns = [
            re.compile(r"\*\*.*?\*\*"), re.compile(r"\*.*?\*"),
            re.compile(r"__.*?__"), re.compile(r"_.*?_"),
            re.compile(r"`.*?`"), re.compile(r"```.*?```", re.DOTALL),
            re.compile(r"^\s*[-*+]\s+", re.MULTILINE),
            re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
        ]
    
    def analyze(self, response_text: str, previous_response_metric_feature: Optional[Any] = None) -> int:
        """Calcule un score basé sur la présence de différents types de Markdown."""
        if not response_text: return 0
        score = 0
        for pattern in self.markdown_patterns:
            if pattern.search(response_text):
                score += 1
        return score
    
    def compare(self, prev_score: int, curr_score: int, thresholds: Dict[str, Any]) -> List[str]:
        """Détecte une différence significative dans l'utilisation de Markdown."""
        diff_threshold = thresholds.get("markdown_diff", self.default_thresholds["markdown_diff"])
        if abs(curr_score - prev_score) > diff_threshold:
            return [f"markdown_score (abs_diff:{abs(curr_score - prev_score)} > thr:{diff_threshold})"]
        return []

class EmojiMetric(BaseMetric):
    """Métrique basée sur l'utilisation d'émojis."""
    def __init__(self, default_thresholds: Optional[Dict[str, Any]] = None):
        super().__init__("emoji_count", default_thresholds or {"emoji_diff": 1})
        self.common_emojis_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F" # Émoticônes
            "\U0001F300-\U0001F5FF" # Symboles et pictogrammes
            "\U0001F680-\U0001F6FF" # Symboles de transport et cartes
            "\U0001F1E0-\U0001F1FF" # Drapeaux (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
    
    def analyze(self, response_text: str, previous_response_metric_feature: Optional[Any] = None) -> int:
        """Compte le nombre d'émojis dans le texte."""
        if not response_text: return 0
        return len(self.common_emojis_pattern.findall(response_text))
    
    def compare(self, prev_count: int, curr_count: int, thresholds: Dict[str, Any]) -> List[str]:
        """Détecte l'apparition, la disparition ou un changement significatif du nombre d'émojis."""
        diff_threshold = thresholds.get("emoji_diff", self.default_thresholds["emoji_diff"])
        if (prev_count == 0 and curr_count > 0) or \
           (prev_count > 0 and curr_count == 0) or \
           (prev_count > 0 and abs(curr_count - prev_count) > diff_threshold):
            return [f"emoji_count (prev:{prev_count}, curr:{curr_count}, thr_for_diff_if_prev_gt_0:{diff_threshold})"]
        return []

class PunctuationMetric(BaseMetric):
    """Métrique basée sur le comptage de caractères de ponctuation spécifiques."""
    def __init__(self, punctuations_to_track: List[str], default_thresholds: Optional[Dict[str, Any]] = None):
        base_thresholds = {f"punc_{p}_diff": 2 for p in punctuations_to_track}
        if default_thresholds:
            base_thresholds.update(default_thresholds)
        super().__init__("punctuation_counts", base_thresholds)
        self.punctuations_to_track = punctuations_to_track
    
    def analyze(self, response_text: str, previous_response_metric_feature: Optional[Any] = None) -> Counter:
        """Compte les occurences de chaque caractère de ponctuation suivi."""
        if not response_text: return Counter()
        return Counter(char for char in response_text if char in self.punctuations_to_track)
    
    def compare(self, prev_counts: Counter, curr_counts: Counter, thresholds: Dict[str, Any]) -> List[str]:
        """Détecte une différence significative dans le comptage de chaque ponctuation."""
        deviations = []
        for punc_char in self.punctuations_to_track:
            prev_punc_count = prev_counts.get(punc_char, 0)
            curr_punc_count = curr_counts.get(punc_char, 0)

            threshold_key = f"punc_{punc_char}_diff"
            diff_threshold = thresholds.get(threshold_key, self.default_thresholds.get(threshold_key, 2))

            if abs(curr_punc_count - prev_punc_count) > diff_threshold:
                deviations.append(f"punc_{punc_char}_count (abs_diff:{abs(curr_punc_count - prev_punc_count)} > thr:{diff_threshold})")
        return deviations

class VocabularyRichnessMetric(BaseMetric):
    """Métrique basée sur la richesse du vocabulaire (Type-Token Ratio)."""
    def __init__(slef, default_thresholds: Optional[dict[str, Any]] = None):
        super().__init__("vocabulary_richness_ttr", default_thresholds or {"ttr_diff": 0.1, "min_tokens_for_ttr": 10})
    
    def analyze(self, response_text: str, previous_response_metric_feature: Optional[Any] = None) -> float:
        """Calcule le TTR (nombre de mots uniques / nombre total de mots)."""
        if not response_text: return 0.0
        tokens = nltk.word_tokenize(response_text.lower())
        if len(tokens) < self.default_thresholds.get("min_tokens_for_ttr", 10):
            return 0.0
        if not tokens: return 0.0
        return len(set(tokens)) / len(tokens)
    
    def compare(self, prev_ttr: float, curr_ttr: float, thresholds: Dict[str, Any]) -> List[str]:
        """Détecte une différence significative dans le TTR."""
        diff_threshold = thresholds.get("ttr_diff", self.default_thresholds["ttr_diff"])
        if prev_ttr > 0.0 and curr_ttr > 0.0 and abs(curr_ttr - prev_ttr) > diff_threshold:
            return [f"vocabulary_richness_ttr (abs_diff:{abs(curr_ttr - prev_ttr):.2f} > thr:{diff_threshold})"]
        elif (prev_ttr == 0.0 and curr_ttr > 0.1) or (curr_ttr == 0.0 and prev_ttr > 0.1):
            return [f"vocabulary_richness_ttr (change from/to negligible: prev={prev_ttr:.2f}, curr={curr_ttr:.2f})"]
        return []

class SentenceComplexityMetric(BaseMetric):
    """Métrique basée sur la complexité des phrases (longueur moyenne des phrases)."""
    def __init__(self, default_thresholds: Optional[Dict[str, Any]] = None):
        super().__init__("sentence_complexity_avg_len", default_thresholds or {
            "avg_len_abs_diff": 5.0,
            "avg_men_ratio_diff": 0.3,
            "min_sentences_for_metric": 2
        })
    
    def analyze(self, response_text: str, previous_response_metric_feature: Optional[Any] = None) -> float:
        """Calcule la longueur moyenne des phrases en tokens."""
        if not response_text: return 0.0
        sentences = nltk.sent_tokenize(response_text)
        if len(sentences) < self.default_thresholds.get("min_sentences_for_metric", 2):
            tokens = nltk.word_tokenize(response_text)
            return len(tokens) if tokens else 0.0
        
        total_tokens = 0
        num_sentences = 0
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            if tokens:
                total_tokens += len(tokens)
                num_sentences += 1
        
        if num_sentences == 0: return 0.0
        return total_tokens / num_sentences
    
    def compare(self, prev_avg_len: float, curr_avg_len: float, thresholds: Dict[str, Any]) -> List[str]:
        """Détecte une différence absolue ou relative significative dans la longueur moyenne des phrases."""
        deviations = []
        abs_diff_threshold = thresholds.get("avg_len_abs_diff", self.default_thresholds["avg_len_abs_diff"])
        ratio_diff_threshold = thresholds.get("avg_len_ratio_diff", self.default_thresholds["avg_len_ratio_diff"])

        if prev_avg_len > 0 and curr_avg_len > 0:
            if abs(curr_avg_len - prev_avg_len) > abs_diff_threshold:
                deviations.append(f"sentence_complexity_avg_len (abs_diff:{abs(curr_avg_len - prev_avg_len):.2f} > thr:{abs_diff_threshold})")
            
            if prev_avg_len > 1.0:
                ratio_diff = abs(curr_avg_len - prev_avg_len) / prev_avg_len
                if ratio_diff > ratio_diff_threshold:
                    deviations.append(f"sentence_complexity_avg_len (ratio_diff:{ratio_diff:.2f} > thr:{ratio_diff_threshold})")
        elif (prev_avg_len == 0.0 and curr_avg_len > abs_diff_threshold) or \
             (curr_avg_len == 0.0 and prev_avg_len > abs_diff_threshold):
            deviations.append(f"sentence_complexity_avg_len (change from/to zero: prev={prev_avg_len:.2f}, curr={curr_avg_len:.2f})")
        
        return deviations

class ModelMatchPluginRefactored:
    """
    Classe de test principale qui orchestre la détection de changement de modèle en utilisant un ensemble de métriques.
    """
    def __init__(self, together_api_key: str, metrics: List[BaseMetric]):
        if not together_api_key:
            raise ValueError("Together API key is required.")
        self.client = Together(api_key=together_api_key)
        self.metrics = metrics
    
    def _get_model_response(self, model_name: str, messages: list) -> str | None:
        """Appelle l'API Together pour obtenir une réponse du modèle."""
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=350,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Together API for {model_name}: {type(e).__name__} - {e}")
    
    def _analyze_all_features(self, response_text: str, previous_all_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Exécute l'analyse de toutes les métriques configurées sur une réponse."""
        current_features = {"text": response_text}
        for metric in self.metrics:
            prev_metric_feature = None
            if previous_all_features and metric.name in previous_all_features:
                prev_metric_feature = previous_all_features[metric.name]
            current_features[metric.name] = metric.analyze(response_text, prev_metric_feature)
        return current_features
    
    def _compare_all_features(self, prev_all_features: Dict[str, Any], curr_all_features: Dict[str, Any], global_thresholds_config: Optional[Dict[str, Any]]) -> List[str]:
        """Compare les caractéristiques de deux réponses en utilisant toutes les métriques."""
        all_deviations = []
        for metric in self.metrics:
            prev_feature_value = prev_all_features.get(metric.name)
            curr_feature_value = curr_all_features.get(metric.name)

            if prev_feature_value is not None and curr_feature_value is not None:
                metric_specific_thresholds_from_global = None
                if global_thresholds_config and metric.name in global_thresholds_config:
                    metric_specific_thresholds_from_global = global_thresholds_config[metric.name]

                effective_metric_thresholds = metric.get_effective_thresholds(metric_specific_thresholds_from_global)

                deviations = metric.compare(prev_feature_value, curr_feature_value, effective_metric_thresholds)
                all_deviations.extend(deviations)
        return all_deviations

    def verify_conversation_for_change(
            self,
            user_prompts: List[str],
            initial_model_name: str,
            thresholds_config: Optional[Dict[str, Any]] = None,
            actual_model_change_index: Optional[int] = None,
            actual_changed_model_name: Optional[str] = None
    ) -> tuple[bool, int, List[str]]:
        """
        Simule une conversation et vérifie si un changement de modèle se produit.
        """
        default_min_deviations = 4
        if thresholds_config and "global_min_deviations_for_change" in thresholds_config:
            min_deviations_for_change = thresholds_config["global_min_deviations_for_change"]
        else:
            min_deviations_for_change = default_min_deviations
        
        history = []
        previous_response_all_features = None

        print(f"Starting verification. Expecting model: {initial_model_name}")
        if actual_model_change_index is not None and actual_changed_model_name is not None:
            print(f"Simulating a switch to '{actual_changed_model_name}' after prompt index {actual_model_change_index - 1}.")
        
        current_model_for_api_call = initial_model_name

        for i, prompt_text in enumerate(user_prompts):
            print(f"\n--- Turn {i+1}/{len(user_prompts)} ---")
            print(f"User: {prompt_text[:100].replace(os.linesep, ' ')}...")

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
                detected_change_prompt_idx = i - 1 if i > 0 else 0
                return True, detected_change_prompt_idx, ["api_error_or_no_response"]
            
            print(f"Model ({current_model_for_api_call}): {response_text[:100].replace(os.linesep, ' ')}...")

            current_response_all_features = self._analyze_all_features(response_text, previous_response_all_features)

            history.append({"role": "user", "content": prompt_text})
            history.append({"role": "assistant", "content": response_text})

            if i == 0:
                previous_response_all_features = current_response_all_features
                print("Established baseline features from first response.")
                continue

            if previous_response_all_features is None:
                print("Critical error: previous_response_all_features is None after the first turn. This should not happen.")
                previous_response_all_features = current_response_all_features
                continue
            
            deviations = self._compare_all_features(previous_response_all_features, current_response_all_features, thresholds_config)

            if deviations:
                print(f"Potential change detected after prompt {i} (turn {i+1}) due to: {deviations}")
                if len(deviations) >= min_deviations_for_change:
                    return True, i - 1, deviations
            
            previous_response_all_features = current_response_all_features

        print("\nNo significant model change detected based on the analyzed metrics.")
        return False, -1, []

if __name__ == "__main__":
    API_KEY = os.environ.get("TOGETHER_API_KEY")
    if not API_KEY:
        print("Please set the TOGETHER_API_KEY environment variable to run this example.")
        exit()

    embedding_model_instance = SentenceTransformer('all-MiniLM-L6-v2')

    metrics_to_use = [
        EmbeddingMetric(embedding_model=embedding_model_instance,
                        default_thresholds={"cosine_similarity_drop": 0.70}),
        LengthMetric(default_thresholds={"length_ratio_min": 0.4,
                                         "length_ratio_max": 2.5,
                                         "length_min_if_prev_empty": 10}),
        PolitenessMetric(default_thresholds={"politeness_diff": 3}),
        AdverbMetric(default_thresholds={"adverb_diff": 4}),
        MarkdownMetric(default_thresholds={"markdown_diff": 2}),
        EmojiMetric(default_thresholds={"emoji_diff": 1}),
        PunctuationMetric(punctuations_to_track=['!', '?'],
                          default_thresholds={"punc_!_diff": 2, "punc_?_diff": 2}),
        VocabularyRichnessMetric(default_thresholds={"ttr_diff": 0.08, "min_tokens_for_ttr": 15}),
        SentenceComplexityMetric(default_thresholds={"avg_len_abs_diff": 4.0, "avg_len_ratio_diff": 0.25, "min_sentences_for_metric": 2})
    ]

    plugin = ModelMatchPluginRefactored(
        together_api_key=API_KEY,
        metrics=metrics_to_use
    )

    available_chat_models = []
    try:
        print("Fetching available models from Together API...")
        models_list_response = plugin.client.models.list()
        available_chat_models = [
            model.id for model in models_list_response
            if hasattr(model, 'id') and model.id and \
               hasattr(model, 'type') and model.type == 'chat' 
        ]
        if available_chat_models:
            print(f"Found {len(available_chat_models)} chat models from Together API.")
        else:
            print("Warning: No chat models found via Together API or the list was empty.")
    except Exception as e:
        print(f"Warning: Could not fetch model list from Together API: {e}.")
    
    # Utilise une liste de modèles de secours si l'API échoue
    if not available_chat_models:
        available_chat_models = [
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "meta-llama/Llama-2-70b-chat-hf",
            "codellama/CodeLlama-70b-Instruct-hf",
            "Qwen/Qwen1.5-72B-Chat"
        ]
        print(f"Using hardcoded fallback model list ({len(available_chat_models)} models).")
    
    dataset_file_path = "../data/modelmatch_dataset.json"
    all_conversations = []
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

    if not all_conversations:
        print("No conversations available in the dataset to test.")
        exit()
    
    conversation_to_test = random.choice(all_conversations)

    test_prompts = [p['text'] for p in conversation_to_test.get("user_prompts", [])]
    initial_model_for_plugin = conversation_to_test.get("target_model_name")

    if not initial_model_for_plugin:
        print(f"Warning: 'target_model_name' is missing for conversation ID {conversation_to_test.get('id', 'N/A')}. Using a default initial model.")
        initial_model_for_plugin = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    expects_change_from_dataset = conversation_to_test.get("has_model_change", False)
    dataset_model_change_index = conversation_to_test.get("model_change_index")

    actual_model_change_index_for_simulation = None
    actual_changed_model_name_for_simulation = None

    print(f"\n--- Test setup ---")
    print(f"Testing with conversation ID: {conversation_to_test.get('id', 'N/A')}")
    print(f"Dataset 'target_model_name' (plugin's initial model): {initial_model_for_plugin}")
    print(f"Dataset 'has_model_change': {expects_change_from_dataset}")
    print(f"Dataset 'model_change_index': {dataset_model_change_index} (0-indexed, at which new model is used)")
    print(f"Number of prompts: {len(test_prompts)}")

    if not test_prompts:
        print(f"Selected conversation (ID: {conversation_to_test.get('id', 'N/A')}) has no user prompts. Skipping test.")
        exit()
    
    # Configure la simulation de changement de modèle si spécifié dans le dataset
    if expects_change_from_dataset:
        if dataset_model_change_index is not None and (0 <= dataset_model_change_index < len(test_prompts)):
            actual_model_change_index_for_simulation = dataset_model_change_index

            potential_switch_models = [m for m in available_chat_models if m != initial_model_for_plugin]
            if potential_switch_models:
                actual_changed_model_name_for_simulation = random.choice(potential_switch_models)
            else:
                print(f"Warning: No suitable different model found in the dynamic pool to switch from '{initial_model_for_plugin}'. Using a hardcoded alternative.")
                if initial_model_for_plugin != "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO":
                    actual_changed_model_name_for_simulation = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
                else:
                    actual_changed_model_name_for_simulation = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                
                if actual_changed_model_name_for_simulation == initial_model_for_plugin:
                    print(f"CRITICAL WARNING: Hardcoded fallback model '{actual_changed_model_name_for_simulation}' is THE SAME as initial model '{initial_model_for_plugin}'. Change simulation will not be effective.")
                else:
                    print(f"Using hardcoded alternative switch model: {actual_changed_model_name_for_simulation} at prompt index {actual_model_change_index_for_simulation}")
            
            print(f"SIMULATING modelswitch to: {actual_changed_model_name_for_simulation} at prompt index {actual_model_change_index_for_simulation}")
        else:
            print(f"WARNING: Conversation expects change, but 'model_change_index' ({dataset_model_change_index}) is invalid or missing. Treating as NO CHANGE for simulation.")
            expects_change_from_dataset = False
    else:
        print("NOT SIMULATING model switch (conversation does not expect one, or index is invalid).")
    
    print("--- Starting verification ---")

    custom_thresholds = None

    change_detected, detected_change_idx, reasons = plugin.verify_conversation_for_change(
        user_prompts=test_prompts,
        initial_model_name=initial_model_for_plugin,
        thresholds_config=custom_thresholds,
        actual_model_change_index=actual_model_change_index_for_simulation,
        actual_changed_model_name=actual_changed_model_name_for_simulation
    )

    print("\n--- Test result ---")
    if change_detected:
        print(f"\nPlugin result: Model change DETECTED after prompt index {detected_change_idx} (0-indexed). Reasons: {reasons}")
        if expects_change_from_dataset:
            expected_detection_idx = actual_model_change_index_for_simulation - 1 if actual_model_change_index_for_simulation is not None and actual_model_change_index_for_simulation > 0 else 0

            if detected_change_idx == expected_detection_idx:
                print(f"SUCCESS: Detected change at the correct point (after prompt {expected_detection_idx}, matching expected change before prompt {actual_model_change_index_for_simulation}).")
            else:
                print(f"PARTIAL SUCCESS/INFO: Change detected at index {detected_change_idx}, but simulated/expected change was to occur before prompt {actual_model_change_index_for_simulation} (expected detection after prompt {expected_detection_idx}).")
        else:
            print(f"FAILURE (False positive): Change detected, but no change was simulated/expected.")
    else:
        print("Plugin result: No model change detected.")
        if expects_change_from_dataset:
            print(f"FAILURE (False negative): No change detected, but a change was simulated/expected to occur before prompt index {actual_model_change_index_for_simulation}.")
        else:
            print("SUCCESS: No change detected, and no change was simulated/expected.")    