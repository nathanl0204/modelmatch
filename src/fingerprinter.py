import json
import re
from collections import Counter
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import spacy
from spacy.tokens import Doc, Span, Token
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from tqdm import tqdm
import math
import os
import logging

class BaseMetric(ABC):
    """
    Classe de base abstraite pour toutes les métriques.
    Chaque métrique doit hériter de cette classe et implémenter la méthode `calculate`.
    """
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, doc: Doc) -> Any:
        """
        Calcule la valeur de la métrique pour un Doc spaCy.
        Le Doc spaCy est utilisé pour l'efficacité, car il est pré-traité.
        """
        pass

class AvgSentenceLengthMetric(BaseMetric):
    """Calcule la longueur moyenne des phrases en tokens."""
    def __init__(self):
        super().__init__("avg_sentence_length")
    def calculate(self, doc: Doc) -> float:
        if not doc.has_annotation("SENT_START"):
            return 0.0
        num_sentences = len(list(doc.sents))
        return len(doc) / num_sentences if num_sentences > 0 else 0.0

class SentenceLengthDistributionMetric(BaseMetric):
    """Calcule les statistiques de distribution de la longueur des phrases (moyenne, écart-type, min, max)."""
    def __init__(self):
        super().__init__("sentence_length_distribution")
        self.stats_keys = ["mean", "std", "min", "max"]
    
    def get_metric_keys(self) -> List[str]:
        """Retourne les clés pour chaque statistique calculée."""
        return [f"{self.name}_{key}" for key in self.stats_keys]
    
    def calculate(self, doc: Doc) -> Dict[str, float]:
        if not doc.has_annotation("SENT_START"):
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        lengths = [len(sent) for sent in doc.sents]
        if not lengths:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": np.mean(lengths),
            "std": np.std(lengths),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths))
        }

class TTRMetric(BaseMetric):
    """Calcule le Type-Token Ratio (TTR), une mesure de la richesse lexicale."""
    def __init__(self):
        super().__init__("type_token_ratio")
    def calculate(self, doc: Doc) -> float:
        tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

class AvgWordLengthMetric(BaseMetric):
    """Calcule la longueur moyenne des mots en caractères."""
    def __init__(self):
        super().__init__("avg_word_length")
    def calculate(self, doc: Doc) -> float:
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if not tokens:
            return 0.0
        return sum(len(t) for t in tokens) / len(tokens)

class LexicalEntropyMetric(BaseMetric):
    """Calcule l'entropie lexicale, mesurant l'incertitude ou la diversité du vocabulaire."""
    def __init__(self):
        super().__init__("lexical_entropy")
    def calculate(self, doc: Doc) -> float:
        tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
        if len(tokens) < 2:
            return 0.0
        counts = Counter(tokens)
        props = [count / len(tokens) for count in counts.values()]
        return entropy(props)

class HapaxLegomenaRatioMetric(BaseMetric):
    """Calcule le ratio de hapax legomena (mots n'apparaissant qu'une seule fois)."""
    def __init__(self):
        super().__init__("hapax_legomena_ratio")
    def calculate(self, doc: Doc) -> float:
        tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
        if not tokens:
            return 0.0
        freqs = Counter(tokens)
        hapaxes = sum(1 for token in freqs if freqs[token] == 1)
        return hapaxes / len(tokens)

class PosDistributionMetric(BaseMetric):
    """Calcule la distribution des Part-of-Speech (POS) tags."""
    def __init__(self):
        super().__init__("pos_distribution")
        self.pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "ADP", "AUX", "CCONJ", "SCONJ", "DET", "NUM", "PART", "PRON"]
    
    def get_metric_keys(self) -> List[str]:
        """Retourne les clés pour chaque POS tag."""
        return [f"{self.name}_{tag}" for tag in self.pos_tags]
    
    def calculate(self, doc: Doc) -> Dict[str, float]:
        counts = doc.count_by(spacy.attrs.POS)
        total = sum(counts.values())
        if total == 0:
            return {tag: 0.0 for tag in self.pos_tags}
        
        dist = {doc.vocab.strings[tag_id]: count / total for tag_id, count in counts.items()}

        final_dist = {tag: dist.get(tag, 0.0) for tag in self.pos_tags}
        return final_dist

class SyntacticStructureFrequencyMetric(BaseMetric):
    """Calcule la fréquence de structures syntaxiques spécifiques (voix passive, propositions subordonnées)."""
    def __init__(self):
        super().__init__("syntactic_structure_frequency")
        self.structures = ["passive_ratio", "subordinate_clause_ratio"]
    
    def get_metric_keys(self) -> List[str]:
        return [f"{self.name}_{struct}" for struct in self.structures]
    
    def calculate(self, doc: Doc) -> Dict[str, float]:
        total_tokens = len(doc)
        if total_tokens == 0:
            return {"passive_ratio": 0.0, "subordinat_clause_ratio": 0.0}
        
        passives = sum(1 for token in doc if token.dep_ == "nsubjpass")
        subordinates = sum(1 for token in doc if token.dep_ in ["advcl", "ccomp", "csubj", "csubjpass"])

        return {
            "passive_ratio": passives / total_tokens,
            "subordinate_clause_ratio": subordinates / total_tokens
        }

class AvgSyntacticTreeDepthMetric(BaseMetric):
    """Calcule la profondeur moyenne des arbres de dépendance syntaxique."""
    def __init__(self):
        super().__init__("avg_syntactic_tree_depth")
    
    def _get_depth(self, token: Token, depth=0):
        """Calcule récursivement la profondeur maximale d'un sous-arbre."""
        if not token:
            return 0
        
        stack = [(token, 0)]
        max_depth = 0

        while stack:
            current_token, current_depth = stack.pop()

            children = list(current_token.children)
            if not children:
                max_depth = max(max_depth, current_depth)
            else:
                for child in children:
                    stack.append((child, current_depth + 1))
        
        return max_depth
    
    def calculate(self, doc: Doc) -> float:
        if not doc.has_annotation("SENT_START"):
            return 0.0
        depths = [self._get_depth(sent.root) for sent in doc.sents]
        return np.mean(depths) if depths else 0.0

class CoordinationVsSubordinationRatioMetric(BaseMetric):
    """Calcule le ratio entre les constructions de coordination et de subordination."""
    def __init__(self):
        super().__init__("coordination_subordination_ratio")
    def calculate(self, doc: Doc) -> float:
        coords = sum(1 for token in doc if token.dep_ == "conj")
        subords = sum(1 for token in doc if token.dep_ in ["advcl", "ccomp", "csubj"])
        return coords / subords if subords > 0 else float(coords)

class NominalizationRatioMetric(BaseMetric):
    """Calcule le ratio de nominalisations (noms dérivés de verbes ou d'adjectifs)."""
    def __init__(self):
        super().__init__("nominalization_ratio")
        self.suffixes = ("tion", "sion", "ment", "ness", "ity", "ance", "ence")
    def calculate(self, doc: Doc) -> float :
        nouns = [t for t in doc if t.pos_ == "NOUN"]
        if not nouns:
            return 0.0
        nominalized = sum(1 for n in nouns if n.lower_.endswith(self.suffixes))
        return nominalized / len(nouns)

class LogicalConnectorsMetric(BaseMetric):
    """Calcule la fréquence de différents types de connecteurs logiques."""
    def __init__(self):
        super().__init__("logical_connectors_frequency")
        self.connectors = {
            'contrast': r'\b(however|but|although|though|conversely|on the other hand|yet|still|nevertheless|in contrast|while|whereas)\b',
            'causal': r'\b(therefore|consequently|thus|hence|as a result|because|so|accordingly|for this reason)\b',
            'addition': r'\b(furthermore|moreover|in addition|also|additionally|besides|what\'s more)\b',
            'temporal': r'\b(meanwhile|afterwards|before|then|next|subsequently|finally)\b',
            'exemplification': r'\b(for example|for instance|to illustrate|such as)\b'
        }
        self.compiled_patterns = {cat: re.compile(pattern, re.I) for cat, pattern in self.connectors.items()}
    
    def get_metric_keys(self) -> List[str]:
        return [f"{self.name}_{cat}" for cat in self.connectors]

    def calculate(self, doc: Doc) -> Dict[str, float]:
        total_tokens = len(doc)
        if total_tokens == 0:
            return {k: 0.0 for k in self.connectors}
        
        text = doc.text.lower()
        counts = {cat: len(pattern.findall(text)) for cat, pattern in self.compiled_patterns.items()}
        return {cat: count / total_tokens for cat, count in counts.items()}

class ImpersonalConstructionsMetric(BaseMetric):
    """Calcule le taux de constructions impersonnelles (ex: "is is said that...")."""
    def __init__(self):
        super().__init__("impersonal_constructions_rate")
        self.patterns = [
            re.compile(r"\bit is (?:necessary|possible|important|clear|true|evident|argued|believed|known|said|thought|understood) that\b", re.I),
            re.compile(r"\bone (?:might|could|can|should) (?:say|argue|suggest|conclude|posit)\b", re.I),
            re.compile(r"\b(it can be|it could be|it should be) (?:seen|argued|noted|observed|stated)\b", re.I),
            re.compile(r"\bthere is (?:a tendency|a possibility|evidence|reason to believe)\b", re.I)
        ]
    def calculate(self, doc: Doc) -> float:
        if not doc.text: return 0.0
        matches = sum(len(p.findall(doc.text)) for p in self.patterns)
        return matches / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0.0

class PolitenessMetric(BaseMetric):
    """Calcule un score de politesse basé sur la présence de mots et phrases polis."""
    def __init__(self):
        super().__init__("politeness_score")
        self.polite_words = {
            "please", "thank", "thanks", "sorry", "excuse", "pardon", 
            "appreciate", "grateful", "kindly", "apologies", "apologize"
        }
        self.polite_phrases = [
            re.compile(r"\b(could|would|can|will) you\b", re.I),
            re.compile(r"\bif you (?:don't|do not) mind\b", re.I),
            re.compile(r"\b(I'd|I would) be grateful\b", re.I)
        ]
    def calculate(self, doc: Doc) -> int:
        word_count = sum(1 for token in doc if token.lower_ in self.polite_words)
        phrase_count = sum(len(p.findall(doc.text)) for p in self.polite_phrases)
        return word_count + phrase_count

class FormalityMetric(BaseMetric):
    """Calcule un score de formalité basé sur la fréquence des contractions (un score élevé indique une plus grande informalité)."""
    def __init__(self):
        super().__init__("formality_score_contractions")
        self.contractions = {
            "ain't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", 
            "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "i'd", "i'll", 
            "i'm", "i've", "isn't", "it's", "let's", "mightn't", "mustn't", 
            "shan't", "she'd", "she'll", "she's", "shouldn't", "that's", "there's", 
            "they'd", "they'll", "they're", "they've", "we'd", "we're", "we've", 
            "weren't", "what'll", "what're", "what's", "what've", "where's", 
            "who'd", "who'll", "who're", "who's", "who've", "won't", "wouldn't", 
            "you'd", "you'll", "you're", "you've"
        }
    def calculate(self, doc: Doc) -> float:
        tokens = [t for t in doc if not t.is_space]
        if not tokens: return 0.0
        contraction_count = sum(1 for t in tokens if t.lower_ in self.contractions)
        return contraction_count / len(tokens)

class SelfPositioningMetric(BaseMetric):
    """Calcule le taux d'expressions de positionnement personnel (ex: "I think", "in my opinion", "as an AI")."""
    def __init__(self):
        super().__init__("self_positioning_rate")
        self.patterns = [
            re.compile(r"\bI (?:think|believe|feel|suppose|assume|guess|find|would say|would argue|contend|reckon|consider)\b", re.I),
            re.compile(r"\b(in my opinion|from my perspective|to my mind|it seems to me|as far as I'm concerned)\b", re.I),
            re.compile(r"\b(as an AI|as a large language model|as an AI assistant|my purpose is|my knowledge is)\b", re.I),
        ]
    def calculate(self, doc: Doc) -> float:
        if not doc.text: return 0.0
        matches = sum(len(p.findall(doc.text)) for p in self.patterns)
        return matches / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0.0

class FleschKincaidReadabilityMetric(BaseMetric):
    """Calcule le score de lisibilité Flesch-Kincaid, estimant le niveau scolaire nécessaire pour comprendre le texte."""
    def __init__(self):
        super().__init__("flesch_kincaid_grade")
        self.vowel_pattern = re.compile(r"[aeiouy]+", re.I)
        self.exception_pattern = re.compile(r"(?:es|ed)$", re.I)
    
    def _count_syllables(self, word: str) -> int:
        """Compte les syllabes dans un mot avec une méthode heuristique."""
        if len(word) <= 3:
            return 1
        
        word = self.exception_pattern.sub('', word)

        vowel_groups = self.vowel_pattern.findall(word)
        count = len(vowel_groups)

        if word.endswith('e') and not word.endswith('le') and count > 1:
            count -= 1
        
        return max(1, count)

    def calculate(self, doc: Doc) -> float:
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        num_tokens = len(tokens)
        num_sentences = len(list(doc.sents))
        num_syllables = sum(self._count_syllables(t.text) for t in tokens)

        if num_tokens == 0 or num_sentences == 0:
            return 0.0
        
        # Formule du Flesch-Kincaid Grade Level
        grade = 0.39 * (num_tokens / num_sentences) + 11.8 * (num_syllables / num_tokens) - 15.59
        return grade

class NERMetric(BaseMetric):
    """Calcule la densité et la diversité des entités nommées (NER)."""
    def __init__(self):
        super().__init__("ner_metrics")
    
    def get_metric_keys(self) -> List[str]:
        return [f"{self.name}_density", f"{self.name}_type_diversity"]
    
    def calculate(self, doc: Doc) -> Dict[str, float]:
        num_tokens = len([t for t in doc if not t.is_space])
        if num_tokens == 0:
            return {"density": 0.0, "type_diversity": 0.0}
        
        num_entities = len(doc.ents)
        entity_types = set(ent.label_ for ent in doc.ents)

        return {
            "density": num_entities / num_tokens,
            "type_diversity": len(entity_types)
        }

class QuestionRateMetric(BaseMetric):
    """Calcule le ratio de phrases se terminant par un point d'interrogation."""
    def __init__(self):
        super().__init__("question_rate")
    
    def calculate(self, doc: Doc) -> float:
        sentences = list(doc.sents)
        if not sentences:
            return 0.0
        question_count = sum(1 for sent in sentences if sent.text.strip().endswith('?'))
        return question_count / len(sentences)

class NumericalExpressionMetric(BaseMetric):
    """Calcule le ratio de tokens qui ressemblent à des nombres."""
    def __init__(self):
        super().__init__("numerical_expression_rate")
    
    def calculate(self, doc: Doc) -> float:
        tokens = [t for t in doc if not t.is_space]
        if not tokens:
            return 0.0
        numerical_count = sum(1 for t in tokens if t.like_num)
        return numerical_count / len(tokens)

class PunctuationDistributionMetric(BaseMetric):
    """Calcule la distribution de différents types de ponctuation."""
    def __init__(self):
        super().__init__("punctuation_distribution")
        self.punctuations = {',': 'comma', '.': 'dot', ';': 'semicolon', ':': 'colon', '!': 'exclamation', '?': 'question', '-': 'hyphen', '(': 'lparen', ')': 'rparen', '"': 'quote', "'": 'apostrophe'}
    
    def get_metric_keys(self) -> List[str]:
        return [f"{self.name}_{name}" for name in self.punctuations.values()]
    
    def calculate(self, doc: Doc) -> Dict[str, float]:
        total_tokens = len([t for t in doc if not t.is_space])
        if total_tokens == 0:
            return {name: 0.0 for name in self.punctuations.values()}
        
        counts = Counter(t.text for t in doc if t.text in self.punctuations)

        dist = {}
        for punc_char, punc_name in self.punctuations.items():
            dist[punc_name] = counts.get(punc_char, 0) / total_tokens

        return dist

class ModalVerbFrequencyMetric(BaseMetric):
    """Calcule la fréquence des verbes modaux (can, could, may, might, etc.)."""
    def __init__(self):
        super().__init__("modal_verb_frequency")
    
    def calculate(self, doc: Doc) -> float:
        num_sentences = len(list(doc.sents))
        if num_sentences == 0:
            return 0.0
        
        modal_verbs = sum(1 for t in doc if t.pos_ == 'AUX' and t.tag_ == 'MD')
        return modal_verbs / num_sentences

class RepetitionScoreMetric(BaseMetric):
    """Calcule un score de répétition basé sur les n-grammes."""
    def __init__(self, n: int = 3):
        super().__init__(f"{n}-gram_repetition_score")
        self.n = n
    
    def calculate(self, doc: Doc) -> float:
        tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
        if len(tokens) < self.n:
            return 0.0
        
        ngrams = [" ".join(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1)]
        if not ngrams:
            return 0.0
        
        total_ngrams = len(ngrams)
        unique_ngrams = len(set(ngrams))

        return (total_ngrams - unique_ngrams) / total_ngrams

class ZipfLawSlopeMetric(BaseMetric):
    """Calcule la pente de la loi de Zipf, qui décrit la relation entre la fréquence et le rang d'un mot."""
    def __init__(self):
        super().__init__("zipf_law_slope")
    
    def calculate(self, doc: Doc) -> float:
        tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
        if len(tokens) < 2:
            return 0.0
        
        freqs = Counter(tokens)
        if not freqs:
            return 0.0
        
        sorted_freqs = sorted(freqs.values(), reverse=True)
        ranks = np.arange(1, len(sorted_freqs) + 1)

        log_ranks = np.log(ranks)
        log_freqs = np.log(sorted_freqs)

        if len(log_ranks) < 2:
            return 0.0
        
        try:
            slope, _ = np.polyfit(log_ranks, log_freqs, 1)
        except np.linalg.LinAlgError:
            slope = 0.0
        
        return slope

class DiscourseMarkersMetric(BaseMetric):
    """Calcule le taux de marqueurs discursifs."""
    def __init__(self):
        super().__init__("discourse_markers_rate")
    
    def calculate(self, doc: Doc) -> float:
        num_sentences = len(list(doc.sents))
        if num_sentences == 0 or not doc.text:
            return 0.0

        discourse_markers_count = sum(1 for token in doc if token.dep_ == "discourse")

        return discourse_markers_count / num_sentences

class AddresseeFocusMetric(BaseMetric):
    """Calcule le taux de focus sur l'interlocuteur (pronoms de la 2ème personne et impératifs)."""
    def __init__(self):
        super().__init__("addressee_focus_rate")
        self.addressee_pronouns = {"you", "your", "yours", "yourself"}
    
    def calculate(self, doc: Doc) -> float:
        tokens = [t for t in doc if not t.is_space]
        if not tokens:
            return 0.0
        
        addressee_pronoun_count = sum(1 for t in tokens if t.lower_ in self.addressee_pronouns)

        imperative_count = 0
        for sent in doc.sents:
            # Impératif = forme verbale basique et pas de sujet
            has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
            if sent.root.tag_ == "VB" and not has_subject:
                imperative_count += 1
        
        total_focus_count = addressee_pronoun_count + imperative_count
        return total_focus_count / len(tokens)

class CodeKeywordFrequencyMetric(BaseMetric):
    """Calcule la fréquence des mots-clés de programmation."""
    def __init__(self):
        super().__init__("code_keyword_frequency")
        self.keywords = {
            'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while', 
            'try', 'except', 'finally', 'with', 'as', 'lambda', 'yield', 'async', 'await',
            'function', 'var', 'let', 'const', 'export', 'new', 'this',
            'public', 'private', 'protected', 'static', 'void', 'int', 'String', 'float'
        }

    def calculate(self, doc: Doc) -> float:
        tokens = [t.lower_ for t in doc if not t.is_space]
        if not tokens:
            return 0.0
        keyword_count = sum(1 for token in tokens if token in self.keywords)
        return keyword_count / len(tokens)

class CommentRatioMetric(BaseMetric):
    """Calcule le ratio de lignes de code qui sont des commentaires."""
    def __init__(self):
        super().__init__("comment_ratio")
        self.comment_patterns = [
            re.compile(r'^\s*#'),  # Python, Ruby, etc.
            re.compile(r'^\s*//'), # JavaScript, Java, C++, etc.
        ]
    
    def calculate(self, doc: Doc) -> float:
        lines = doc.text.splitlines()
        if not lines:
            return 0.0

        comment_lines = 0
        for line in lines:
            if any(pattern.match(line) for pattern in self.comment_patterns):
                comment_lines += 1
        
        return comment_lines / len(lines)

class CodeNestingDepthMetric(BaseMetric):
    """Estime la profondeur maximale d'imbrication du code en se basant sur l'indentation."""
    def __init__(self, tab_size=4):
        super().__init__("code_nesting_depth")
        self.tab_size = tab_size
    
    def calculate(self, doc: Doc) -> float:
        lines = doc.text.splitlines()
        if not lines:
            return 0.0
        
        max_depth = 0
        for line in lines:
            stripped_line = line.lstrip()
            if not stripped_line:
                continue

            indentation = len(line) - len(stripped_line)
            depth = indentation / self.tab_size
            if depth > max_depth:
                max_depth = depth
        return max_depth
    
class NamingConventionMetric(BaseMetric):
    """Calcule la distribution des conventions de nommage (snake_case, camelCase, PascalCase)."""
    def __init__(self):
        super().__init__("naming_convention_dist")
        self.snake_case = re.compile(r'^[a-z0-9_]+$')
        self.camel_case = re.compile(r'^[a-z]+[a-zA-Z0-9]*$')
        self.pascal_case = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
    
    def get_metric_keys(self) -> List[str]:
        return [f"{self.name}_{case}" for case in ["snake", "camel", "pascal"]]

    def calculate(self, doc: Doc) -> Dict[str, float]:
        potential_vars = [
            t.text for t in doc
            if t.is_alpha and not t.is_stop and not t.like_num
        ]
        if not potential_vars:
            return {"snake": 0.0, "camel": 0.0, "pascal": 0.0}
        
        counts = Counter()
        for var in potential_vars:
            if self.pascal_case.match(var) and not self.camel_case.match(var):
                counts['pascal'] += 1
            elif self.snake_case.match(var) and '_' in var:
                counts['snake'] += 1
            elif self.camel_case.match(var):
                counts['camel'] += 1
        
        total = sum(counts.values())
        if total == 0:
            return {"snake": 0.0, "camel": 0.0, "pascal": 0.0}
        
        return {k: v / total for k, v in counts.items()}

class StringLiteralRatioMetric(BaseMetric):
    """Calcule le ratio de chaînes de caractères littérales dans le code."""
    def __init__(self):
        super().__init__("string_literal_ratio")
        self.string_pattern = re.compile(r"(\".*?\"|\'.*?\')")
    
    def calculate(self, doc: Doc) -> float:
        num_tokens = len([t for t in doc if not t.is_space])
        if num_tokens == 0:
            return 0.0
        
        string_literals = self.string_pattern.findall(doc.text)
        return len(string_literals) / num_tokens

class MagicNumberRatioMetric(BaseMetric):
    """Calcule le ratio de "nombres magiques" (nombres codés en dur qui ne sont pas des assignations, etc.)."""
    def __init__(self):
        super().__init__("magic_number_ratio")
        # Cible les nombres qui ne sont pas à côté de certains opérateurs ou mots-clés
        self.magic_number_pattern = re.compile(r'(?<![=,\[\(\s])\b\d+\b(?![;,\]\)\s])')
    
    def calculate(self, doc: Doc) -> float:
        tokens = [t for t in doc if not t.is_space]
        if not tokens:
            return 0.0
        
        magic_numbers = len(self.magic_number_pattern.findall(doc.text))

        return magic_numbers / len(tokens)

def get_model_family(model_name: str) -> str:
    """Détermine la famille d'un modèle à partir de son nom."""
    model_name = model_name.lower()
    if 'llama-3' in model_name:
        return 'Llama-3'
    if 'gemma' in model_name:
        return 'Gemma'
    if 'qwen2' in model_name:
        return 'Qwen2'
    if 'phi-3' in model_name:
        return 'Phi-3'
    if 'mistral' in model_name:
        return 'Mistral'
    if 'deepseek' in model_name:
        return 'Deepseek'
    if 'gpt-4o' in model_name:
        return 'GPT-4o'
    return model_name.split('/')[0]

def get_text_metrics():
    """Retourne une liste d'instances de toutes les métriques textuelles."""
    return [
        SentenceLengthDistributionMetric(),
        TTRMetric(),
        AvgWordLengthMetric(),
        LexicalEntropyMetric(),
        HapaxLegomenaRatioMetric(),
        PosDistributionMetric(),
        SyntacticStructureFrequencyMetric(),
        AvgSyntacticTreeDepthMetric(),
        CoordinationVsSubordinationRatioMetric(),
        NominalizationRatioMetric(),
        LogicalConnectorsMetric(),
        ImpersonalConstructionsMetric(),
        PolitenessMetric(),
        FormalityMetric(),
        SelfPositioningMetric(),
        FleschKincaidReadabilityMetric(),
        NERMetric(),
        QuestionRateMetric(),
        NumericalExpressionMetric(),
        PunctuationDistributionMetric(),
        ModalVerbFrequencyMetric(),
        RepetitionScoreMetric(),
        ZipfLawSlopeMetric(),
        DiscourseMarkersMetric(),
        AddresseeFocusMetric()
    ]

def get_code_metrics():
    """Retourne une liste d'instances de toutes les métriques de code."""
    return [
        CodeKeywordFrequencyMetric(),
        CommentRatioMetric(),
        CodeNestingDepthMetric(),
        NamingConventionMetric(),
        StringLiteralRatioMetric(),
        MagicNumberRatioMetric()
    ]

def get_interaction_metrics():
    """Fonction de fabrique pour les métriques d'interaction (actuellement gérée dans la classe Fingerprinter)."""
    pass
    
def is_code_response(text: str) -> bool:
    """Détermine si une réponse contient principalement du code en utilisant plusieurs heuristiques."""
    # Heuristique 1 : Présence de blocs de code Markdown
    if re.search(r"```.*```", text, re.DOTALL):
        return True
    
    # Heuristique 2 : Densité de mots-clés de programmation
    keywords = {'def', 'class', 'import', 'function', 'const', 'let', 'public', 'static', 'return'}
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return False
    
    keyword_count = sum(1 for word in words if word in keywords)
    if keyword_count / len(words) > 0.05:
        return True
    
    # Heuristique 3 : Présence de multiples accolades ou d'opérateurs spécifiques
    if text.count('{') > 2 or text.count('=>') > 0 or text.count('->') > 0:
        return True
    
    return False

def extract_code_from_response(text: str) -> str:
    """Extrait le contenu des blocs de code Markdown d'un texte."""
    code_blocks = re.findall(r"```(?:[a-zA-Z0-9]*)?\n(.*?)\n```", text, re.DOTALL)
    return "\n".join(code_blocks)

class InteractionMetrics:
    """Calcule les métriques basées sur la relation entre un prompt et une réponse."""
    def __init__(self, nlp):
        self.nlp = nlp
        self.ttr_metric = TTRMetric()
    
    def get_metric_keys(self) -> List[str]:
        """Retourne les clés pour chaque métrique d'interaction."""
        return ["length_ratio", "ttr_ratio", "ner_jaccard_similarity", "embedding_similarity"]
    
    def calculate(self, prompt_doc: Doc, response_doc: Doc) -> Dict[str, float]:
        """Calcule toutes les métriques d'interaction."""
        prompt_len = len(prompt_doc.text)
        response_len = len(response_doc.text)
        length_ratio = response_len / prompt_len if prompt_len > 0 else 0.0

        prompt_ttr = self.ttr_metric.calculate(prompt_doc)
        response_ttr = self.ttr_metric.calculate(response_doc)
        ttr_ratio = response_ttr / prompt_ttr if prompt_ttr > 0 else 0.0

        prompt_ents = {ent.text.lower() for ent in prompt_doc.ents}
        response_ents = {ent.text.lower() for ent in response_doc.ents}
        intersection = len(prompt_ents.intersection(response_ents))
        union = len(prompt_ents.union(response_ents))
        ner_jaccard = intersection / union if union > 0 else 0.0

        if prompt_doc.has_vector and response_doc.has_vector and prompt_doc.vector_norm and response_doc.vector_norm:
            embedding_similarity = prompt_doc.similarity(response_doc)
        else:
            embedding_similarity = 0.0
        
        return {
            "length_ratio": length_ratio,
            "ttr_ratio": ttr_ratio,
            "ner_jaccard_similarity": ner_jaccard,
            "embedding_similarity": embedding_similarity
        }

class Fingerprinter:
    """
    Classe principale pour orchestrer la génération d'empreintes stylistiques.
    """
    def __init__(self, text_metrics: List[BaseMetric], code_metrics: List[BaseMetric], spacy_model: str = "en_core_web_md"):
        self.text_metrics = text_metrics
        self.code_metrics = code_metrics
        print(f"Loading spaCy model '{spacy_model}'...")
        self.nlp = spacy.load(spacy_model)
        self.interaction_metrics = InteractionMetrics(self.nlp)
        print("spaCy model loaded.")
    
    def get_all_metric_names(self) -> List[str]:
        """Retourne une liste complète de tous les noms de métriques possibles."""
        metric_names = []

        for metric in self.text_metrics + self.code_metrics:
            if hasattr(metric, 'get_metric_keys'):
                metric_names.extend(metric.get_metric_keys())
            else:
                metric_names.append(metric.name)
        
        metric_names.extend(self.interaction_metrics.get_metric_keys())

        metric_names.append("is_code")

        return sorted(list(set(metric_names)))
    
    def create_fingerprint(self, text: str, prompt_text: Optional[str] = None) -> Dict[str, Any]:
        """Crée une empreinte stylistique unique pour un texte donné."""
        fingerprint = {}

        # Initialise toutes les clés de métriques à 0.0 pour garantir une structure cohérente
        for metric in self.text_metrics + self.code_metrics:
            # Cas des métriques retournant un dictionnaire de valeurs
            if hasattr(metric, 'get_metric_keys'):
                for key in metric.get_metric_keys():
                    fingerprint[key] = 0.0
            else:
                fingerprint[metric.name] = 0.0
        
        for key in self.interaction_metrics.get_metric_keys():
            fingerprint[key] = 0.0

        # Sépare le contenu textuel du contenu de code
        code_block_pattern = re.compile(r"```(?:[a-zA-Z0-9]*)?\n.*?\n```", re.DOTALL)
        code_content = extract_code_from_response(text)
        text_parts = code_block_pattern.split(text)
        text_content = "\n".join(p.strip() for p in text_parts if p.strip())

        text_doc = self.nlp(text_content if text_content else " ")
        
        if text_content:
            for metric in self.text_metrics:
                result = metric.calculate(text_doc)
                if isinstance(result, dict):
                    for k, v in result.items():
                        fingerprint[f"{metric.name}_{k}"] = v
                else:
                    fingerprint[metric.name] = result
        
        if prompt_text:
            prompt_doc = self.nlp(prompt_text)
            interaction_results = self.interaction_metrics.calculate(prompt_doc, text_doc)
            fingerprint.update(interaction_results)
        
        fingerprint["is_code"] = 1.0 if code_content else 0.0
        if code_content:
            code_doc = self.nlp(code_content)
            for metric in self.code_metrics:
                result = metric.calculate(code_doc)
                if isinstance(result, dict):
                    for k, v in result.items():
                        fingerprint[f"{metric.name}_{k}"] = v
                else:
                    fingerprint[metric.name] = result
        
        return fingerprint
    
    def process_dataset(self, dataset_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Traite un fichier de dataset entier pour générer des empreintes pour chaque réponse de modèle."""
        print(f"Loading dataset from {dataset_path}...")
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Dataset file not found at {dataset_path}")
            if logger:
                logger.error(f"Dataset file not found at {dataset_path}")
            return {}
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {dataset_path}")
            if logger:
                logger.error(f"Could not decode JSON from {dataset_path}")
            return {}
        
        all_fingerprints = {}
        conversations = data.get('conversations', [])

        """ clarity_mapping = {"unclear": 1, "ambiguous": 2, "acceptable": 3, "clear": 4}
        complexity_mapping = {"low": 1, "medium": 2, "high": 3} """

        print(f"Processing {len(conversations)} conversations...")
        for conv in tqdm(conversations, desc="Generating fingerprints"):
            conv_id = conv.get("id", "unknown_conv_id")
            user_prompts = conv.get("user_prompts", [])
            model_responses = conv.get('model_responses', {})
            if not model_responses and logger:
                logger.warning(f"Conv ID {conv_id}: No 'model_responses' found in conversation.")

            for model_name, responses in conv.get('model_responses', {}).items():
                if model_name not in all_fingerprints:
                    all_fingerprints[model_name] = []

                if not responses and logger:
                    logger.warning(f"Conv ID {conv_id}, model '{model_name}': 'responses' list is empty.")
                
                for i, response_text in enumerate(responses):
                    if not response_text or not isinstance(response_text, str):
                        if logger:
                            logger.warning(f"Conv ID {conv_id}, model '{model_name}', prompt index {i}: invalid or empty response text. Skipping fingerprint generation.")
                        continue

                    prompt_text = None
                    if i < len(user_prompts):
                        prompt_text = user_prompts[i].get('text')

                    try:
                        fp = self.create_fingerprint(response_text, prompt_text)
                        fp['conversation_id'] = conv_id
                        fp['prompt_index'] = i

                        """ if i < len(user_prompts):
                            prompt_info = user_prompts[i]
                            clarity_str = prompt_info.get('clarity')
                            complexity_str = prompt_info.get('complexity')

                            fp['prompt_clarity'] = clarity_mapping.get(clarity_str, 0)
                            fp['prompt_complexity'] = complexity_mapping.get(complexity_str, 0)
                        else:
                            fp['prompt_clarity'] = 0
                            fp['prompt_complexity'] = 0 """

                        all_fingerprints[model_name].append(fp)
                    except Exception as e:
                        if logger:
                            logger.error(f"Conv ID {conv_id}, model '{model_name}', prompt index {i}: exception during fingerprint creation: {e}", exc_info=True)
        
        print("Fingerprint generation complete.")
        return all_fingerprints

class ModelProfiler:
    """
    Crée et visualise des profils moyens pour les modèles basés sur leurs empreintes.
    ATTENTION : La réprésentation est peu lisible c'est pourquoi les figures générées ont été retirées.
    """
    def __init__(self, fingerprints_by_model: Dict[str, List[Dict[str, Any]]]):
        self.profiles = {}
        for model_name, fingerprints in fingerprints_by_model.items():
            if fingerprints:
                df = pd.DataFrame(fingerprints)
                numeric_df = df.select_dtypes(include=np.number)
                self.profiles[model_name] = numeric_df.mean().to_dict()

    def get_average_profile(self, model_name: str) -> Optional[Dict[str, float]]:
        """Retourne le profil moyen pour un modèle donné."""
        return self.profiles.get(model_name)

    def plot_profile_comparison(self, model_names: List[str], save_path: str = "model_profiles_comparison.png"):
        """Génère et sauvegarde des graphiques "spider" (radar) pour comparer les profils de modèles."""
        profiles_to_plot = {name: self.profiles[name] for name in model_names if name in self.profiles}
        if not profiles_to_plot:
            print("None of the requested models have profiles to plot.")
            return

        df = pd.DataFrame(profiles_to_plot)
        df = df.drop(index=['is_code', 'prompt_index'], errors='ignore')
        df_normalized = (df - df.min()) / (df.max() - df.min())
        df_normalized = df_normalized.fillna(0)

        labels = df_normalized.index
        labels = [label.replace('_', ' ').title() for label in labels]
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        save_dir, base_filename = os.path.split(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename, file_extension = os.path.splitext(base_filename)
        if not file_extension:
            file_extension = ".png"

        for model_name in df_normalized.columns:
            fig, ax = plt.subplots(figsize=(22, 22), subplot_kw=dict(polar=True))

            values = df_normalized[model_name].tolist()
            values += values[:1]

            ax.plot(angles, values, label=model_name, linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=10)
            ax.set_title(f"Profile for {model_name}", size=16, y=1.1)
            
            safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
            model_save_path = os.path.join(save_dir, f"{filename}_{safe_model_name}{file_extension}")

            plt.tight_layout(pad=3.0)
            plt.savefig(model_save_path, bbox_inches='tight')
            print(f"Spider plot for {model_name} saved to {model_save_path}")
            plt.close(fig)


if __name__ == "__main__":
    log_file = "fingerprint_generation.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    logger.info("--- Starting fingerprinter script ---")

    text_metrics_to_use = get_text_metrics()
    code_metrics_to_use = get_code_metrics()

    fingerprinter = Fingerprinter(
        text_metrics=text_metrics_to_use,
        code_metrics=code_metrics_to_use
    )

    dataset_path = "../data/modelmatch_dataset_reduced_no_change.json"
    logger.info(f"Processing dataset: {dataset_path}")
    fingerprints = fingerprinter.process_dataset(dataset_path, logger=logger)

    if not fingerprints:
        logger.warning("No fingerprints were generated. Exiting script.")
        print("No fingerprints were generated. Check fingerprint_generation.log for details.")
        exit()

    profiler = ModelProfiler(fingerprints)
    models_to_compare = list(profiler.profiles.keys())
    if models_to_compare:
        print(f"\nGenerating comparison plot for: {', '.join(models_to_compare)}")
        profiler.plot_profile_comparison(models_to_compare, save_path="../data/model_profiles/model_profile.png")
    else:
        print("No model profiles were generated to create a comparison plot.")

    print("\nTo create a CSV for a classifier:")
    all_data = []
    for model_name, fps in fingerprints.items():
        for fp in fps:
            row = fp.copy()
            row['model_name'] = model_name
            all_data.append(row)
    
    if all_data:
        df_classifier = pd.DataFrame(all_data)
        csv_path = "../data/fingerprints_for_classification.csv"
        df_classifier.to_csv(csv_path, index=False)
        print(f"CSV dataset for classifier saved to {csv_path}")
        logger.info(f"CSV dataset for classifier saved to {csv_path}")
    
    logger.info("--- Fingerprinter script finished ---")