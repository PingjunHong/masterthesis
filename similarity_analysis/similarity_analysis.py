import json
from collections import defaultdict
from collections import Counter
from itertools import combinations
from tqdm import tqdm

# scorer
from functools import cache

import nltk
import numpy as np
import spacy
import torch
from nltk import ngrams
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
import torch.nn.functional as F
from scipy.spatial.distance import cosine, euclidean
from typing import List


class Scorer:
    def __init__(self, lang: str):
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.spacy_processor = (
            spacy.load("en_core_web_sm") if lang == "en-sent" else spacy.load("de_core_news_md")
        )
        self.tokenizer = (
            AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1",use_auth_token=None)
            if lang == "en-sent"
            else AutoTokenizer.from_pretrained(
                "sentence-transformers/distiluse-base-multilingual-cased-v1"
            )
        )
        self.model = (
            AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1", use_auth_token=None)
            .eval()
            .to(self.device)
            if lang == "en-sent"
            else AutoModelWithLMHead.from_pretrained(
                "sentence-transformers/distiluse-base-multilingual-cased-v1"
            )
            .eval()
            .to(self.device)
        )
        self.lang = lang

    @cache
    def _tokenize(self, s: str, max_len: int = None):
        # Cut string at the LM sub-word token length to mimic the generation setting
        if max_len:
            s = self.tokenizer.decode(
                self.tokenizer(
                    s,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                ).input_ids
            )

        doc = self.spacy_processor(s.strip())
        return doc

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.hidden_states[-1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def ngram_overlap(
        self,
        string1: str,
        string2: str,
        max_len1: int,
        max_len2: int,
        n: int,
        pos: bool = False,
    ):
        tokenized_string1 = self._tokenize(string1, max_len=max_len1)
        ngrams1 = list(
            ngrams(
                [token.pos_ if pos else token.text.lower() for token in tokenized_string1],
                n,
            )
        )
        tokenized_string2 = self._tokenize(string2, max_len=max_len2)
        ngrams2 = list(
            ngrams(
                [token.pos_ if pos else token.text.lower() for token in tokenized_string2],
                n,
            )
        )

        count_1_in_2 = sum([1 if ngram2 in ngrams1 else 0 for ngram2 in ngrams2])
        count_2_in_1 = sum([1 if ngram1 in ngrams2 else 0 for ngram1 in ngrams1])
        combined_length = len(ngrams1) + len(ngrams2)
        return (
            (count_1_in_2 + count_2_in_1) / combined_length if combined_length > 0 else float("nan")
        )

    def compute_embeddings(self, strings: List[str], max_len: int):
        tokenized_strings = [self._tokenize(string, max_len=max_len).text for string in strings]
        encoded_input = self.tokenizer(tokenized_strings, padding=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.device), output_hidden_states=True)
        batch_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        return F.normalize(batch_embeddings, p=2, dim=1).to("cpu").numpy()

    @staticmethod
    def cosine_similarity(embed1: np.ndarray, embed2: np.ndarray):
        return -cosine(embed1, embed2) + 1

    @staticmethod
    def euclidean_similarity(embed1: np.ndarray, embed2: np.ndarray):
        return 1 / (1 + euclidean(embed1, embed2))

    def length(self, string: str):
        doc = self._tokenize(string)
        return len([token for token in doc])

# load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# group explanations by category and NLI label
def group_by_pair_label_category(data):
    group_to_explanations = defaultdict(list)
    for item in data:
        explanation = item.get("explanation", "").strip()
        pair_id = item.get("pairID", "").strip()
        gold_label = item.get("gold_label", "").strip()
        categories = item.get("explanation_category", [])

        for category in categories:
            if explanation:
                group_to_explanations[(pair_id, category, gold_label)].append(explanation)
    return group_to_explanations

# pairwise similarity metrics
def compute_all_metrics(explanations, scorer, max_len=128):
    if len(explanations) < 2:
        return {
            "cosine": float("nan"),
            "euclidean": float("nan"),
            "unigram_overlap": float("nan"),
            "bigram_overlap": float("nan"),
            "pos_unigram_overlap": float("nan"),
            "pos_bigram_overlap": float("nan"),
        }


    embeddings = scorer.compute_embeddings(explanations, max_len=max_len)

    cosine_total = 0.0
    euclidean_total = 0.0
    unigram_total = 0.0
    bigram_total = 0.0
    pos_unigram_total = 0.0
    pos_bigram_total = 0.0
    count = 0
    
    bigram_count = 0
    pos_bigram_count = 0

    for i, j in combinations(range(len(explanations)), 2):
        s1 = explanations[i]
        s2 = explanations[j]

        cosine_total += scorer.cosine_similarity(embeddings[i], embeddings[j])
        euclidean_total += scorer.euclidean_similarity(embeddings[i], embeddings[j])

        unigram_total += scorer.ngram_overlap(s1, s2, max_len, max_len, n=1, pos=False)
        bigram = scorer.ngram_overlap(s1, s2, max_len, max_len, n=2, pos=False)
        if not np.isnan(bigram):
            bigram_total += bigram
            bigram_count += 1

        pos_unigram = scorer.ngram_overlap(s1, s2, max_len, max_len, n=1, pos=True)
        pos_unigram_total += pos_unigram

        pos_bigram = scorer.ngram_overlap(s1, s2, max_len, max_len, n=2, pos=True)
        if not np.isnan(pos_bigram):
            pos_bigram_total += pos_bigram
            pos_bigram_count += 1

        count += 1

    return {
        "cosine": cosine_total / count,
        "euclidean": euclidean_total / count,
        "unigram_overlap": unigram_total / count,
        "bigram_overlap": bigram_total / bigram_count if bigram_count > 0 else float("nan"),
        "pos_unigram_overlap": pos_unigram_total / count,
        "pos_bigram_overlap": pos_bigram_total / count if pos_bigram_count > 0 else float("nan"),
    }

def print_all_category_label_combinations(data):
    counter = Counter()
    for item in data:
        gold_label = item.get("gold_label", "").strip()
        categories = item.get("explanation_category", [])
        for category in categories:
            counter[(category, gold_label)] += 1
    print("\nAll (category, label) combinations with counts:")
    for k, v in sorted(counter.items()):
        print(f"{k}: {v}")
        
def save_results_to_jsonl(results, output_path):
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            safe_item = {k: convert(v) for k, v in item.items()}
            f.write(json.dumps(safe_item, ensure_ascii=False) + '\n')

def main(jsonl_path):
    print("Loading data")
    data = load_jsonl(jsonl_path)

    print("Grouping explanations by category and gold_label")
    grouped_explanations = group_by_pair_label_category(data)
    
    print_all_category_label_combinations(data)

    print("Initializing scorer")
    scorer = Scorer(lang="en-sent")

    print("Computing all similarity metrics per category...")
    results = []

    category_label_metric_totals = defaultdict(lambda: defaultdict(list))

    for (pair_id, category, gold_label), explanations in tqdm(grouped_explanations.items()):
        if len(explanations) < 2:
            print(f"Skipped group (pairID={pair_id}, category={category}, label={gold_label}) due to < 2 explanations")
            continue
        
        metrics = compute_all_metrics(explanations, scorer)
        
        print(f"\nMetrics for (pairID={pair_id}, category={category}, label={gold_label}):")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}" if not np.isnan(value) else f"  {metric_name}: NaN")
            
        result_entry = {
            "pair_id": pair_id,
            "category": category,
            "gold_label": gold_label,
            "num_explanations": len(explanations),
            **metrics
        }
        results.append(result_entry)

        for metric_name, metric_value in metrics.items():
            if not np.isnan(metric_value):
                category_label_metric_totals[(category, gold_label)][metric_name].append(metric_value)

    print("\n=== Final Averaged Metrics per (category, label) across pairIDs ===")
    for (category, gold_label), metric_lists in category_label_metric_totals.items():
        print(f"\nCategory: {category} | Label: {gold_label}")
        for metric_name, values in metric_lists.items():
            avg = np.mean(values)
            print(f"  {metric_name}: {avg:.4f}")

    save_path = "similarity_per_instance_results.jsonl"
    save_results_to_jsonl(results, save_path)
    print(f"\nSaved per-instance similarity results to: {save_path}")


if __name__ == "__main__":
    jsonl_path = "path/to/your/data.jsonl"  # Replace with your actual path
    main(jsonl_path)
