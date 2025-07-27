# Explanation Similarity Scorer

This script computes similarity metrics among NLI explanations for each `(pairID, explanation_category, gold_label)` group using embedding-based and ngram-based methods.

It supports command-line input of `.jsonl` files and outputs the per-instance similarity results in JSONL format.

## Input Format
The input should be a `.jsonl` file (one JSON object per line) with the following keys:
```
{
  "pairID": "abc123",
  "premise": "...",
  "hypothesis": "...",
  "gold_label": "entailment",
  "explanation": "The man is in a black shirt refers to the man in the black t-shirt.",
  "explanation_category": [1, 2]
}
```

## Output
The script saves a `.jsonl` file where each line contains:
```
{
  "pair_id": "abc123",
  "category": 1,
  "gold_label": "entailment",
  "num_explanations": 3,
  "cosine": 0.8321,
  "euclidean": 0.5523,
  "unigram_overlap": 0.667,
  "bigram_overlap": 0.489,
  "pos_unigram_overlap": 0.756,
  "pos_bigram_overlap": 0.423
}
```

## Supported Metrics
- Cosine similarity (sentence embeddings)
- Euclidean similarity
- Unigram overlap (token-level)
- Bigram overlap
- POS-based unigram/bigram overlap

## Usage

```bash
python similarity_scorer.py \
  --input ./your_explanations.jsonl \
  --output ./similarity_results.jsonl
```