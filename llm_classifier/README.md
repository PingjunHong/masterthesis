# LLM-Based Explanation Category Classifier

`unified_explanation_classifier.py` provides a unified pipeline to classify NLI explanations into the standardized linguistic-based categories using LLMs.\\

It supports both **HuggingFace models (e.g., LLaMA, Mistral)** and **OpenAI GPT-3.5 / GPT-4** for **zero-shot** and **few-shot** classification.\\

## Task

Given the following input fields:

- `premise`
- `hypothesis`
- `label` (entailment, contradiction, neutral)
- `explanation`

The model classifies the explanation into one of the following 8 reasoning categories:

1. **Coreference Resolution**
2. **Semantic-level Inference**
3. **Syntactic-level Inference**
4. **Pragmatic-level Inference**
5. **Absence of Mention**
6. **Logical Structure Conflict**
7. **Factual Knowledge**
8. **World-Informed Logical Reasoning**

## Input Format

Input should be a `.jsonl` file while each line ia a JSON object with the following fields:

```bash
{
  "pairID": "12345",
  "premise": "A man is riding a bike.",
  "hypothesis": "A person is on a bicycle.",
  "gold_label": "entailment",
  "explanation": "A man is a person and riding a bike implies being on a bicycle.",
  "explanation_category": "Semantic-level Inference"
}
```

## Usage