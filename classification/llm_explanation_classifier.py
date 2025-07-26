import os
import json
import argparse
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import classification_report

# label mapping
categories = [
    "Coreference",
    "Semantic",
    "Syntactic",
    "Pragmatic",
    "Absence of Mention",
    "Logic Conflict",
    "Factual Knowledge",
    "Inferential Knowledge"
]

def category_to_index(category):
    return str(categories.index(category) + 1)

def index_to_category(index):
    if index.isdigit() and 1 <= int(index) <= len(categories):
        return categories[int(index) - 1]
    return "Invalid"

# build prompt
def build_prompt(premise, hypothesis, label, explanation, few_shot_examples=None):
    instruction = (
        "You are a reasoning expert. Classify the explanation into one of the following reasoning categories.\n\n"
        "Here are the reasoning types:\n"
        "1. Coreference\n"
        "2. Semantic\n"
        "3. Syntactic\n"
        "4. Pragmatic\n"
        "5. Absence of Mention\n"
        "6. Logic Conflict\n"
        "7. Factual Knowledge\n"
        "8. Inferential Knowledge\n\n"
        "Respond only with the category index (1–8). No explanation, no label name.\n"
    )
    
    # instruction = """
    # You are a reasoning expert. Your task is to classify the following explanation into one of the reasoning categories listed below. 
    # Each category reflects a specific type of inference between the premise and hypothesis.

    # Here are the categories:

    # 1. Coreference
    #     - The explanation resolves references (e.g., pronouns or demonstratives) across premise and hypothesis.

    # 2. Semantic
    #     - Based on word meaning (e.g., synonyms, antonyms, negation).

    # 3. Syntactic
    #     - Based on structural rephrasing with the same meaning.

    # 4. Pragmatic
    #     - Based on implicature, presupposition, or speaker intent.

    # 5. Absence of Mention
    #     - The hypothesis contains new info not present in the premise.

    # 6. Logic Conflict
    #     - Structural logical exclusivity(e.g., either-or, at most), quantifier conflict.

    # 7. Factual Knowledge
    #     - Relies on commonsense, background, or domain-specific facts.

    # 8. Inferential Knowledge
    #     - Requires real-world causal or probabilistic reasoning.


    # Respond **only with the number (1–8)** corresponding to the most appropriate category.

    # """
    
    if few_shot_examples:
        examples = "\n".join([
            f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nLabel: {ex['gold_label']}\nExplanation: {ex['explanation']}\nAnswer: {category_to_index(ex['explanation_category'])}"
            for ex in few_shot_examples
        ])
        instruction += "\nExamples:\n" + examples + "\n\n"

    instruction += f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}\nExplanation: {explanation}\nAnswer:"
    return instruction

# few-shot examples
few_shot = [
    {
        "premise": "The man in the black t-shirt is trying to throw something.",
        "hypothesis": "The man is in a black shirt.",
        "gold_label": "entailment",
        "explanation": "The man is in a black shirt refers to the man in the black t-shirt.",
        "explanation_category": "Coreference"
    },
    {
        "premise": "A man in a black tank top is wearing a red plaid hat.",
        "hypothesis": "A man in a hat.",
        "gold_label": "entailment",
        "explanation": "A red plaid hat is a specific type of hat.",
        "explanation_category": "Semantic"
    },
    {
        "premise": "Two women walk down a sidewalk along a busy street in a downtown area.",
        "hypothesis": "The women were walking downtown.",
        "gold_label": "entailment",
        "explanation": "The women were walking downtown is a rephrase of, Two women walk down a sidewalk along a busy street in a downtown area.",
        "explanation_category": "Syntactic"
    },
    {
        "premise": "A girl in a blue dress takes off her shoes and eats blue cotton candy.",
        "hypothesis": "The girl is eating while barefoot.",
        "gold_label": "entailment",
        "explanation": "If a girl takes off her shoes, then she becomes barefoot, and if she eats blue candy, then she is eating.",
        "explanation_category": "Pragmatic"
    },
    {
        "premise": "A person with a purple shirt is painting an image of a woman on a white wall.",
        "hypothesis": "A woman paints a portrait of a person.",
        "gold_label": "neutral",
        "explanation": "A person with a purple shirt could be either a man or a woman. We can't assume the gender of the painter.",
        "explanation_category": "Absence of Mention"
    },
    {
        "premise": "Five girls and two guys are crossing an overpass.",
        "hypothesis": "The three men sit and talk about their lives.",
        "gold_label": "contradiction",
        "explanation": "Three is not two.",
        "explanation_category": "Logic Conflict"
    },
    {
        "premise": "Two people crossing by each other while kite surfing.",
        "hypothesis": "The people are both males.",
        "gold_label": "neutral",
        "explanation": "Not all people are males.",
        "explanation_category": "Factual Knowledge"
    },
    {
        "premise": "A girl in a blue dress takes off her shoes and eats blue cotton candy.",
        "hypothesis": "The girl in a blue dress is a flower girl at a wedding.",
        "gold_label": "neutral",
        "explanation": "A girl in a blue dress doesn’t imply the girl is a flower girl at a wedding.",
        "explanation_category": "Inferential Knowledge"
    }
]

def main():
    # load data
    with open(args.data, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # model choice
    if args.model == "hf":
        print(f"Loading model from huggingface: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20, do_sample=False)

    elif args.model == "gpt":
        print("Using OPEN AI API")
        client = OpenAI(api_key = "YOUR_API_KEY")
    else:
        raise ValueError("Unsupported model type")


    # classification
    true_labels, pred_labels, logs = [], [], []
    invalid_count = 0

    for i, row in tqdm(enumerate(data), total=len(data)):
        if "explanation_category" not in row:
            continue

        prompt = build_prompt(
            row["premise"], row["hypothesis"], row["gold_label"], row["explanation"],
            few_shot_examples=few_shot if args.fewshot else None
        )

        try:
            if args.model == "hf":
                output = generator(prompt, do_sample=False)[0]["generated_text"]
                raw_answer = output.split("Answer:")[-1].strip()
            else:
                model_name = args.openai_model
                response = client.chat.completions.create(
                    model = model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful and precise reasoning classifier."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                raw_answer = response.choices[0].message.content.strip()

            import re
            match = re.search(r"\b([1-8])\b", raw_answer)
            pred_index = match.group(1) if match else "Invalid"
            
            pred_category = index_to_category(pred_index)
            if pred_category == "Invalid":
                invalid_count += 1

        except Exception as e:
            print(f"Error at {i}: {e}")
            pred_index = "Invalid"
            pred_category = "Invalid"
            raw_answer = "Error"
            invalid_count += 1

        true_labels.append(row["explanation_category"])
        pred_labels.append(pred_category)

        logs.append({
            "pairID": row.get("pairID"),
            "true_label": row["explanation_category"],
            "predicted_index": pred_index,
            "predicted": pred_category,
            "raw_output": raw_answer
        })
            
    # classification report
    print(f"\nTotal predictions: {len(pred_labels)}")
    print(f"Invalid predictions: {invalid_count}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, labels=categories, zero_division=0))

    # save results
    output_file = f"predictions_{args.model}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
            
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hf", choices=["hf", "gpt"], help="Model source: hf or gpt")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="HuggingFace model name")
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo", help="Model name, e,g, gpt-4o")
    parser.add_argument("--fewshot", action="store_true", help="Use few-shot prompting")
    parser.add_argument("--data", type=str, required=True, help="Path to input JSONL file")
    args = parser.parse_args()
    
    main()