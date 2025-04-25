import json
import argparse
from tqdm import tqdm
from openai import OpenAI
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def init_client(model_name, backend):
    if backend == "openai":
        if model_name == "gpt4o":
            return OpenAI(
                api_key="sk-bvHc2XGbyz9GGWkgTXKLbS6U9PoPrlv2Y1HIvlHLYev3vR3M",
                base_url="https://xiaoai.plus/v1"
            ), "gpt-4o"
        elif model_name == "deepseek-chat":
            return OpenAI(
                api_key="sk-fa448d9f39624998a86405f230a462ba",
                base_url="https://api.deepseek.com/v1"
            ), "deepseek-chat"
        else:
            raise ValueError("Invalid model. Choose from gpt4o or deepseek-chat")
    elif backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        return (tokenizer, model), model_name
    else:
        raise ValueError("Invalid backend. Choose from openai or hf")


def generate_with_hf(model_pack, prompt):
    tokenizer, model = model_pack
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()


def mark_tokens(text, highlight_indices_str):
    highlight_indices = sorted(set(int(i.strip()) for i in highlight_indices_str.split(",") if i.strip().isdigit()))
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    for i in highlight_indices:
        if 0 <= i < len(tokens):
            tokens[i] = f"**{tokens[i]}**"
    return " ".join(tokens)


def build_prompt(mode, premise, hypothesis, gold_label, highlighted_1="", highlighted_2=""):
    if mode == "highlight_index":
        return f"""You are an expert in Natural Language Inference (NLI). Your task is to generate possible explanations for why the following statement is **{gold_label}**, focusing on the highlighted parts of the sentences.\n\n    Content: {premise}\n    Highlighted word indices in Content: {highlighted_1}\n\n    Statement: {hypothesis}\n    Highlighted word indices in Statement: {highlighted_2}\n\n    Please list all possible explanations without introductory phrases.\n    Answer:"""
    elif mode == "highlight_marked":
        marked_premise = mark_tokens(premise, highlighted_1)
        marked_hypothesis = mark_tokens(hypothesis, highlighted_2)
        return f"""You are an expert in Natural Language Inference (NLI). Your task is to generate possible explanations for why the following statement is **{gold_label}**, focusing on the highlighted parts of the sentences. Highlighted parts are marked in \"**\".\n\n    Content: {marked_premise}\n    Statement: {marked_hypothesis}\n\n    Please list all possible explanations without introductory phrases.\n    Answer:"""
    elif mode == "label":
        return f"""You are an expert in Natural Language Inference (NLI). Please list all possible explanations for why the following statement is {gold_label} given the content below without introductory phrases.\n    Content: {premise}\n    Statement: {hypothesis}\n    Answer:"""
    elif mode == "taxonomy":
        return f"""You are an expert in Natural Language Inference (NLI). Given the following Premise, Hypothesis, and a gold label, your task is to generate explanations for **each** of the explanation categories listed below, assuming the same gold label holds. Each category reflects a specific type of inference in the explanation between the premise and hypothesis.
        The explanation categories are:
        1. Coreference Resolution – The explanation resolves coreferences (e.g., pronouns or demonstratives) across premise and hypothesis.
        2. Semantic-level Inference – Based on word meaning (e.g., synonyms, antonyms, negation).
        3. Syntactic-level Inference – Based on structural rephrasing that preserves meaning (e.g., alternation, coordination, subordination).
        4. Pragmatic-level Inference – Based on implicature, presupposition, or speaker intent.
        5. Absence of Mention – Hypothesis introduces plausible but unsupported or unmentioned information.
        6. Logical Structure Conflict – Identifies logical inconsistency (e.g., either-or, temporal, quantifier).
        7. Factual Knowledge – Explanation based on commonsense or factual knowledge, no further inference.
        8. World-Informed Logical Reasoning – Requires real-world causal or assumed reasoning beyond the text.
        Premise: {premise}\n Hypothesis: {hypothesis}\n Label: {gold_label}\n 
        Please list all possible explanations without introductory phrases. Format your answer as:
        1. Coreference Resolution:  
        - [Your explanation(s) here]  

        2. Semantic-level Inference:  
        - [Your explanation(s) here]  

        ... (continue for all 8 categories)
        Answer:"""
        
    elif mode == "classify_and_generate":
        return f"""You are an expert in Natural Language Inference (NLI). Your task is to examine the relationship between the following Premise and Hypothesis under the given gold label, and:
        Firstly, identify all categories for explanations from the list below (you may choose more than one) that could reasonably support the label.
        Secondly, for each selected category, generate all possible explanations that reflects that type.
        
        The explanation categories are:
        1. Coreference Resolution – The explanation resolves coreferences (e.g., pronouns or demonstratives) across premise and hypothesis.
        2. Semantic-level Inference – Based on word meaning (e.g., synonyms, antonyms, negation).
        3. Syntactic-level Inference – Based on structural rephrasing that preserves meaning (e.g., alternation, coordination, subordination).
        4. Pragmatic-level Inference – Based on implicature, presupposition, or speaker intent.
        5. Absence of Mention – Hypothesis introduces plausible but unsupported or unmentioned information.
        6. Logical Structure Conflict – Identifies logical inconsistency (e.g., either-or, temporal, quantifier).
        7. Factual Knowledge – Explanation based on commonsense or factual knowledge, no further inference.
        8. World-Informed Logical Reasoning – Requires real-world causal or assumed reasoning beyond the text.

        Premise: {premise}\n Hypothesis: {hypothesis}\n Label: {gold_label}\n 

        Please list all possible explanations without introductory phrases for all the chosen categories. Format your answer as:
        1. Coreference Resolution:  
        - [Your explanation(s) here]  

        2. Semantic-level Inference:  
        - [Your explanation(s) here]  

        ... (continue for all reasonable categories)
        Answer:"""
    
    elif mode == "classify_only":
        return f"""You are an expert in Natural Language Inference (NLI). Your task is to identify all applicable reasoning categories for explanations from the list below that could reasonably support the label. Multiple categories may apply. Only output the numbers of the applicable categories, separated by commas. Do not include any explanation.

        The explanation categories are:

        1. Coreference Resolution – The explanation resolves coreferences (e.g., pronouns or demonstratives) across premise and hypothesis.
        2. Semantic-level Inference – Based on word meaning (e.g., synonyms, antonyms, negation).
        3. Syntactic-level Inference – Based on structural rephrasing that preserves meaning (e.g., alternation, coordination, subordination).
        4. Pragmatic-level Inference – Based on implicature, presupposition, or speaker intent.
        5. Absence of Mention – Hypothesis introduces plausible but unsupported or unmentioned information.
        6. Logical Structure Conflict – Identifies logical inconsistency (e.g., either-or, temporal, quantifier).
        7. Factual Knowledge – Explanation based on commonsense or factual knowledge, no further inference.
        8. World-Informed Logical Reasoning – Requires real-world causal or assumed reasoning beyond the text.

        Premise: {premise}\n Hypothesis: {hypothesis}\n Label: {gold_label}\n 

        Answer (category numbers only, separated by commas):"""
    
    else:
        raise ValueError("Invalid mode. Choose from highlight_index, highlight_marked, label, taxonomy, classify_and_generate or classify_only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt4o", "deepseek-chat"], required=False, default="gpt4o")
    parser.add_argument("--hf_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--backend", type=str, choices=["openai", "hf"], default="openai")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="api_generated_output.jsonl")
    parser.add_argument("--mode", type=str, choices=["highlight_index", "highlight_marked", "label", "taxonomy" , "classify_and_generate", "classify_only"], required=True)
    args = parser.parse_args()

    client, model_name = init_client(args.hf_model if args.backend == "hf" else args.model, args.backend)

    with open(args.input, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    seen_pair_ids = set()
    try:
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                seen_pair_ids.add(record["pairID"])
    except FileNotFoundError:
        pass

    if args.mode in ["label", "taxonomy", "classify_and_generate", "classify_only"]:
        unique_data = {}
        for item in all_data:
            pid = item["pairID"]
            if pid not in unique_data:
                unique_data[pid] = {
                    "pairID": pid,
                    "premise": item["premise"].strip(),
                    "hypothesis": item["hypothesis"].strip(),
                    "gold_label": item["gold_label"].strip()
                }
        data_to_process = list(unique_data.values())
    else:
        data_to_process = all_data

    with open(args.output, "a", encoding="utf-8") as out_f:
        for item in tqdm(data_to_process, desc=f"Generating with mode: {args.mode}", total=len(data_to_process)):
            pid = item["pairID"]
            if pid in seen_pair_ids:
                continue
            try:
                prompt = build_prompt(
                    args.mode,
                    premise=item["premise"],
                    hypothesis=item["hypothesis"],
                    gold_label=item["gold_label"],
                    highlighted_1=item.get("Sentence1_Highlighted", ""),
                    highlighted_2=item.get("Sentence2_Highlighted", "")
                )

                if args.backend == "openai":
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    answer = response.choices[0].message.content.strip()
                elif args.backend == "hf":
                    answer = generate_with_hf(client, prompt)

            except Exception as e:
                answer = f"Error: {str(e)}"

            result = {
                "pairID": pid,
                "Sentence1_Highlighted": item.get("Sentence1_Highlighted", ""),
                "Sentence2_Highlighted": item.get("Sentence2_Highlighted", ""),
                "Answer": answer
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
