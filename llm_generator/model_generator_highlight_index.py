import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["gpt4o", "deepseek-chat"], required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, default="api_generated_output.jsonl")
args = parser.parse_args()

# API key settings
if args.model == "gpt4o":
    client = OpenAI(
        api_key="sk-bvHc2XGbyz9GGWkgTXKLbS6U9PoPrlv2Y1HIvlHLYev3vR3M", 
        base_url="https://xiaoai.plus/v1"
    )
    model_name = "gpt-4o"
elif args.model == "deepseek-chat":
    client = OpenAI(
        api_key="sk-fa448d9f39624998a86405f230a462ba", 
        base_url="https://api.deepseek.com/v1"
    )
    model_name = "deepseek-chat"
else:
    raise ValueError("Invalid model. Choose from gpt4o or deepseek")

# read input data
with open(args.input, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]
results = []

    
# prompt for explanation generation
def build_highlight_prompt(premise, hypothesis, highlighted_1, highlighted_2, gold_label):
    return f"""You are an expert in Natural Language Inference (NLI). Your task is to generate possible explanations for why the following statement is **{gold_label}**, focusing on the highlighted parts of the sentences.

    Content: {premise}
    Highlighted word indices in Content: {highlighted_1}

    Statement: {hypothesis}
    Highlighted word indices in Statement: {highlighted_2}

    Please list all possible explanations without introductory phrases.
    Answer:"""
    

# generation start
with open(args.input, "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f if line.strip()]

with open(args.output, "w", encoding="utf-8") as out_f:
    for item in tqdm(all_data, desc="Generating highlight-guided explanations", total=len(all_data)):
        try:
            prompt = build_highlight_prompt(
                premise=item["premise"],
                hypothesis=item["hypothesis"],
                highlighted_1=item.get("Sentence1_Highlighted", "{}"),
                highlighted_2=item.get("Sentence2_Highlighted", "{}"),
                gold_label=item["gold_label"]
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            answer = response.choices[0].message.content.strip()

        except Exception as e:
            answer = f"Error: {str(e)}"

        out_f.write(json.dumps({
            "pairID": item["pairID"],
            "Sentence1_Highlighted": item["Sentence1_Highlighted"],
            "Sentence2_Highlighted": item["Sentence2_Highlighted"],
            "Answer": answer
        }, ensure_ascii=False) + "\n")
        out_f.flush()