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

# generation per pairID
seen_pair_ids = set()
try:
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            seen_pair_ids.add(record["pairID"])
    print(f"Skipping {len(seen_pair_ids)} already processed pairIDs.")
except FileNotFoundError:
    print("Output file does not exist yet. All examples will be processed.")
    
# prompt for explanation generation
def build_prompt(premise, hypothesis, gold_label):
    return f"""You are an expert in Natural Language Inference (NLI). Please list all possible explanations for why the following statement is {gold_label} given the content below without introductory phrases.
    Content: {premise}
    Statement: {hypothesis}
    Answer:"""
    

# generation start
with open(args.input, "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f if line.strip()]

# filter unique pairID 
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
        
with open(args.output, "w", encoding="utf-8") as out_f:
    for item in tqdm(unique_data.values(), desc="Generating", total=len(data)):
        pid = item["pairID"]
        if pid in seen_pair_ids:
            continue
        
        try:
            premise = item.get("premise", "").strip()
            hypothesis = item.get("hypothesis", "").strip()
            gold_label = item.get("gold_label", "").strip()

            prompt = build_prompt(premise, hypothesis, gold_label)

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()

        except Exception as e:
            answer = f"Error: {str(e)}"

        out_f.write(json.dumps({"pairID": pid, "Answer": answer}, ensure_ascii=False) + "\n")
        out_f.flush()