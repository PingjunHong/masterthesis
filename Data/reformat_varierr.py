import json
import csv 
import argparse
from pathlib import Path

def preprocess_varierr(input_file, output_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    with open(output_file, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["pairID", "premise", "hypothesis", "gold_label", "explanation"])
        writer.writeheader()

        for item in data:
            pair_id = item.get("id")
            premise = item.get("context")
            hypothesis = item.get("statement")
            labels = item.get("label_set_round_2", [])

            for label in labels:
                explanations = item.get(label, [])
                for e in explanations:
                    explanation_text = e.get("reason", "")
                    writer.writerow({
                        "pairID": pair_id,
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "gold_label": label,
                        "explanation": explanation_text
                    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess VariErr JSONL dataset into CSV")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
    else:
        print(f"Processing VariErr file: {input_path}")
        preprocess_varierr(input_path, output_path)
        print(f"Output saved to: {output_path}")