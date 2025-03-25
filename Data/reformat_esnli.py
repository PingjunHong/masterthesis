import pandas as pd
import argparse
from pathlib import Path

def reformat_esnli(input_path, output_path):
    df = pd.read_csv(input_path)

    records = []

    for _, row in df.iterrows():
        for i in range(1, 3 + 1):  # Explanation_1 to Explanation_3
            explanation = row.get(f"Explanation_{i}")
            if pd.notna(explanation) and str(explanation).strip() != "":
                records.append({
                    "pairID": row["pairID"],
                    "premise": row["Sentence1"],
                    "hypothesis": row["Sentence2"],
                    "gold_label": row["gold_label"],
                    "explanation": explanation.strip()
                })

    df_flat = pd.DataFrame(records)
    df_flat.to_csv(output_path, index=False)
    print(f"Saved expanded explanations to: {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat e-SNLI CSV by flattening explanations.")
    parser.add_argument("--input", "-i", required=True, help="Path to input eSNLI CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to output flattened CSV file")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
    else:
        reformat_esnli(input_path, output_path)