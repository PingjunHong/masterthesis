# LLM-based Explanation Generator

This folder contains scripts for generating Natural Language Inference (NLI) explanations using **OpenAI API**, **DeepSeek**, or **local HuggingFace models**.

## Available Modes (`--mode`)

1. `label`
Generate explanations without highlight or taxonomy guidance, based on: `premise` + `hypothesis` + `gold_label`

2. `highlight_index`
Use **word index** annotations (`Sentence1_Highlighted`, `Sentence2_Highlighted`) to generate focused explanations.

3. `highlight_marked`
Highlights are **rendered in-line** using `**XXX**` markers for token emphasis.

## Backends (--backend)

- `openai` *(default)*: Uses either:

  - `gpt4o` via OpenAI API

  - `deepseek-chat` via DeepSeek API

- `hf`: Local generation with Huggingface model

  - Default: `meta-llama/Llama-3.2-3B-Instruct`

## Usage

```bash
python model_generator.py \\
  --mode highlight_index \\
  --backend openai \\
  --model gpt4o \\
  --input ./your_input.jsonl \\
  --output ./your_output.jsonl
```

```
python model_generator.py \\
  --mode label \\
  --backend hf \\
  --input ./your_input.jsonl \\
  --output ./output_llama.jsonl
```