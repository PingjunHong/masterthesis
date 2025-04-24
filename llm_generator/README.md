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

  - You can now specify your HF model with `--hf_model` (default: `meta-llama/Llama-3.2-3B-Instruct`).

## Usage

```bash
python model_generator.py \
  --backend openai \
  --model gpt4o \
  --mode highlight_index \
  --input ./data.jsonl \
  --output ./gpt4o_results.jsonl
```

```bash
python model_generator.py \
  --backend hf \
  --hf_model meta-llama/Llama-3.2-3B-Instruct \
  --mode label \
  --input ./data.jsonl \
  --output ./llama_output.jsonl
```