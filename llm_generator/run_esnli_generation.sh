#!/bin/bash

INPUT_PATH="input_path"
OUTPUT_PATH="output_path"

MODE="label"
BACKEND="hf"
HF_MODEL="meta-llama/Llama-3.2-3B-Instruct"

python model_generator.py \
  --backend $BACKEND \
  --hf_model "$HF_MODEL" \
  --mode $MODE \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH"
