#!/bin/bash

MODEL=$1
MODE=$2
INPUT=$3
OUTPUT=$4

python model_generator.py --model $MODEL --mode $MODE --input $INPUT --output $OUTPUT
