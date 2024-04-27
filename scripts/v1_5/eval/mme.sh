#!/bin/bash

model_name=$1
model_name_replace=${model_name//\//_}
echo $model_name_replace


python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$model_name \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$model_name_replace.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $model_name_replace

cd eval_tool

python calculation.py --results_dir answers/$model_name_replace
