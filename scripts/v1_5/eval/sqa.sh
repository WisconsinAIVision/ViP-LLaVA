#!/bin/bash
model_name=$1
model_name_replace=${model_name//\//_}
echo $model_name_replace

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/$model_name \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$model_name_replace.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$model_name_replace.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$model_name_replace_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$model_name_replace_result.json
