#!/bin/bash

model_name=$1
model_name_replace=${model_name//\//_}
echo $model_name_replace

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$model_name \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$model_name_replace.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$model_name_replace.jsonl
