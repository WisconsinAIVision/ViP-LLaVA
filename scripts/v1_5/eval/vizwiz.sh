#!/bin/bash

model_name=$1
model_name_replace=${model_name//\//_}
echo $model_name_replace

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$model_name_replace \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$model_name_replace.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$model_name_replace.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$model_name_replace.json
