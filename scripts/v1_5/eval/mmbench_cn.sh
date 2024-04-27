#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"
model_name=$1
model_name_replace=${model_name//\//_}
echo $model_name_replace

python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/$model_name_replace \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/$model_name_replace.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment $model_name_replace
