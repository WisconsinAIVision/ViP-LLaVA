#!/bin/bash
model_name=$1
model_name_replace=${model_name//\//_}
echo $model_name_replace
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$model_name_replace \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$model_name_replace.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$model_name_replace.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$model_name_replace.json

