#!/bin/bash
model_name=$1
model_name_replace=${model_name//\//_}
echo $model_name_replace

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$model_name_replace \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$model_name_replace.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$model_name_replace.jsonl
