model_name=mucai/vip-llava-7b-base-vcr-ft
eval_dataset=vcr-val 
python llava/eval/model_vqa_loader_vip.py  \
      --model-path   $model_name \
      --question-file ./dataset/$eval_dataset.json \
      --image-folder  ./dataset \
      --alpha 128 \
      --visual_prompt_style vcr_qa \
      --image_aspect_ratio resize \
      --answers-file ./playground/data/eval/$eval_dataset-qa-$model_name.json
