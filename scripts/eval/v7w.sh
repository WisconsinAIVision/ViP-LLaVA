
model_name=mucai/vip-llava-13b
eval_dataset=v7w-test
python llava/eval/model_vqa_loader_vip.py  \
      --model-path  $model_name  \
      --question-file ./dataset/$eval_dataset.json \
      --image-folder  ./dataset \
      --alpha 128 \
      --answers-file ./playground/data/eval/$eval_dataset-$model_name-alpha128.json
