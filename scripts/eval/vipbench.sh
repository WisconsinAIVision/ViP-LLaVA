model_name=vip-llava-7b
model_path=mucai/$model_name
folder=ViP-Bench
split=$1
mkdir -p ./playground/data/eval/$folder/answers
python -m llava.eval.model_vqa \
    --model-path $model_path \
    --question-file ./playground/data/eval/$folder/$split/questions.jsonl \
    --image-folder ./playground/data/eval/$folder/$split/images \
    --answers-file ./playground/data/eval/$folder/answers/$model_name-$split.jsonl \
    --temperature 0 

mkdir -p ./playground/data/eval/$folder/results

python scripts/convert_vipbench_for_eval.py \
    --src ./playground/data/eval/$folder/answers/$model_name-$split.jsonl \
    --dst ./playground/data/eval/$folder/results/$model_name-$split.json