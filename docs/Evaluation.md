# Evaluation

# ViP-Bench

1. Extract contents of [`ViP-Bench`](https://huggingface.co/datasets/mucai/ViP-Bench) to `./playground/data/eval/ViP-Bench`.
2. Single-GPU inference and evaluate for bbox and human drawn visual prompts, respectively.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/vipbench.sh bbox
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/vipbench.sh human
```
Optionally, Change the model name from `vip-llava-7b` to other LLaVA or ViP-LLaVA models.

3. Submit the results to the [evaluation server](https://huggingface.co/spaces/mucai/ViP-Bench_Evaluator): `./playground/data/eval/ViP-Bench/results/vip-llava-7b-human.json`.


Optionally, see [here](https://github.com/mu-cai/ViP-LLaVA/blob/main/scripts/eval/vip-bench_evaluator.py), which is an evaluation script using your own openai key. 

## Source annotation

In `source_image`, we provide the source plain images along with the bounding box/mask annotations. Researchers can use such grounding information to match the special tokens such as `<obj>` in `"question"` entry of `vip-bench-meta-data.json`. For example, `<obj>` can be replaced by textual coordinates to evaluate the region-level multimodal models.  





# Academic Benchmarks

Please download the evaluation `json` dataset [here](https://huggingface.co/datasets/mucai/ViP-LLaVA-Instruct/tree/main). 

## Visusl7W

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/v7w.sh
```


## PointQA-LookTwice

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/pointQA.sh
```


## Visual Commonsense Reasoning

For Q -> A:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/vcr-qa.sh
```

For QA -> R:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/vcr-qar.sh
```





