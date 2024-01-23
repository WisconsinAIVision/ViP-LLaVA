# Finetune ViP-LLaVA on Custom Datasets

## Dataset Format

Convert your data to a JSON file of a List of all samples. Sample metadata should contain `id` (a unique identifier), `image` (the path to the image), `conversations` (the conversation data between human and AI), `bboxes` (a list of bounding boxes), and optionally `segmentations` (a list of segmentation masks). Note that the image can be annotaed with arbitrary visual prompts. Also try using different visual prompts. 

A sample JSON for finetuning ViP-LLaVA for generating tag-style captions for Stable Diffusion:

```json
[
  {
    "id": "997bb945-628d-4724-b370-b84de974a19f",
    "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
    "bbox": [ [25, 30, 120, 150], [35, 23, 134, 213] ]
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image, foucsing on <bbox0> and <bbox1>."
      },
      {
        "from": "gpt",
        "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
      },
    ]
  },
  ...
]
```

## Command

If you have a limited task-specific data, we recommend finetuning from ViP-LLaVA checkpoints with LoRA or fully finetune by with a small learning rate like 2e-6 or 2e-7.

If the amount of the task-specific data is sufficient, you can also finetune from ViP-LLaVA checkpoints with full-model.

You may need to adjust the hyperparameters to fit each specific dataset and your hardware constraint.


