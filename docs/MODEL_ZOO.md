# Model Zoo

**To Use ViP-LLaVA checkpoints, your llava package version must be newer than 1.1.0.**

If you are interested in including any other details in Model Zoo, please open an issue :)

The model weights below are *merged* weights. You do not need to apply delta. The usage of ViP-LLaVA checkpoints should comply with the base LLM's model license: [Llama 2](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).

## ViP-LLaVA

| Version | Size | Schedule | Checkpoint | Visual7W | PointQA-LookTwice | RegionBench@Box | RegionBench@Human
|----------|----------|-----------|-----------|---|---|---|---|
| ViP-LLaVA | 7B | full_ft-1e | [mucai/vip-llava-7b](https://huggingface.co/mucai/vip-llava-7b) | 86.09 | 71.31 | 48.4 | 48.3 | 
| ViP-LLaVA | 13B | full_ft-1e | [mucai/vip-llava-13b](https://huggingface.co/mucai/vip-llava-13b) | 88.28 | 71.77 | 48.3 | 48.2 |

Base model: Vicuna v1.5.


## Projector weights

These are projector weights we have pretrained. You can use these projector weights for visual instruction tuning. They are just pretrained on image-text pairs and are NOT instruction-tuned, which means they do NOT follow instructions as well as our official models and can output repetitive, lengthy, and garbled outputs.


NOTE: When you use our pretrained projector for visual instruction tuning, it is very important to use the same base LLM and vision encoder as the one we used for pretraining the projector. Otherwise, the performance will be very poor.

When using these projector weights to instruction-tune your LMM, please make sure that these options are correctly set as follows,

```Shell
--mm_use_im_start_end False
--mm_use_im_patch_token False
```

Coming soon... 

## VCR checkpoint

Coming soon...


## RefCOCOg Region Captioning checkpoint

Coming soon...
