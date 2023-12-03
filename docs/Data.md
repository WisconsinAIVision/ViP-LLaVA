## Data


### Pretraining Dataset
The pretraining dataset used in this release is a subset of CC-3M dataset, filtered with a more balanced concept coverage distribution.  Please see [here](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) for a detailed description of the dataset structure and how to download the images.

If you already have CC-3M dataset on your disk, the image names follow this format: `GCC_train_000000000.jpg`.  You may edit the `image` field correspondingly if necessary.

| Data | Chat File | Meta Data | Size |
| --- |  --- |  --- | ---: |
| CC-3M Concept-balanced 595K | [chat.json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json) | [metadata.json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/metadata.json) | 211 MB
| LAION/CC/SBU BLIP-Caption Concept-balanced 558K | [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json) | [metadata.json](#) | 181 MB

**Important notice**: Upon the request from the community, as ~15% images of the original CC-3M dataset are no longer accessible, we upload [`images.zip`](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip) for better reproducing our work in research community. It must not be used for any other purposes. The use of these images must comply with the CC-3M license. This may be taken down at any time when requested by the original CC-3M dataset owner or owners of the referenced images.