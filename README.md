# STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation

This repo contains code and data for the paper: "STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation" - <a href='https://dl.acm.org/doi/'>STEP</a>

## 1. Setup

Please install the libraries, and packages listed in the requirements.txt file. Make sure that you are using CUDA 11.8, Python 3.8, Pytorch 2.0.0.
Download the DialoGPT-small & Roberta-base and put them in rec/src/models, the same as dataset in rec/src/data.

## 2. Data Preprocessing

You can download the processed ReDial dataset from this <a href = 'https://drive.google.com/drive/folders/1kEOn-lDQ9L5NgBhohg4Upwo9Kr4T01a6?usp=share_link'>Google Drive link</a>, as provided in the <a href = 'https://github.com/huyquangdao/DCRS'>DCRS</a> repository. If you are unsure which specific files to download, please contact the author email address for assistance.

For the INSPIRED dataset, we recommend that you use natively INSPIRED datasets to ensure training quality, as the data processed by DCRS is insufficient to train F-Former. You can use the native INSPIRED dataset we provide.

For a fair comparison, we adopted the code from <a href = 'https://github.com/wxl1999/UniCRS/tree/main'>UNICRS</a> to process data for the recommendation engine and dialogue module respectively. 

## 3. Key Files

If you want to use the F-Former framework, its code in Qformer directory "rec/src/Qformer/models/blip2_models/fusion_qformer.py"
We have rewritten the Bert architecture of Qformer.py and ported it to Roberta. Therefore, when using it, please combine the code of the entire Qformer directory with F-former.

## 3. Training Recommendation Engine

To train our recommendation engine, you need to first pre-train neural embeddings of demonstrations. 

```
cd rec
sh scripts/pretrain.sh
```

To finetune our recommendation engine, please following commands:

```
sh scripts/train_rec.sh
```

## 4. Training Response Generation Model

For the ReDial dataset, to train our response generation model, please run:

```
cd conv
sh scripts/train_conv_retrieval_redial.sh
```

To produce generated responses, please run the following command:

```
sh scripts/infer_retrieval.sh
```


## Acknowledgement
We thank <a href='https://github.com/zxd-octopus/VRICR/tree/master'>DCRS</a> and <a href = 'https://github.com/wxl1999/UniCRS/tree/main'>UNICRS </a> for providing the useful source code for the data preprocessing and prompt learning steps.

Please cite the following our paper as references if you use our codes.

```bibtex

```
