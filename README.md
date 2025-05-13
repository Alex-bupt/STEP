# STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation

This repo contains code and data for the paper: "STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation"

## 1. Setup

Please install the libraries, and packages listed in the conv/requirements.txt file.

## 2. Data Preprocessing

We use the dataset from the DCRS repository. You can download the processed data (including conv, rec, and retrieval data) from the following <a href = 'https://drive.google.com/drive/folders/1kEOn-lDQ9L5NgBhohg4Upwo9Kr4T01a6?usp=share_link'>link</a>.

## 3. Training Response Generation Model

For the ReDial dataset, to train our response generation model, please run:

```shell
cd conv
sh scripts/train_conv_retrieval_redial.sh
```

To produce generated responses, please run the following command:

```shell
sh scripts/infer_retrieval.sh
```

## 4. Training Recommendation Engine

To train our recommendation engine, you need to first pre-train neural embeddings of demonstrations.

```shell
cd rec
sh scripts/pretrain.sh
```

To finetune our recommendation engine, please following commands:

```shell
sh scripts/train_rec.sh
```

## Acknowledgement

We thank <a href='https://dl.acm.org/doi/10.1145/3626772.3657755'>DCRS</a> and <a href = 'https://github.com/wxl1999/UniCRS/tree/main'>UNICRS </a> for providing the useful source code for the data preprocessing and prompt learning steps.


