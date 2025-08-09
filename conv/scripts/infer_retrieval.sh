# # # # infer
cd src

export CUDA_VISIBLE_DEVICES=1

accelerate launch infer_conv_retrieval.py \
    --dataset inspired \
    --split test \
    --tokenizer /root/autodl-tmp/DCRS-main/conv/src/models/DialoGPT-small \
    --model /root/autodl-tmp/DCRS-main/conv/src/models/DialoGPT-small \
    --text_tokenizer /root/autodl-tmp/DCRS-main/conv/src/models/roberta-base \
    --text_encoder /root/autodl-tmp/DCRS-main/conv/src/models/roberta-base \
    --n_prefix_conv 110 \
    --n_examples 3 \
    --mapping \
    --type_of_run Test \
    --check_point ./prompt-conv-retrieval-bart-inspired-42/final \
    --per_device_eval_batch_size 64 \
    --context_max_length 200 \
    --resp_max_length 80 \
    --prompt_max_length 50 \
    --entity_max_length 32