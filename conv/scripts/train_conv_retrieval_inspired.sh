# train
# cp -r data/redial src/data/
cd src
# python data/redial/process_mask.py

CUDA_VISIBLE_DEVICES=4 accelerate launch --gpu_ids 4 train_conv_retrieval_bart.py \
    --dataset inspired \
    --tokenizer microsoft/DialoGPT-small \
    --model microsoft/DialoGPT-small \
    --text_tokenizer roberta-base \
    --text_encoder roberta-base \
    --num_train_epochs 10 \
    --n_examples 5 \
    --mapping \
    --type_of_run random_retrieval \
    --gradient_accumulation_steps 1 \
    --ignore_pad_token_for_loss \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_warmup_steps 967 \
    --prompt_max_length 70 \
    --context_max_length 200 \
    --resp_max_length 80 \
    --learning_rate 1e-4 \
    --output_dir ./prompt-conv-retrieval-bart-inspired-42 \
    --seed 24