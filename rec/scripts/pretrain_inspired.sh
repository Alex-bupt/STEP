

CUDA_VISIBLE_DEVICES=0 accelerate launch train_pre_inspired.py \
    --dataset inspired_rec \
    --tokenizer /root/autodl-tmp/DCRS-main/rec/src/models/DialoGPT-small \
    --model /root/autodl-tmp/DCRS-main/rec/src/models/DialoGPT-small \
    --text_tokenizer /root/autodl-tmp/DCRS-main/rec/src/models/roberta-base \
    --text_encoder /root/autodl-tmp/DCRS-main/rec/src/models/roberta-base \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_warmup_steps 168  \
    --max_length 200 \
    --prompt_max_length 200 \
    --entity_max_length 32 \
    --learning_rate 1e-4 \
    --output_dir ./pre-trained-prompt-inspired \
    --seed 64