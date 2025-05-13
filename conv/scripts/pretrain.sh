cp -r data/redial src/data/
# cp -r data/inspired src/data/
cd src
# python data/inspired/process.py
python data/redial/process.py

export CUDA_VISIBLE_DEVICES=2

accelerate launch train_pre_retrieval.py \
    --dataset redial \
    --tokenizer /root/autodl-tmp/DCRS-main/conv/src/models/DialoGPT-small \
    --model /root/autodl-tmp/DCRS-main/conv/src/models/DialoGPT-small \
    --text_tokenizer /root/autodl-tmp/DCRS-main/conv/src/models/roberta-base \
    --text_encoder /root/autodl-tmp/DCRS-main/conv/src/models/roberta-base \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_warmup_steps 1389 \
    --max_length 200 \
    --prompt_max_length 2 \
    --entity_max_length 32 \
    --learning_rate 5e-4 \
    --output_dir ./pre-trained-prompt-retrieval \
    --seed 18