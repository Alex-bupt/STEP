# cp -r data/inspired src/data/
# cd src/data/inspired
# python process.py
# cd ../../

# cp -r data/redial src/data/
cd src
# python data/redial /process.py

CUDA_VISIBLE_DEVICES=0 accelerate launch train_pre.py \
    --dataset Redial_c2_copy_new \
    --tokenizer /root/autodl-tmp/STEP-main/rec/src/models/DialoGPT-small \
    --model /root/autodl-tmp/STEP-main/rec/src/models/DialoGPT-small \
    --text_tokenizer /root/autodl-tmp/STEP-main/rec/src/models/roberta-base \
    --text_encoder /root/autodl-tmp/STEP-main/rec/src/models/roberta-base \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 54 \
    --per_device_eval_batch_size 128 \
    --num_warmup_steps 1389  \
    --max_length 200 \
    --prompt_max_length 200 \
    --entity_max_length 32 \
    --learning_rate 5e-4 \
    --output_dir ./pre-trained-prompt-vricr \
    --seed 22 