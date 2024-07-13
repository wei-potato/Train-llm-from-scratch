output_model=save_folder
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
cp ./finetune.sh ${output_model}
deepspeed  finetune_clm.py \
    --model_name_or_path /pretrained_model \
    --train_files /train_llm/data/sft_data/alpaca_output.jsonl \
    --validation_files  /train_llm/data/dev_sft.jsonl \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --warmup_steps 5 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log

    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \