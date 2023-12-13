torchrun --standalone --nnodes=1 --nproc_per_node=4 finetune/lora.py \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --model_type llama \
    --data_dir data/general/alpaca_translate_GPT_35_10_20k.json \
    --output_dir finetuned/meta-llama/Llama-2-7b-chat-hf \
    --lora_target_modules '["q_proj", "v_proj"]' \
    --micro_batch_size 1