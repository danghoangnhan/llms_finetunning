python inference/run_exp.py \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --model_type llama \
    --data_dir data/general/alpaca_translate_GPT_35_10_20k.json \
    --output_dir finetuned/meta-llama/Llama-2-7b-chat-hf \
    --log_dir './inference/local.log'