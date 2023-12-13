import logging
from model import ModelServe
import json
import click

def init_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_dir, "w", "utf-8")
    handler.setFormatter(logging.Formatter('%(name)s %(message)s'))
    logger.addHandler(handler)
    return logger



@click.command()
@click.option("--base_model","base_model",type=str,default="meta-llama/Llama-2-7b-chat-hf",)
@click.option("--model_type", "model_type", type=str, default="llama")
@click.option("--data_dir","data_dir",type=str,default="data/general/alpaca_translate_GPT_35_10_20k.json")
@click.option("--output_dir","output_dir",type=str,default="finetuned/meta-llama/Llama-2-7b-chat-hf",)
@click.option("--log_dir","log_dir",type=str,default="./inference/local.log")

def main(
    base_model: str,
    model_type: str,
    data_dir: str,
    output_dir: str,
    log_dir: str,
):  
    print(
        f"Inference parameters: \n"
        f"base_model: {base_model}\n"
        f"model_type: {model_type}\n"
        f"data_dir: {data_dir}\n"
        f"output_dir: {output_dir}\n"
        f"log_dir: {log_dir}\n"
    )

    with open(data_dir, 'r') as file:
        prompts = json.load(file)
    logger = init_logger(log_dir=log_dir)
    model = ModelServe(load_8bit=True,base_model=base_model,model_type=model_type,finetuned_weights=output_dir)
    for idx, sample in enumerate(prompts):
        logger.info(f'===== {idx} ===== \n> Instruction: \n{sample["instruction"]}\n\n> Input: \n{sample["input"]}\n\n')
        output = model.generate(instruction=sample["instruction"], input=sample["input"])
        if "### Trả lời:" in output:
            output = output.split("### Trả lời:")[1]
        logger.info(f"> Output: \n{output}\n\n")

if __name__ == "__main__":
    main()
