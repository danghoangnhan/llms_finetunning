import ast
import os

import click


def check_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = local_rank = world_size = -1
    return rank, local_rank, world_size


def tokenize(tokenizer, prompt, cutoff_len, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_sample(tokenizer, item, max_seq_length, add_eos_token=True):
    assert tokenizer is not None
    tokenizer.pad_token_id = 0

    result = tokenizer(
        item["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    result = {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_seq_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    return result


def generate_prompt(data_point):
    if data_point["input"]:
        return ("Dưới đây là một chỉ thị mô tả nhiệm vụ, cùng với thông tin đầu vào liên quan đến nhiệm vụ. Hãy soạn một câu trả lời phù hợp để hoàn thành chỉ thị này\n\n"
        f'### Chỉ thị:\n{data_point["instruction"]}\n\n### Đầu vào:\n{data_point["input"]}\n\n'
        f'### Trả lời:\n{data_point["output"]}')
    else:
        return ("Dưới đây là một chỉ thị mô tả nhiệm vụ. Hãy soạn một câu trả lời phù hợp để hoàn thành chỉ thị này\n\n"
        f'### Chỉ thị:\n{data_point["instruction"]}\n\n### Trả lời:\n{data_point["output"]}')


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)