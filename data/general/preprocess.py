import json
import random

from opencc import OpenCC

converter = OpenCC("s2twp")

with open(
    "./data/general/alpaca_translate_GPT_35_10_20k.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

data = [
    {
        "instruction": converter.convert(item["instruction"]),
        "input": converter.convert(item["input"]),
        "output": converter.convert(item["output"]),
    }
    for item in data
]
random.shuffle(data)

with open(
    "./data/general/alpaca_translate_GPT_35_10_20k.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
