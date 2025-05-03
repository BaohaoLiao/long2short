import os
import json
from pathlib import Path
from typing import Iterable, Union, Any


PROMPT_TEMPLATES = {
    "deepseek-r1": (
        "<｜begin▁of▁sentence｜><｜User｜>Question: {input}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n"
    ),
    "deepseek-r1-choice": (   # For multiple choice question
        "<｜begin▁of▁sentence｜>"
        "<｜User｜>Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n{input}<｜Assistant｜><think>\n"
    ),
    "qwen3-think": (
        "<|im_start|>system<|im_end|>\n"
        "<|im_start|>user\nQuestion: {input}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    ),
    "qwen3-think-choice": (
        "<|im_start|>user\nAnswer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n"
        "{input}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    ),
}


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def load_data(data_name, data_dir="./datas"):
    data_file = f"{data_dir}/{data_name}/test.jsonl"
    assert os.path.exists(data_file)
    examples = list(load_jsonl(data_file))

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def construct_prompt(example, args):
    input_template = PROMPT_TEMPLATES[args.prompt_type]
    full_prompt = input_template.format(input=example["question"])
    return full_prompt