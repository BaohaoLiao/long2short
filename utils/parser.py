from typing import Any, Dict
from math_verify import parse, verify, StringExtractionConfig, LatexExtractionConfig


def parse_question(example, data_name):
    question = ""
    if data_name in ["mmlu_stem", "gpqa"]:
        options = example["choices"]
        assert len(options) == 4
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"{label}. {str(option).strip()}\n"
        options = " ".join(options).strip()
        question = f"{example['question'].strip()}\n\n {options}"
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    return question.strip()


def parse_ground_truth(example: Dict[str, Any], data_name="math500"):
    if data_name in [
        "math", "math500", "math500_level1", "math500_level2", 
        "math500_level3", "math500_level4", "math500_level5",
        "aime24", "aime25", "aimo2", "hmmt"
    ]:
        answer = "$" + example["answer"] + "$"
    elif data_name in ["mmlu_stem", "gpqa"]:
        abcd = "ABCD"
        answer = "$" + abcd[example["answer"]] + "$"
    return parse(answer)


def extract_pred_and_parse(completion, data_name):
    if data_name in ["gpqa"]:
        pred = parse(
            completion,
            extraction_config=[StringExtractionConfig(lowercase=False)],
        )
    elif "boxed" in completion:
        pred = parse(
            completion, 
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, 
                    try_extract_without_anchor=True,
                ),
            ]
        )
    else:
        pred = []
    return pred


def extract_and_verify_pred(completion, gt, data_name="math500"):
    pred = extract_pred_and_parse(completion, data_name)
    score = verify(gt, pred)
    if pred:
        return str(pred[0]), score
    else:
        return None, score