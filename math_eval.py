import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

from utils.data import load_data, construct_prompt
from utils.parser import parse_question, parse_ground_truth, extract_and_verify_pred
from utils.eval import majority_voting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math", type=str)
    parser.add_argument("--data_dir", default="./datas", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="deepseek-r1", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--min_p", default=0.05, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--enable_prefix_caching', action='store_true', default=False)
    parser.add_argument('--disable_chunked_prefill', action='store_true', default=False)
    parser.add_argument('--max_model_len', type=int, default=64000)
    parser.add_argument("--n_sampling", type=int, default=1)

    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")


def prepare_data(data_name, args):
    if "math500_level" in data_name:
        level = int(data_name.strip()[-1])
        examples = load_data("math500", args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(data_name, args.data_dir)

    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # get out_file name
    out_file_prefix = f"{args.prompt_type}_seed{args.seed}_t{args.temperature}topp{args.top_p}minp{args.min_p}topk{args.top_k}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_num{args.num_test_sample}_n{args.n_sampling}.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    # infer & eval
    data_list = args.data_names.split(",")
    for data_name in data_list:
        main(llm, data_name, args)


def main(llm, data_name, args):
    examples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        idx = int(example["idx"])

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        full_prompt = construct_prompt(example, args)

        if i == 0:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": example["answer"],
            "answer": example["answer"],
            "prompt": full_prompt,
        }
        samples.append(sample)

    # start inference
    prompts = [sample["prompt"] for sample in samples]
    start_time = time.time()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens_per_call,
            n=args.n_sampling,
            skip_special_tokens=False,
            seed=args.seed,
        ),
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    assert len(outputs) == len(prompts)
    end_time = time.time()

    # Extract pred and eval
    results = []
    avg_acc = []
    avg_maj_acc = []
    for sample, output in zip(samples, outputs):
        gt = parse_ground_truth(sample, data_name)

        preds = []
        scores = []
        for o in output.outputs:
            # Avoid the bug in math_verify for multiple boxeds
            if "</think>" in o.text:
                model_output = o.text.split("</think>")[-1]
            else:
                model_output = o.text
            pred, score = extract_and_verify_pred(model_output, gt, data_name)
            preds.append(pred)
            scores.append(score)
        avg_acc.append(np.mean(scores))

        maj_pred, maj_score = majority_voting(preds, scores)
        avg_maj_acc.append(maj_score)

        results.append(
            {
                "idx": sample["idx"],
                "question": sample["question"],
                "gt": str(sample["answer"]),
                "pred": preds,
                "score": scores,
                "maj_pred": maj_pred,
                "maj_score": maj_score,
                "model_output": [o.text for o in output.outputs],
            }
        )

    time_use = (end_time - start_time) / 60
    result_json = {
        "num_samples": len(samples),
        "pass@1": np.mean(avg_acc),
        f"maj@{args.n_sampling}": np.mean(avg_maj_acc),
        "time_use_in_min": time_use,
    }
    print(result_json)

    print(f"Saving model outputs for {data_name} to {out_file}")
    json.dump(results, open(out_file, "w",), indent=4)

    with open(out_file.replace(".json", f"_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)