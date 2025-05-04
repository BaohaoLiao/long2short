import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

from utils.data import load_jsonl, construct_prompt
from utils.parser import parse_ground_truth, extract_and_verify_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="deepseek-r1", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0.6, type=int)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--min_p", default=0.0, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--enable_prefix_caching', action='store_true', default=False)
    parser.add_argument('--disable_chunked_prefill', action='store_true', default=False)
    parser.add_argument('--max_model_len', type=int, default=64000)
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--n_chunks", type=int, default=4)

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


def prepare_data(args):
    examples = list(load_jsonl(args.data_path))

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])

    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # get out_file name
    out_file_prefix = f"{args.prompt_type}_seed{args.seed}_t{args.temperature}topp{args.top_p}minp{args.min_p}topk{args.top_k}_len{args.max_tokens_per_call}_chunks{args.n_chunks}"
    out_file = f"{args.output_dir}/{out_file_prefix}_num{args.num_test_sample}_n{args.n_sampling}.json"
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    return examples, out_file


def prepare_prompt(sample, n_chunks, tokenizer):
    prompt = sample["prompt"]
    gen = sample["gen"]
    thinking_chunks = ["\n</think>\n"]

    thinking = gen.split("<think>")[-1]
    if "</think>" in thinking:
        thinking = thinking.split("</think>")[0].strip()
    
    tok_thinking = tokenizer.encode(thinking)[1:]
    chunk_len = len(tok_thinking) // n_chunks
    for i in range(1, n_chunks):
        thinking_chunks.append(tokenizer.decode(tok_thinking[:i*chunk_len]) + "\n</think>\n")
    
    return [prompt + thinking_chunk for thinking_chunk in thinking_chunks]


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
    tokenizer = llm.get_tokenizer()

    # infer & eval
    main(args, llm, tokenizer)


def main(args, llm, tokenizer):
    samples, out_file = prepare_data(args)

    prompts = []
    for i, sample in tqdm(enumerate(samples), total=len(samples)):
        sample["question"] = sample["question"].strip()
        full_prompt = construct_prompt(sample, args)
        if i == 0:
            print(full_prompt)
        sample["prompt"] = full_prompt

        prompts += prepare_prompt(sample, args.n_chunks, tokenizer)

    # Inf
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
    assert len(outputs) == len(samples) * args.n_chunks
    end_time = time.time()

    # Reformat
    model_outputs = []
    for i in range(len(samples)):
        model_outputs.append([output.outputs[0].text for output in outputs[i*args.n_chunks:(i+1)*args.n_chunks]])

    # Extract pred and eval
    results = []
    avg_acc = []
    for sample, model_output in zip(samples, model_outputs):
        gt = parse_ground_truth(sample)

        preds = []
        scores = []
        for o in model_output:
            pred, score = extract_and_verify_pred(model_output, gt)
            preds.append(pred)
            scores.append(score)
        avg_acc.append(np.mean(scores))

        results.append(
            {
                "ori_ind": sample["ori_ind"],
                "idx": sample["idx"],
                "question": sample["question"],
                "answer": str(sample["answer"]),
                "pred": preds,
                "score": scores,
                "gen": sample["gen"],
                "model_output": model_output,
            }
        )

    time_use = (end_time - start_time) / 60
    result_json = {
        "num_samples": len(samples),
        "pass@1": np.mean(avg_acc),
        "time_use_in_min": time_use,
    }
    print(result_json)

    print(f"Saving model outputs to {out_file}")
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