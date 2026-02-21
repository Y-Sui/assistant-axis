#!/usr/bin/env python3
"""
Experiment 2: Steering judge

Scores baseline vs steered responses from steering_generate.py using
the category eval_prompt. Prints mean score comparison per category.

Expected result: steered_mean > baseline_mean for all categories.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from assistant_axis.judge import RateLimiter, call_judge_batch, parse_judge_score

load_dotenv()


async def judge_prompts(client, rate_limiter, judge_model, prompts):
    return await call_judge_batch(
        client=client, prompts=prompts, model=judge_model,
        max_tokens=10, rate_limiter=rate_limiter, batch_size=50,
    )


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", required=True, help="Output dir from steering_generate.py")
    parser.add_argument("--categories_dir", required=True, help="Category JSON files (for eval_prompt)")
    parser.add_argument("--judge_model", default="gpt-4.1-mini")
    args = parser.parse_args()

    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(100)
    categories_dir = Path(args.categories_dir)

    print(f"\n{'category':<20} {'baseline':>10} {'steered':>10} {'delta':>8} {'n':>5}")
    print("-" * 60)

    for pairs_file in sorted(Path(args.pairs_dir).glob("*.json")):
        category = pairs_file.stem
        cat_file = categories_dir / f"{category}.json"
        if not cat_file.exists():
            continue

        eval_prompt_template = json.loads(cat_file.read_text()).get("eval_prompt", "")
        if not eval_prompt_template:
            continue

        pairs = json.loads(pairs_file.read_text())
        baseline_prompts = [
            eval_prompt_template.format(question=p["question"], answer=p["baseline"])
            for p in pairs
        ]
        steered_prompts = [
            eval_prompt_template.format(question=p["question"], answer=p["steered"])
            for p in pairs
        ]

        baseline_texts = await judge_prompts(client, rate_limiter, args.judge_model, baseline_prompts)
        steered_texts  = await judge_prompts(client, rate_limiter, args.judge_model, steered_prompts)

        baseline_scores = [parse_judge_score(t) for t in baseline_texts]
        steered_scores  = [parse_judge_score(t) for t in steered_texts]

        baseline_valid = [s for s in baseline_scores if s is not None]
        steered_valid  = [s for s in steered_scores  if s is not None]

        if not baseline_valid or not steered_valid:
            print(f"{category:<20}  (no valid scores)")
            continue

        b_mean = sum(baseline_valid) / len(baseline_valid)
        s_mean = sum(steered_valid)  / len(steered_valid)
        delta  = s_mean - b_mean
        n      = min(len(baseline_valid), len(steered_valid))
        print(f"{category:<20} {b_mean:>10.3f} {s_mean:>10.3f} {delta:>+8.3f} {n:>5}")

        # Save scores alongside the pairs file
        out = Path(args.pairs_dir) / f"{category}_scores.json"
        out.write_text(json.dumps({
            "baseline_scores": baseline_scores,
            "steered_scores": steered_scores,
        }, indent=2))


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
