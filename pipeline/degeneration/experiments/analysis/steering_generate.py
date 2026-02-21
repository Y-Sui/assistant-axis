#!/usr/bin/env python3
"""
Experiment 2: Steering generation

For each category:
  1. Generate responses using the degen system prompt (baseline — should be degenerate)
  2. Generate the same prompts WITH the category axis applied (steered — should improve)

Uses HuggingFace model (not vLLM) because steering requires forward hooks.
Output: {output_dir}/{category}.json with paired baseline/steered responses.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from assistant_axis.internals.model import ProbingModel
from assistant_axis.steering import ActivationSteering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--axes_file", required=True)
    parser.add_argument("--categories_dir", required=True)
    parser.add_argument("--questions_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--question_count", type=int, default=50)
    parser.add_argument("--coeff", type=float, default=20.0)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    axes_data = torch.load(args.axes_file, map_location="cpu", weights_only=False)
    axes = axes_data["axes"]

    import jsonlines
    with jsonlines.open(args.questions_file) as reader:
        questions = [e["question"] for e in reader][:args.question_count]

    pm = ProbingModel(args.model)
    categories_dir = Path(args.categories_dir)

    for cat_file in sorted(categories_dir.glob("*.json")):
        category = cat_file.stem
        out_file = output_dir / f"{category}.json"
        if out_file.exists():
            print(f"{category}: skipping (exists)")
            continue
        if category not in axes:
            print(f"{category}: no axis, skipping")
            continue

        cat_data = json.loads(cat_file.read_text())
        # Use first degen system prompt to induce degeneration
        system_prompt = cat_data["instruction"]["degen"][0]

        axis = axes[category]
        best_layer = int(axis.norm(dim=1).argmax().item())
        steering_vec = axis[best_layer]
        print(f"\n{category}: layer={best_layer}  norm={axis[best_layer].norm():.3f}  coeff={args.coeff}")

        results = []
        for q_idx, question in enumerate(questions):
            if pm.supports_system_prompt():
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
            else:
                messages = [{"role": "user", "content": f"{system_prompt}\n\n{question}"}]

            chat_kwargs = {}
            if pm.is_qwen:
                chat_kwargs["enable_thinking"] = False

            prompt = pm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
            inputs = pm.tokenizer(prompt, return_tensors="pt").to(pm.device)
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pm.tokenizer.eos_token_id,
            )

            with torch.no_grad():
                out = pm.model.generate(**inputs, **gen_kwargs)
                baseline = pm.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                with ActivationSteering(pm.model, steering_vectors=[steering_vec],
                                        coefficients=[args.coeff], layer_indices=[best_layer]):
                    out = pm.model.generate(**inputs, **gen_kwargs)
                    steered = pm.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            results.append({
                "question_index": q_idx,
                "question": question,
                "system_prompt": system_prompt,
                "baseline": baseline,
                "steered": steered,
                "category": category,
                "best_layer": best_layer,
                "coeff": args.coeff,
            })

            if (q_idx + 1) % 10 == 0:
                print(f"  {q_idx + 1}/{len(questions)}")

        out_file.write_text(json.dumps(results, indent=2))
        print(f"  saved {len(results)} pairs → {out_file}")


if __name__ == "__main__":
    main()
