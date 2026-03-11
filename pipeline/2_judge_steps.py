"""LLM-as-Judge labeling of trajectory steps.

For each assistant turn in each trajectory, calls an LLM judge to evaluate
hallucination and dishonesty. Outputs per-trajectory judgment files.

Usage:
    python pipeline/2_judge_steps.py [--input-dir DIR] [--output-dir DIR] [--judge-model MODEL] [--workers N]
"""

import argparse
import asyncio
import json
from pathlib import Path

import litellm
from tqdm import tqdm

from src.trajectory import parse_trajectory
from src.judge import build_judge_prompt, parse_judge_response, judge_step_async


async def judge_trajectory(
    traj: dict,
    run_id: str,
    output_dir: Path,
    judge_model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Path | None:
    """Judge all assistant steps in a single trajectory.

    Returns the output path on success, None on failure.
    """
    instance_id = traj["instance_id"]
    conversation = traj["conversation"]
    output_path = output_dir / f"{instance_id}__run_{run_id}.judgments.json"

    if output_path.exists():
        return output_path

    # Extract problem statement from system message
    problem_statement = ""
    for msg in conversation:
        if msg["role"] == "system":
            problem_statement = msg["content"]
            break

    # Find assistant turn indices
    assistant_indices = [
        i for i, msg in enumerate(conversation) if msg["role"] == "assistant"
    ]

    steps = []
    for step_index, asst_idx in enumerate(assistant_indices):
        # Build context: all messages before this assistant turn
        context_prefix = conversation[:asst_idx]
        assistant_content = conversation[asst_idx]["content"]

        messages = build_judge_prompt(
            problem_statement, context_prefix, step_index, assistant_content,
        )

        # Call judge with retries
        result = None
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    response_text = await judge_step_async(
                        litellm, judge_model, messages,
                    )
                except Exception as e:
                    print(f"  {instance_id} step {step_index}: API error (attempt {attempt+1}): {e}")
                    continue

            parsed = parse_judge_response(response_text)
            if parsed is not None:
                result = parsed
                break
            else:
                print(f"  {instance_id} step {step_index}: malformed JSON (attempt {attempt+1})")

        if result is None:
            print(f"  {instance_id} step {step_index}: skipping after {max_retries} failed attempts")
            continue

        steps.append({
            "step_index": step_index,
            "global_msg_index": asst_idx,
            "metrics": result,
        })

    judgment = {
        "instance_id": instance_id,
        "run_id": run_id,
        "n_steps": len(steps),
        "steps": steps,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(judgment, f, indent=2)

    return output_path


async def main_async(args):
    """Async main: discover trajectories, judge in parallel."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.workers)

    # Discover trajectories in problem-based layout:
    #   {input_dir}/{instance_id}/run_XX.traj.json
    problem_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())

    tasks = []
    for problem_dir in problem_dirs:
        for traj_path in sorted(problem_dir.glob("run_*.traj.json")):
            # Extract run_id from filename: "run_03.traj.json" -> "03"
            run_id = traj_path.stem.replace(".traj", "").replace("run_", "")

            traj = parse_trajectory(traj_path)
            if traj is None:
                continue

            output_path = args.output_dir / f"{traj['instance_id']}__run_{run_id}.judgments.json"
            if output_path.exists():
                continue

            tasks.append((traj, run_id))

    print(f"Found {len(tasks)} trajectories to judge")

    # Process with progress bar
    results = []
    for traj, run_id in tqdm(tasks, desc="Judging trajectories"):
        result = await judge_trajectory(
            traj, run_id, args.output_dir, args.judge_model, semaphore,
        )
        results.append(result)

    n_success = sum(1 for r in results if r is not None)
    print(f"Done. {n_success}/{len(tasks)} trajectories judged successfully.")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge step labeling")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/raw"),
        help="Directory containing {instance_id} subdirectories with run_XX.traj.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/judgments"),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openrouter/openai/gpt-5.2",
        help="Judge model identifier for litellm",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max concurrent judge API calls",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
