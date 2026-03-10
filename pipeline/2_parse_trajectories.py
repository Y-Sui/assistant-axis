"""Parse mini-swe-agent .traj.json files into clean JSONL format.

Reads raw trajectory files and extracts conversation turns with metadata.

Usage:
    python pipeline/2_parse_trajectories.py [--input-dir DIR] [--output FILE]
"""

import argparse
import json
from pathlib import Path


def parse_trajectory(traj_path: Path) -> dict | None:
    """Parse a single .traj.json file into a clean format.

    Returns:
        Dict with instance_id, n_turns, exit_status, and conversation
        (list of {role, content} dicts), or None if parsing fails.
    """
    with open(traj_path) as f:
        traj = json.load(f)

    instance_id = traj.get("instance_id", traj_path.stem.replace(".traj", ""))

    # Extract conversation turns from the trajectory
    # mini-swe-agent stores messages in a "history" or "messages" field
    messages = traj.get("messages", traj.get("history", []))
    if not messages:
        print(f"  Skipping {instance_id}: no messages found")
        return None

    conversation = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Normalize roles
        if role in ("system", "user", "assistant"):
            conversation.append({"role": role, "content": content})
        elif role == "tool":
            # Tool responses become user messages (observation)
            conversation.append({"role": "user", "content": content})
        else:
            # Unknown role, include as-is
            conversation.append({"role": role, "content": content})

    # Count assistant turns
    n_assistant_turns = sum(1 for m in conversation if m["role"] == "assistant")

    if n_assistant_turns == 0:
        print(f"  Skipping {instance_id}: no assistant turns")
        return None

    exit_status = traj.get("exit_status", "unknown")

    return {
        "instance_id": instance_id,
        "n_turns": len(conversation),
        "n_assistant_turns": n_assistant_turns,
        "exit_status": exit_status,
        "conversation": conversation,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse trajectory files into JSONL")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/raw"),
        help="Directory containing .traj.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drift/qwen3-4b/parsed/trajectories.jsonl"),
        help="Output JSONL file path",
    )
    args = parser.parse_args()

    # Find all trajectory files
    traj_files = sorted(args.input_dir.glob("*.traj.json"))
    if not traj_files:
        print(f"No .traj.json files found in {args.input_dir}")
        return

    print(f"Found {len(traj_files)} trajectory files")

    # Parse each trajectory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parsed_count = 0

    with open(args.output, "w") as out_f:
        for traj_path in traj_files:
            print(f"Parsing {traj_path.name}...")
            result = parse_trajectory(traj_path)
            if result is not None:
                out_f.write(json.dumps(result) + "\n")
                parsed_count += 1
                print(
                    f"  {result['instance_id']}: {result['n_assistant_turns']} assistant turns, "
                    f"exit={result['exit_status']}"
                )

    print(f"\nParsed {parsed_count}/{len(traj_files)} trajectories -> {args.output}")


if __name__ == "__main__":
    main()
