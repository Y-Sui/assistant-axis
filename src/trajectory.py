"""Shared trajectory parsing utilities.

Parses .traj.json files from mini-swe-agent into clean conversation format.
Used by both the judge pipeline (step 2) and activation extraction (step 3).
"""

import json
from pathlib import Path


def parse_trajectory(traj_path: Path) -> dict | None:
    """Parse a single .traj.json file into a clean format.

    Returns:
        Dict with instance_id, n_assistant_turns, and conversation
        (list of {role, content} dicts), or None if parsing fails.
    """
    with open(traj_path) as f:
        traj = json.load(f)

    # Fallback: use parent dir name (problem-based layout) or filename
    fallback_id = traj_path.parent.name if traj_path.parent.name != "raw" else traj_path.stem.replace(".traj", "")
    instance_id = traj.get("instance_id", fallback_id)

    messages = traj.get("messages", traj.get("history", []))
    if not messages:
        print(f"  Skipping {instance_id}: no messages found")
        return None

    conversation = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("system", "user", "assistant"):
            conversation.append({"role": role, "content": content})
        elif role == "tool":
            # Tool responses become user messages (observation)
            conversation.append({"role": "user", "content": content})
        else:
            conversation.append({"role": role, "content": content})

    n_assistant_turns = sum(1 for m in conversation if m["role"] == "assistant")

    if n_assistant_turns == 0:
        print(f"  Skipping {instance_id}: no assistant turns")
        return None

    return {
        "instance_id": instance_id,
        "n_assistant_turns": n_assistant_turns,
        "conversation": conversation,
    }
