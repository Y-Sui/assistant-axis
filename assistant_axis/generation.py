"""Minimal generation helpers and compatibility stubs."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch


def supports_system_prompt(tokenizer) -> bool:
    """Heuristic: Gemma-2 chat templates do not support explicit system messages."""
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    return "gemma-2" not in name


def format_conversation(
    instruction: Optional[str],
    question: str,
    tokenizer,
) -> List[Dict[str, str]]:
    """Build a two-message or one-message chat conversation."""
    if instruction:
        if supports_system_prompt(tokenizer):
            return [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question},
            ]
        return [{"role": "user", "content": f"{instruction}\n\n{question}"}]

    return [{"role": "user", "content": question}]


def generate_response(
    model: torch.nn.Module,
    tokenizer,
    conversation: Union[str, List[Dict[str, str]]],
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    do_sample: bool = True,
    suppress_eos: bool = False,
    **chat_kwargs,
) -> str:
    """Generate an assistant response for a chat conversation."""
    if isinstance(conversation, str):
        conversation = [{"role": "user", "content": conversation}]

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if suppress_eos and tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = None

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    generated = output[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


class _UnsupportedGenerator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "This lightweight compatibility stub does not implement this generator class. "
            "Use the degeneration pipeline scripts instead."
        )


class VLLMGenerator(_UnsupportedGenerator):
    pass


class RoleResponseGenerator(_UnsupportedGenerator):
    pass


class DegenerationResponseGenerator(_UnsupportedGenerator):
    pass
