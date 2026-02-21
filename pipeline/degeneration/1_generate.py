#!/usr/bin/env python3
"""
Generate paired (clean vs degen) responses for degeneration categories using vLLM.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from assistant_axis.generation import DegenerationResponseGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    root_dir = Path(__file__).resolve().parents[2]
    default_categories_dir = root_dir / "data" / "degeneration" / "categories"
    default_questions_file = root_dir / "data" / "extraction_questions.jsonl"

    parser = argparse.ArgumentParser(description='Generate paired degeneration responses using vLLM')
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--categories_dir', type=str, default=str(default_categories_dir))
    parser.add_argument('--questions_file', type=str, default=str(default_questions_file))
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--question_count', type=int, default=240)

    parser.add_argument('--clean_temperature', type=float, default=0.7)
    parser.add_argument('--clean_max_tokens', type=int, default=256)
    parser.add_argument('--clean_top_p', type=float, default=0.9)

    parser.add_argument('--degen_temperature', type=float, default=1.2)
    parser.add_argument('--degen_max_tokens', type=int, default=512)
    parser.add_argument('--degen_top_p', type=float, default=0.95)

    parser.add_argument('--categories', nargs='+', help='Specific categories to process')
    args = parser.parse_args()

    gen = DegenerationResponseGenerator(
        model_name=args.model,
        categories_dir=args.categories_dir,
        output_dir=args.output_dir,
        questions_file=args.questions_file,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        question_count=args.question_count,
        clean_sampling={
            "temperature": args.clean_temperature,
            "max_tokens": args.clean_max_tokens,
            "top_p": args.clean_top_p,
        },
        degen_sampling={
            "temperature": args.degen_temperature,
            "max_tokens": args.degen_max_tokens,
            "top_p": args.degen_top_p,
        },
    )
    gen.process_all_categories(categories=args.categories)


if __name__ == '__main__':
    main()
