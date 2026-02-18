#!/usr/bin/env python3
"""
Generate paired (good vs degen) responses for degeneration categories using vLLM.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from assistant_axis.generation import DegenerationResponseGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_categories_on_worker(worker_id: int, gpu_ids: List[int], category_names: List[str], args):
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - Worker-{worker_id}[GPUs:{gpu_ids_str}] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)

    worker_logger.info(f"Starting Worker {worker_id} with GPUs {gpu_ids} and {len(category_names)} categories")

    try:
        generator = DegenerationResponseGenerator(
            model_name=args.model,
            categories_dir=args.categories_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            good_sampling={
                "temperature": args.good_temperature,
                "max_tokens": args.good_max_tokens,
                "top_p": args.good_top_p,
            },
            degen_sampling={
                "temperature": args.degen_temperature,
                "max_tokens": args.degen_max_tokens,
                "top_p": args.degen_top_p,
            },
        )

        generator.good_generator.load()
        generator.degen_generator.load()

        categories_dir = Path(args.categories_dir)
        category_files = {}
        for file_path in sorted(categories_dir.glob("*.json")):
            category_name = file_path.stem
            if category_name in category_names:
                try:
                    category_data = generator.load_category(file_path)
                    if "instruction" in category_data:
                        category_files[category_name] = category_data
                except Exception as e:
                    worker_logger.error(f"Error loading {file_path}: {e}")

        from tqdm import tqdm
        completed_count = 0
        failed_count = 0

        for category_name, category_data in tqdm(category_files.items(), desc=f"Worker-{worker_id}", position=worker_id):
            try:
                responses = []
                for label in ("good", "degen"):
                    responses.extend(generator.generate_category_responses(category_name, category_data, label))
                if responses:
                    generator.save_responses(category_name, responses)
                    completed_count += 1
                else:
                    failed_count += 1
                    worker_logger.warning(f"No responses generated for category '{category_name}'")
            except Exception as e:
                failed_count += 1
                worker_logger.error(f"Exception processing {category_name}: {e}")

        worker_logger.info(f"Worker {worker_id} completed: {completed_count} successful, {failed_count} failed")

    except Exception as e:
        worker_logger.error(f"Fatal error on Worker {worker_id}: {e}")

    finally:
        worker_logger.info(f"Worker {worker_id} cleanup completed")


def run_multi_worker(args) -> int:
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    total_gpus = len(gpu_ids)
    if total_gpus == 0:
        logger.error("No GPUs available.")
        return 1

    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size > total_gpus:
        logger.error(f"tensor_parallel_size ({tensor_parallel_size}) cannot be greater than available GPUs ({total_gpus})")
        return 1

    num_workers = total_gpus // tensor_parallel_size

    if total_gpus % tensor_parallel_size != 0:
        logger.warning(
            f"Total GPUs ({total_gpus}) not evenly divisible by tensor_parallel_size ({tensor_parallel_size}). "
            f"Using {num_workers} workers, leaving {total_gpus % tensor_parallel_size} GPU(s) unused."
        )

    logger.info(f"Available GPUs: {gpu_ids}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Number of workers: {num_workers}")

    categories_dir = Path(args.categories_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_names = []
    for file_path in sorted(categories_dir.glob("*.json")):
        category_name = file_path.stem
        if args.categories and category_name not in args.categories:
            continue
        output_file = output_dir / f"{category_name}.jsonl"
        if output_file.exists():
            logger.info(f"Skipping category '{category_name}' (already exists)")
            continue
        category_names.append(category_name)

    if not category_names:
        logger.info("No categories to process")
        return 0

    logger.info(f"Processing {len(category_names)} categories across {num_workers} workers")

    gpu_chunks = []
    for i in range(num_workers):
        start_gpu_idx = i * tensor_parallel_size
        end_gpu_idx = start_gpu_idx + tensor_parallel_size
        worker_gpus = gpu_ids[start_gpu_idx:end_gpu_idx]
        gpu_chunks.append(worker_gpus)

    categories_per_worker = len(category_names) // num_workers
    remainder = len(category_names) % num_workers

    category_chunks = []
    start_idx = 0
    for i in range(num_workers):
        chunk_size = categories_per_worker + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        chunk = category_names[start_idx:end_idx]
        category_chunks.append(chunk)
        logger.info(f"Worker {i} (GPUs {gpu_chunks[i]}): {len(chunk)} categories")
        start_idx = end_idx

    mp.set_start_method('spawn', force=True)

    processes = []
    for worker_id in range(num_workers):
        if category_chunks[worker_id]:
            p = mp.Process(
                target=process_categories_on_worker,
                args=(worker_id, gpu_chunks[worker_id], category_chunks[worker_id], args)
            )
            p.start()
            processes.append(p)

    logger.info(f"Launched {len(processes)} worker processes")
    for p in processes:
        p.join()

    logger.info("Multi-worker processing completed!")
    return 0


def main():
    root_dir = Path(__file__).resolve().parents[2]
    default_categories_dir = root_dir / "data" / "degeneration" / "categories"
    default_questions_file = root_dir / "data" / "extraction_questions.jsonl"

    parser = argparse.ArgumentParser(
        description='Generate paired degeneration responses using vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--categories_dir', type=str, default=str(default_categories_dir), help='Category JSON files')
    parser.add_argument('--questions_file', type=str, default=str(default_questions_file), help='Questions JSONL')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for JSONL files')
    parser.add_argument('--max_model_len', type=int, default=2048, help='Maximum model context length')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='Number of GPUs (auto-detect if None)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU memory utilization')
    parser.add_argument('--question_count', type=int, default=240, help='Number of questions per category')

    parser.add_argument('--good_temperature', type=float, default=0.7, help='Good sampling temperature')
    parser.add_argument('--good_max_tokens', type=int, default=256, help='Good max tokens')
    parser.add_argument('--good_top_p', type=float, default=0.9, help='Good top-p')

    parser.add_argument('--degen_temperature', type=float, default=1.2, help='Degeneration sampling temperature')
    parser.add_argument('--degen_max_tokens', type=int, default=512, help='Degeneration max tokens')
    parser.add_argument('--degen_top_p', type=float, default=0.95, help='Degeneration top-p')

    parser.add_argument('--categories', nargs='+', help='Specific categories to process')

    args = parser.parse_args()

    if args.tensor_parallel_size is None or args.tensor_parallel_size <= 1:
        gen = DegenerationResponseGenerator(
            model_name=args.model,
            categories_dir=args.categories_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            good_sampling={
                "temperature": args.good_temperature,
                "max_tokens": args.good_max_tokens,
                "top_p": args.good_top_p,
            },
            degen_sampling={
                "temperature": args.degen_temperature,
                "max_tokens": args.degen_max_tokens,
                "top_p": args.degen_top_p,
            },
        )
        gen.process_all_categories(categories=args.categories)
        return

    sys.exit(run_multi_worker(args))


if __name__ == '__main__':
    main()
