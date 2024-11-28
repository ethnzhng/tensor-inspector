#!/usr/bin/env python3

import argparse
import os
from collections import OrderedDict

import torch
from vllm import LLM
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.worker.worker import Worker

from utils import aggregate_stats, gather_stats, print_stats


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return get_pp_group().rank_in_group


def get_state_dict_for_rank(self) -> tuple[OrderedDict[str, torch.Tensor], int, int]:
    return (
        self.model_runner.model.state_dict(),
        get_tensor_model_parallel_rank(),
        get_pipeline_model_parallel_rank(),
    )


setattr(Worker, "get_state_dict_for_rank", get_state_dict_for_rank)


def load_vllm_model(model_path, tp_degree, pp_degree):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_degree,
        pipeline_parallel_size=pp_degree,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    state_dicts_and_ranks = llm.llm_engine.model_executor._run_workers(
        method="get_state_dict_for_rank"
    )

    return state_dicts_and_ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", type=str, help="Path to the directory containing the model files"
    )
    parser.add_argument(
        "-t", "--tp_degree", type=int, default=1, help="Tensor parallel degree"
    )
    parser.add_argument(
        "-p", "--pp_degree", type=int, default=1, help="Pipeline parallel degree"
    )
    args = parser.parse_args()

    print("Loading model...")
    state_dicts_and_ranks = load_vllm_model(
        args.model_dir, args.tp_degree, args.pp_degree
    )

    all_stats = []
    for state_dict, tp_rank, pp_rank in state_dicts_and_ranks:
        stats = gather_stats(state_dict)
        all_stats.append(stats)
        print_stats(stats, f"tp rank: {tp_rank}, pp rank {pp_rank}:")

    aggregated_stats = aggregate_stats(all_stats)
    print_stats(aggregated_stats, "Total model stats (all ranks):")

    print("Done.")


if __name__ == "__main__":
    main()
