#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter
from functools import reduce
from utils import aggregate_stats, print_stats


def read_safetensors_header(file_path):
    with open(file_path, "rb") as f:
        header_length_bytes = f.read(8)
        header_length = int.from_bytes(header_length_bytes, "little")

        header_bytes = f.read(header_length)
        header = json.loads(header_bytes.decode("utf-8"))

    return header


def gather_stats_from_header(header):
    dtype_counter = Counter()
    num_params = 0
    num_tensors = len(header)

    for tensor_info in header.values():
        dtype_counter[tensor_info["dtype"]] += 1
        shape = tensor_info["shape"]
        num_params += 1 if not shape else reduce(lambda x, y: x * y, shape)

    return {
        "dtype_counter": dtype_counter,
        "num_params": num_params,
        "num_tensors": num_tensors,
        "device": "N/A",
    }


def load_model_stats(model_dir):
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    stats_list = []

    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
        for shard in set(index["weight_map"].values()):
            shard_path = os.path.join(model_dir, shard)
            header = read_safetensors_header(shard_path)
            stats_list.append(gather_stats_from_header(header))
    else:
        model_path = os.path.join(model_dir, "model.safetensors")
        header = read_safetensors_header(model_path)
        stats_list.append(gather_stats_from_header(header))

    return aggregate_stats(stats_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", type=str, help="Path to the directory containing the model files"
    )
    args = parser.parse_args()

    print("Loading safetensors header...")
    stats = load_model_stats(args.model_dir)
    print_stats(stats, "Total model stats:")

    print("Done.")


if __name__ == "__main__":
    main()
