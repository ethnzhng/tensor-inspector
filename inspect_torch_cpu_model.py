#!/usr/bin/env python3

import argparse
import json
import os
from collections import OrderedDict

from safetensors import safe_open

from utils import gather_stats, print_stats


def load_torch_model(model_dir):
    state_dict = OrderedDict()
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        # Sharded checkpoint
        with open(index_file, "r") as f:
            index = json.load(f)
        for shard in index["weight_map"].values():
            shard_path = os.path.join(model_dir, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
    else:
        # Single checkpoint
        model_path = os.path.join(model_dir, "model.safetensors")
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    return state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", type=str, help="Path to the directory containing the model files"
    )
    args = parser.parse_args()

    print("Loading model...")
    state_dict = load_torch_model(args.model_dir)

    stats = gather_stats(state_dict)
    print_stats(stats, "Total model stats:")

    print("Done.")


if __name__ == "__main__":
    main()
