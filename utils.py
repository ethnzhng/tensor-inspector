from collections import Counter


def gather_stats(state_dict):
    dtype_counter = Counter()
    num_params = 0
    num_tensors = len(state_dict)
    device = None

    for tensor in state_dict.values():
        dtype_counter[str(tensor.dtype)] += 1
        num_params += tensor.numel()
        if device is None:
            device = tensor.device

    return {
        "dtype_counter": dtype_counter,
        "num_params": num_params,
        "num_tensors": num_tensors,
        "device": device,
    }


def print_stats(stats, rank_info=""):
    print(f"{rank_info}")
    print(f"Device: {stats['device']}")
    print(f"Number of tensors: {stats['num_tensors']:,}")
    print(f"Number of parameters: {stats['num_params']:,}")
    print("Dtype distribution:")

    sorted_dtypes = sorted(stats["dtype_counter"].items())
    max_dtype_length = max(len(dtype) for dtype, _ in sorted_dtypes)
    max_count = max(count for _, count in sorted_dtypes)
    count_width = len(f"{max_count:,}")

    for dtype, count in sorted_dtypes:
        percentage = count / stats["num_tensors"] * 100
        print(
            f"  {dtype:<{max_dtype_length}} : {count:>{count_width},} tensors ({percentage:6.2f}%)"
        )
    print()


def aggregate_stats(stats_list):
    aggregated = {
        "dtype_counter": Counter(),
        "num_params": 0,
        "num_tensors": 0,
        "device": set(),
    }
    for stats in stats_list:
        aggregated["dtype_counter"] += stats["dtype_counter"]
        aggregated["num_params"] += stats["num_params"]
        aggregated["num_tensors"] += stats["num_tensors"]
        aggregated["device"].add(str(stats["device"]))

    aggregated["device"] = ", ".join(sorted(aggregated["device"]))
    return aggregated
