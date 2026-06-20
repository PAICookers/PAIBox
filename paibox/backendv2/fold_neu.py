import math

import numpy as np


def get_skew_info(offsets: list[int]) -> tuple[list[int], list[int]] | None:
    offset_diff = np.diff(offsets)
    diff_num = len(np.unique(offset_diff))
    if diff_num > 3:
        return None

    if diff_num == 1:
        diff = offsets[1] - offsets[0]
        skews = [diff]
        ranges = [len(offsets)]
        return skews, ranges

    skews: list[int] = []
    ranges: list[int] = []
    distances: list[int] = []
    for i, diff in enumerate(offset_diff):
        if len(skews) == 0:
            skews.append(diff.item())
            distances.append(1)
        else:
            # has not meet the next skew, continue counting
            for j in reversed(range(len(distances))):
                last_distance = distances[j]
                if (i + 1) % last_distance == 0:
                    if diff == skews[j]:
                        break
                    else:
                        # meet the next skew, insert it after the current skew
                        if j == len(distances) - 1:
                            skews.append(diff.item())
                            distances.append(i + 1)
                            break
                        else:
                            return None
                else:
                    continue

    last_distance = len(offsets)
    for i in reversed(range(len(distances))):
        if last_distance % distances[i] != 0:
            return None
        ranges.insert(0, last_distance // distances[i])
        last_distance = distances[i]
    return skews, ranges


def closest_factor(n: int, partition_num: int) -> int:
    if partition_num == 1:
        return n
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    start = int(math.pow(n, 1 / partition_num))
    return next((i for i in range(start, 0, -1) if n % i == 0), 1)


def process_exceed(
    ranges: list[int], weight_skews: list[int], axon_addr_skews: list[int]
) -> tuple[list[int], list[int], list[int]] | None:
    processed_ranges: list[int] = []
    processed_weight_skews: list[int] = []
    processed_axon_addr_skews: list[int] = []
    remain_space = 3 - len(ranges)
    for i, range_ in enumerate(ranges):
        if range_ < 2048:
            processed_ranges.append(range_)
            processed_weight_skews.append(weight_skews[i])
            processed_axon_addr_skews.append(axon_addr_skews[i])
        else:
            num_partition = math.ceil(math.log(range_, 2047))
            if num_partition > remain_space:
                return None
            remain_num = range_
            for j in range(num_partition):
                partial_num = num_partition - j
                factor = closest_factor(remain_num, partial_num)
                if factor < 2048:
                    processed_ranges.append(factor)
                    processed_weight_skews.append(weight_skews[i])
                    processed_axon_addr_skews.append(axon_addr_skews[i])
                    remain_num = remain_num // factor
                    continue
                else:
                    return None
            remain_space -= num_partition - 1
    processed_ranges = processed_ranges + [1] * (3 - len(processed_ranges))
    processed_weight_skews = processed_weight_skews + [1] * (
        3 - len(processed_weight_skews)
    )
    processed_axon_addr_skews = processed_axon_addr_skews + [1] * (
        3 - len(processed_axon_addr_skews)
    )
    return processed_ranges, processed_weight_skews, processed_axon_addr_skews


def get_fold_info(
    weight_offsets: list[int], axon_addr_offsets: list[int]
) -> tuple[list[int], list[int], list[int]] | None:
    """Return fold ranges with weight-skew and axon-skew sequences.

    The public contract is `(ranges, weight_skews, axon_addr_skews)`. Keep every
    branch in this order because routing writes the two skew families into
    different folded-neuron config fields.
    """
    # weight_offsets_diff = np.diff(weight_offsets)
    # axon_addr_offsets_diff = np.diff(axon_addr_offsets)
    weight_info = get_skew_info(weight_offsets)
    axon_addr_info = get_skew_info(axon_addr_offsets)
    if weight_info is None or axon_addr_info is None:
        return None
    weight_skews, weight_ranges = weight_info
    # print(f"weight_skews: {weight_skews}, weight_ranges: {weight_ranges}")
    axon_addr_skews, axon_addr_ranges = axon_addr_info
    # print(f"axon_addr_skews: {axon_addr_skews}, axon_addr_ranges: {axon_addr_ranges}")
    if len(weight_ranges) == len(axon_addr_ranges):
        if weight_ranges == axon_addr_ranges:
            ranges = weight_ranges
            return process_exceed(ranges, weight_skews, axon_addr_skews)
        else:
            return None
    elif len(weight_ranges) == 1 or len(axon_addr_ranges) == 1:
        if len(weight_ranges) == 1:
            ranges = axon_addr_ranges
            weight_skews = weight_skews * len(ranges)
        else:
            ranges = weight_ranges
            axon_addr_skews = axon_addr_skews * len(ranges)
        return process_exceed(ranges, weight_skews, axon_addr_skews)
    elif len(weight_ranges) == 3 and len(axon_addr_ranges) == 2:
        if weight_ranges[0] == axon_addr_ranges[0]:
            ranges = weight_ranges
            axon_addr_skews = axon_addr_skews + [axon_addr_skews[-1]]
            return ranges, weight_skews, axon_addr_skews
        elif weight_ranges[-1] == axon_addr_ranges[-1]:
            ranges = weight_ranges
            axon_addr_skews = [axon_addr_skews[0]] + axon_addr_skews
            return ranges, weight_skews, axon_addr_skews
        else:
            return None
    elif len(weight_ranges) == 2 and len(axon_addr_ranges) == 3:
        if weight_ranges[0] == axon_addr_ranges[0]:
            ranges = axon_addr_ranges
            weight_skews = weight_skews + [weight_skews[-1]]
            return ranges, weight_skews, axon_addr_skews
        elif weight_ranges[-1] == axon_addr_ranges[-1]:
            ranges = axon_addr_ranges
            weight_skews = [weight_skews[0]] + weight_skews
            return ranges, weight_skews, axon_addr_skews
        else:
            return None
    else:
        return None
