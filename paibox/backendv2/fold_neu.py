import numpy as np


def get_skew_info(offsets: list[int]):
    offset_diff = np.diff(offsets)
    diff_num = len(np.unique(offset_diff))
    if diff_num > 3:
        return None

    if diff_num == 1:
        diff = offsets[1] - offsets[0]
        skews = [diff]
        ranges = [len(offsets)]
        return skews, ranges

    skews = []
    ranges = []
    distances = []
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

    skews = skews

    last_distance = len(offsets)
    for i in reversed(range(len(distances))):
        if last_distance % distances[i] != 0:
            return None
        ranges.insert(0, last_distance // distances[i])
        last_distance = distances[i]
    ranges = ranges
    return skews, ranges


import math


def closest_factor(n, partition_num) -> int:
    if partition_num == 1:
        return n
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    # 1. 计算立方根并取整作为起点
    start = int(math.pow(n, 1 / partition_num))

    # 2. 从起点向 1 递减搜索
    for i in range(start, 0, -1):
        if n % i == 0:
            return i
    return 1  # 如果没有找到任何因数，返回 1


def process_exceed(ranges, weight_skews, axon_addr_skews):
    processed_ranges = []
    processed_weight_skews = []
    processed_axon_addr_skews = []
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
                patial_num = num_partition - j
                factor = closest_factor(remain_num, patial_num)
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


def get_fold_info(weight_offsets: list[int], axon_addr_offsets: list[int]):
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
            process_result = process_exceed(ranges, weight_skews, axon_addr_skews)
            return process_result
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
            return ranges, axon_addr_skews, weight_skews
        elif weight_ranges[-1] == axon_addr_ranges[-1]:
            ranges = weight_ranges
            axon_addr_skews = [axon_addr_skews[0]] + axon_addr_skews
            return ranges, axon_addr_skews, weight_skews
        else:
            return None
    elif len(weight_ranges) == 2 and len(axon_addr_ranges) == 3:
        if weight_ranges[0] == axon_addr_ranges[0]:
            ranges = axon_addr_ranges
            weight_skews = weight_skews + [weight_skews[-1]]
            return ranges, axon_addr_skews, weight_skews
        elif weight_ranges[-1] == axon_addr_ranges[-1]:
            ranges = axon_addr_ranges
            weight_skews = [weight_skews[0]] + weight_skews
            return ranges, axon_addr_skews, weight_skews
        else:
            return None
    else:
        return None
