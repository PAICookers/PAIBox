"""Small IR helper utilities shared across lowering and compile passes."""

import torch

__all__ = ["infer_split_output_shapes"]


def infer_split_output_shapes(
    input_shape: torch.Size,
    sections: int | tuple[int, ...],
    dim: int,
) -> tuple[torch.Size, ...]:
    """Pure shape-only model of ``torch.split`` for compile-time reasoning."""
    rank = len(input_shape)
    split_dim = dim if dim >= 0 else dim + rank
    if split_dim < 0 or split_dim >= rank:
        raise ValueError(f"invalid split dim={dim} for rank {rank}")

    input_extent = input_shape[split_dim]
    if isinstance(sections, tuple):
        if sum(sections) != input_extent:
            raise ValueError(
                f"split sections sum to {sum(sections)}, expected {input_extent}"
            )
        sizes = sections
    else:
        if sections <= 0:
            raise ValueError(f"split size must be positive, got {sections}")
        sizes = []
        start = 0
        while start < input_extent:
            sizes.append(min(sections, input_extent - start))
            start += sections
        if not sizes:
            sizes = [0]

    output_shapes: list[torch.Size] = []
    for size in sizes:
        shape = list(input_shape)
        shape[split_dim] = size
        output_shapes.append(torch.Size(shape))

    return tuple(output_shapes)
