"""Named feature partitions used by the multi-view encoder."""

from __future__ import annotations

from ids_diffusion.errors import ViewPartitionError
from ids_diffusion.types import ViewIndices

UNSW_SEMANTIC_VIEWS: tuple[tuple[str, ...], ...] = (
    (
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "sload",
        "dload",
        "sloss",
        "dloss",
        "sinpkt",
        "dinpkt",
        "smean",
        "dmean",
    ),
    (
        "proto",
        "service",
        "state",
        "sttl",
        "dttl",
        "swin",
        "dwin",
        "ct_state_ttl",
        "ct_srv_src",
        "ct_srv_dst",
        "ct_dst_ltm",
        "ct_src_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "is_sm_ips_ports",
    ),
    (
        "sjit",
        "djit",
        "stcpb",
        "dtcpb",
        "tcprtt",
        "synack",
        "ackdat",
        "trans_depth",
        "response_body_len",
        "ct_flw_http_mthd",
        "is_ftp_login",
        "ct_ftp_cmd",
    ),
)


def named_view_indices(
    feature_names: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
) -> ViewIndices:
    """Resolve named groups to indices, rejecting missing or duplicate columns."""
    index = {name: position for position, name in enumerate(feature_names)}
    flattened = tuple(name for group in groups for name in group)
    if len(flattened) != len(set(flattened)):
        raise ViewPartitionError(detail="a feature name appears in more than one view")
    missing = tuple(name for name in flattened if name not in index)
    if missing:
        raise ViewPartitionError(detail=f"dataset is missing named features: {missing}")
    uncovered = tuple(name for name in feature_names if name not in flattened)
    if uncovered:
        raise ViewPartitionError(detail=f"views leave features uncovered: {uncovered}")
    return tuple(tuple(index[name] for name in group) for group in groups)


def positional_view_indices(feature_count: int, view_count: int) -> ViewIndices:
    """Split a vector into contiguous blocks whose sizes differ by at most one."""
    if feature_count < view_count:
        raise ViewPartitionError(detail="cannot create more views than features")
    base, remainder = divmod(feature_count, view_count)
    sizes = tuple(base + (1 if position < remainder else 0) for position in range(view_count))
    cursor = 0
    result: list[tuple[int, ...]] = []
    for size in sizes:
        result.append(tuple(range(cursor, cursor + size)))
        cursor += size
    return tuple(result)


__all__ = ["UNSW_SEMANTIC_VIEWS", "named_view_indices", "positional_view_indices"]
