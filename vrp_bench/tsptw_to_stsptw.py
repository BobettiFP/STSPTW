import os
from typing import Tuple

import torch

from vrp_bench.common import load_dataset_torch


# STSPTWEnv uses loc_factor=100 (divides node_xy and tw by 100). So it expects
# inputs in ~0-100 scale. TSP-TW uses map_size 1000 and time 0-1440 minutes.
# Match STSPTWEnv internal loc_factor
LOC_FACTOR = 100
TSPTW_MAP_SIZE = 1000.0
TSPTW_MAX_TIME = 1440.0


def tsptw_batch_to_stsptw_tensors(
    data: dict,
    device: torch.device | None = None,
    scale_for_stsptw: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batched TSP-TW torch dataset dict into the tuple expected by
    R-PIP-Constraint's STSPTWEnv.load_problems:

        (node_xy, service_time, tw_start, tw_end)

    - node_xy:      (batch, n_nodes, 2)
    - service_time: (batch, n_nodes)  (all zeros)
    - tw_start:     (batch, n_nodes)
    - tw_end:       (batch, n_nodes)

    If scale_for_stsptw is True (default), locations and time windows are
    scaled so that after STSPTWEnv's internal /100 normalization they match
    typical STSPTW training data (coordinates 0-1, time 0-~1).
    """
    if "locations" not in data or "time_windows" not in data:
        raise KeyError("Expected keys 'locations' and 'time_windows' in TSP-TW dict")

    node_xy = data["locations"].clone()
    time_windows = data["time_windows"]

    if not isinstance(node_xy, torch.Tensor) or not isinstance(time_windows, torch.Tensor):
        raise TypeError("Expected 'locations' and 'time_windows' to be torch.Tensors")

    if node_xy.dim() != 3 or node_xy.size(-1) != 2:
        raise ValueError(f"'locations' must have shape (B, N, 2), got {tuple(node_xy.shape)}")
    if time_windows.dim() != 3 or time_windows.size(-1) != 2:
        raise ValueError(f"'time_windows' must have shape (B, N, 2), got {tuple(time_windows.shape)}")

    if scale_for_stsptw:
        # STSPTW expects ~0-100 before /100. TSP-TW: locations in [0, map_size], tw in [0, 1440].
        node_xy = node_xy * (LOC_FACTOR / TSPTW_MAP_SIZE)
        tw_start = time_windows[..., 0] * (LOC_FACTOR / TSPTW_MAX_TIME)
        tw_end = time_windows[..., 1] * (LOC_FACTOR / TSPTW_MAX_TIME)
    else:
        tw_start = time_windows[..., 0]
        tw_end = time_windows[..., 1]

    batch_size, n_nodes, _ = node_xy.shape
    service_time = torch.zeros((batch_size, n_nodes), dtype=node_xy.dtype)

    if device is not None:
        node_xy = node_xy.to(device)
        service_time = service_time.to(device)
        tw_start = tw_start.to(device)
        tw_end = tw_end.to(device)

    return node_xy, service_time, tw_start, tw_end


def load_tsptw_pt_as_stsptw(
    path: str,
    map_location: str | torch.device | None = "cpu",
    device: torch.device | None = None,
    scale_for_stsptw: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience helper:

    1. Loads a TSP-TW .pt dataset using vrp_bench.common.load_dataset_torch.
    2. Converts it to (node_xy, service_time, tw_start, tw_end) tensors suitable
       for passing directly to STSPTWEnv.load_problems(problems=...).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TSP-TW dataset not found: {path}")

    data = load_dataset_torch(path, map_location=map_location)
    return tsptw_batch_to_stsptw_tensors(data, device=device, scale_for_stsptw=scale_for_stsptw)

