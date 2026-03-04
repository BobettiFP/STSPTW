"""
Generate TSP with time windows (TSP-TW) instances for the benchmark.
Single depot (index 0), single vehicle, no capacity (demands: depot 0, rest 1; capacity 1e9).
Hard time windows; stochastic travel time via existing travel_time_generator.
Saves to data/tsp_tw/<timestamp>/ as npz or PyTorch .pt (--format npz|torch).

Feasibility guarantee: a nearest-neighbor reference tour is constructed first,
coordinates are scaled so the tour fits within the planning horizon. Travel times
are sampled along the tour (current_time = arrival at prev node). Time windows
are built around these stochastic arrival times. The reference tour is feasible
under the stored stochastic travel times (route-first, windows-second).
"""
import argparse
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from common import generate_base_instance, save_dataset, save_dataset_torch, save_metadata
from constants import MAP_SIZE, NUM_INSTANCES, PAPER_SEED, REALIZATIONS_PER_MAP
from travel_time_generator import sample_travel_time

SIZES = [10, 20, 30, 40, 50]
RESIDENTIAL_PROB = 0.6
TIME_BUDGET_RATIO = 0.65
MAX_STOCH_RETRIES = 10


def _normalize_sizes(sizes: Union[int, List[int], None]) -> List[int]:
    if sizes is None:
        return SIZES
    if isinstance(sizes, int):
        return [sizes]
    return list(sizes)


def _compute_distance_dict(locations: np.ndarray) -> Dict:
    """Pairwise Euclidean distances from an (n_nodes, 2) coordinate array."""
    n = len(locations)
    distances = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[(i, j)] = 0.0
            else:
                distances[(i, j)] = float(np.linalg.norm(locations[i] - locations[j]))
    return distances


def _nearest_neighbor_tour(n_nodes: int, distances: Dict) -> List[int]:
    """Greedy nearest-neighbor tour starting and ending at depot (node 0)."""
    visited = {0}
    tour = [0]
    current = 0
    for _ in range(n_nodes - 1):
        best_next, best_dist = None, float('inf')
        for j in range(n_nodes):
            if j not in visited and distances[(current, j)] < best_dist:
                best_dist = distances[(current, j)]
                best_next = j
        if best_next is None:
            break
        tour.append(best_next)
        visited.add(best_next)
        current = best_next
    tour.append(0)
    return tour


def _simulate_deterministic_tour(tour: List[int], distances: Dict):
    """Simulate the tour with deterministic travel (distance / velocity=1).
    Returns ({node: arrival_time}, total_time_at_depot_return)."""
    arrivals = {}
    current_time = 0.0
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        current_time += distances[(a, b)]
        arrivals[b] = current_time
    return arrivals, current_time


def _simulate_stochastic_tour(
    tour: List[int], distances: Dict
) -> Tuple[Dict[int, float], float, Dict[Tuple[int, int], float]]:
    """Simulate tour with stochastic travel times (current_time = arrival at prev node).
    Returns (arrivals, total_time, tour_leg_travel_times)."""
    arrivals = {}
    tour_leg_times = {}
    current_time = 0.0
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        tt = sample_travel_time(a, b, distances, current_time)
        tour_leg_times[(a, b)] = round(tt, 2)
        current_time += tt
        arrivals[b] = current_time
    return arrivals, current_time, tour_leg_times


def _generate_feasible_time_windows(
    n_nodes: int,
    arrivals: Dict,
    use_paper_ratio: bool = True,
) -> List:
    """Build time windows that contain the reference tour's arrival times.

    Window widths follow the residential / commercial distribution from the
    paper.  The deterministic arrival is placed in the first 20-40 % of the
    window so that there is room for stochastic delays.
    """
    time_windows = []
    for node in range(n_nodes):
        if node == 0:
            time_windows.append((0.0, 1440.0))
            continue

        arrival = arrivals.get(node, 720.0)

        if use_paper_ratio:
            ctype = int(np.random.choice([0, 1], p=[RESIDENTIAL_PROB, 1 - RESIDENTIAL_PROB]))
        else:
            ctype = random.randint(0, 1)

        if ctype == 0:  # residential: 1–3 h
            window_length = int(np.random.uniform(1, 4)) * 60
        else:           # commercial:  1–2 h
            window_length = int(np.random.uniform(1, 3)) * 60

        position = np.random.uniform(0.2, 0.4)
        start = arrival - position * window_length
        end = start + window_length

        if start < 0:
            end += abs(start)
            start = 0.0
        if end > 1440:
            start -= (end - 1440)
            end = 1440.0
            start = max(0.0, start)

        if end - start < 30:
            end = min(1440.0, start + 60)

        time_windows.append((round(start, 2), round(end, 2)))

    return time_windows


def _verify_feasibility(tour: List[int], distances: Dict, time_windows) -> bool:
    """Quick deterministic feasibility check for the reference tour."""
    current_time = 0.0
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        current_time += distances[(a, b)]
        if b == 0:
            continue
        start, end = time_windows[b]
        if current_time > end:
            return False
        if current_time < start:
            current_time = start
    return True


def _verify_feasibility_stochastic(
    tour: List[int], travel_times: Dict, time_windows: List
) -> bool:
    """Verify reference tour is feasible under stored stochastic travel_times."""
    current_time = 0.0
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        tt = travel_times.get((a, b), 0.0)
        current_time += tt
        if b == 0:
            continue
        start, end = time_windows[b]
        if current_time > end:
            return False
        if current_time < start:
            current_time = start
    return True


def get_time_matrix(n_nodes: int, travel_times: Dict) -> np.ndarray:
    """Build (n_nodes, n_nodes) time matrix from travel_times dict keyed by (i,j)."""
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for (i, j), t in travel_times.items():
        matrix[i, j] = t
    return matrix


def generate_tsp_tw_instance(
    num_customers: int,
    num_cities: int = None,
    instance: Dict = None,
    use_paper_time_ratio: bool = True,
) -> Dict:
    """Generate one TSP-TW instance with at least one guaranteed feasible tour
    under stored stochastic travel times.

    1. Build a nearest-neighbour reference tour on the generated locations.
    2. Scale coordinates so deterministic tour fits within time budget.
    3. Simulate tour with stochastic travel times (current_time = arrival at prev node).
    4. If total stochastic time > 1440, scale down and retry.
    5. Build time windows around stochastic arrival times.
    6. Tour legs: use stochastic values from simulation; non-tour: random sampling.
    """
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    num_depots = 1
    demand_range = (0, 0)
    if instance is None:
        instance = generate_base_instance(
            num_customers,
            MAP_SIZE,
            num_cities,
            num_depots,
            demand_range,
            is_dynamic=False,
        )
    n_nodes = num_customers + num_depots

    demands = np.zeros(n_nodes, dtype=np.float64)
    demands[1:] = 1
    instance["demands"] = demands
    instance["vehicle_capacity"] = int(1e9)

    locations = instance["locations"].astype(np.float64)

    # --- feasibility-guaranteed generation ---
    distances = _compute_distance_dict(locations)
    tour = _nearest_neighbor_tour(n_nodes, distances)
    _, total_det_time = _simulate_deterministic_tour(tour, distances)

    time_budget = 1440 * TIME_BUDGET_RATIO
    if total_det_time > time_budget and total_det_time > 0:
        scale = time_budget / total_det_time
        locations = locations * scale
        instance["locations"] = locations
        distances = _compute_distance_dict(locations)

    # Stochastic simulation with retry if total time exceeds 1440
    for _ in range(MAX_STOCH_RETRIES):
        stoch_arrivals, total_stoch_time, tour_leg_times = _simulate_stochastic_tour(
            tour, distances
        )
        if total_stoch_time <= 1440:
            break
        scale = 1440 / total_stoch_time
        locations = locations * scale
        instance["locations"] = locations
        distances = _compute_distance_dict(locations)
    else:
        raise RuntimeError(
            f"Stochastic tour still exceeds 1440 after {MAX_STOCH_RETRIES} retries"
        )

    time_windows = _generate_feasible_time_windows(
        n_nodes, stoch_arrivals, use_paper_ratio=use_paper_time_ratio,
    )
    instance["time_windows"] = np.array(time_windows, dtype=np.float64)

    # Build full travel_times: tour legs from simulation, non-tour with random current_time
    travel_times = {}
    det_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    noise_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                travel_times[(i, j)] = 0.0
                continue
            det_time = distances[(i, j)]
            if (i, j) in tour_leg_times:
                full_time = tour_leg_times[(i, j)]
            else:
                current_time = random.randint(0, 1440)
                full_time = round(sample_travel_time(i, j, distances, current_time), 2)
            noise = full_time - det_time
            travel_times[(i, j)] = full_time
            det_matrix[i, j] = det_time
            noise_matrix[i, j] = round(noise, 2)

    assert _verify_feasibility_stochastic(tour, travel_times, time_windows), (
        "BUG: reference tour should be feasible under stored stochastic travel times"
    )

    instance["travel_times"] = travel_times
    instance["reference_tour"] = np.array(tour, dtype=np.int64)
    instance["deterministic_travel_times"] = det_matrix
    instance["added_noise"] = noise_matrix

    return instance


def generate_tsp_tw_dataset(
    num_customers: int,
    num_cities: int = None,
    precision=np.float64,
    use_paper_time_ratio: bool = True,
    num_instances: int = None,
) -> tuple:
    """Generate a dataset of TSP-TW instances (same structure as TWCVRP for vrp_base).
    Returns (dataset, metadata) where metadata contains reference_tours, time_windows,
    deterministic_travel_times, added_noise for saving to a separate file."""
    if num_instances is None:
        num_instances = NUM_INSTANCES
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    num_depots = 1
    n_nodes = num_customers + num_depots
    dataset = {
        "num_vehicles": [],
        "locations": [],
        "demands": [],
        "time_matrix": [],
        "time_windows": [],
        "vehicle_capacities": [],
        "travel_times": [],
        "map_size": MAP_SIZE,
        "num_cities": num_cities,
        "num_depots": num_depots,
    }
    metadata = {
        "reference_tours": [],
        "time_windows": [],
        "deterministic_travel_times": [],
        "added_noise": [],
    }
    instance = None
    for _ in tqdm(range(num_instances), desc=f"TSP-TW n={num_customers}"):
        if _ % REALIZATIONS_PER_MAP == 0:
            instance = generate_tsp_tw_instance(
                num_customers,
                num_cities=num_cities,
                instance=None,
                use_paper_time_ratio=use_paper_time_ratio,
            )
        else:
            instance = generate_tsp_tw_instance(
                num_customers,
                num_cities=num_cities,
                instance=instance,
                use_paper_time_ratio=use_paper_time_ratio,
            )
        dataset["locations"].append(instance["locations"].astype(precision))
        dataset["demands"].append(instance["demands"].astype(precision))
        dataset["vehicle_capacities"].append([int(1e9)])
        dataset["num_vehicles"].append(1)
        time_matrix = get_time_matrix(n_nodes, instance["travel_times"])
        dataset["time_matrix"].append(time_matrix.astype(precision))
        dataset["time_windows"].append(instance["time_windows"].astype(precision))
        dataset["travel_times"].append(instance["travel_times"])
        metadata["reference_tours"].append(instance["reference_tour"])
        metadata["time_windows"].append(instance["time_windows"].astype(precision))
        metadata["deterministic_travel_times"].append(instance["deterministic_travel_times"])
        metadata["added_noise"].append(instance["added_noise"])
    dataset_out = {k: np.array(v, dtype=object) for k, v in dataset.items()}
    metadata_out = {k: np.array(v, dtype=object) for k, v in metadata.items()}
    return dataset_out, metadata_out


def main(format: str = None, num_instances: int = None, sizes: Union[int, List[int], None] = None):
    """Generate TSP-TW datasets. format: 'npz' | 'torch' (default from --format or 'npz').
    sizes: single int or list of node sizes (num_customers); default SIZES when None."""
    if format is None:
        parser = argparse.ArgumentParser(description="Generate TSP-TW datasets (npz or PyTorch format)")
        parser.add_argument(
            "--format",
            choices=["npz", "torch"],
            default="npz",
            help="Output format: npz (numpy compressed) or torch (PyTorch tensors). Default: npz",
        )
        parser.add_argument(
            "--num-instances",
            type=int,
            default=None,
            help="Number of instances per node size (default from constants.NUM_INSTANCES)",
        )
        parser.add_argument(
            "--sizes",
            type=int,
            nargs="*",
            default=None,
            help="Node sizes as one or more integers (e.g. 10000 or 10 20 50). Default: 10,20,30,40,50",
        )
        args = parser.parse_args()
        format = args.format
        if num_instances is None:
            num_instances = args.num_instances
        if sizes is None and args.sizes is not None:
            sizes = args.sizes if len(args.sizes) > 0 else None
    size_list = _normalize_sizes(sizes)
    use_torch = format == "torch"
    ext = "pt" if use_torch else "npz"

    np.random.seed(PAPER_SEED)
    random.seed(PAPER_SEED)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join(os.path.dirname(__file__), "data", "tsp_tw", run_timestamp)
    os.makedirs(base, exist_ok=True)
    for num_customers in size_list:
        n_inst = num_instances if num_instances is not None else NUM_INSTANCES
        dataset, metadata = generate_tsp_tw_dataset(num_customers, use_paper_time_ratio=True, num_instances=n_inst)
        path = os.path.join(base, f"tsp_tw_{num_customers}.{ext}")
        meta_path = os.path.join(base, f"tsp_tw_{num_customers}_metadata.{ext}")
        if use_torch:
            save_dataset_torch(dataset, path)
        else:
            save_dataset(dataset, path)
        save_metadata(metadata, meta_path)
    n_inst = num_instances if num_instances is not None else NUM_INSTANCES
    print(f"Done. TSP-TW datasets written under {base} (sizes={size_list}, {n_inst} instances per file, format={format}).")
    return run_timestamp


if __name__ == "__main__":
    main()
