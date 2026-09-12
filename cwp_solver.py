#!/usr/bin/env python3
"""Solver-free heuristic for the discrete one-rail CWP variant.

The scheduling algorithm is implemented entirely in Python and does not call
OR-Tools, Gurobi, CPLEX, SCIP, PuLP, or any other optimization solver.
Matplotlib is used only to draw the final schedule.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import multiprocessing
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Slot:
    time: int
    crane: int
    state: str
    start_bay: int
    end_bay: int
    work_bay: int | None


@dataclass
class Solution:
    status: str
    method: str
    makespan: int
    makespan_lower_bound: int
    lower_bound_components: dict[str, int]
    makespan_proven_optimal: bool
    proven_lexicographic_optimal: bool
    assignment_count: int
    split_bay_count: int
    load_deviation: int
    reversal_count: int
    movement_count: int
    crane_loads: list[int]
    target_weights: list[int]
    bay_cranes: dict[int, list[int]]
    slots: list[Slot]
    restarts_completed: int
    strategy_evaluations: list[int]
    layered_search_states: int
    layered_search_improvements: int
    mcts_iterations: int
    mcts_improvements: int
    exact_search_nodes: int
    exact_search_improvements: int
    exact_search_proved_optimal: bool
    search_seconds: float
    max_steps: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bay_cranes"] = {
            str(bay): cranes for bay, cranes in self.bay_cranes.items()
        }
        return result


@dataclass(frozen=True)
class _Strategy:
    strict_owner: bool
    equal_load_target: bool
    work_weight: float
    ready_weight: float
    split_penalty: float
    balance_weight: float
    move_penalty: float
    reversal_penalty: float
    priority_weight: float
    noise: float


@dataclass
class _CandidateSchedule:
    slots: list[Slot]
    makespan: int
    assignment_count: int
    split_bay_count: int
    load_deviation: int
    reversal_count: int
    movement_count: int
    loads: list[int]
    owners: list[set[int]]

    @property
    def objective_key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.makespan,
            self.split_bay_count,
            self.assignment_count,
            self.reversal_count,
            self.movement_count,
            self.load_deviation,
        )


def _count_reversals(slots: Sequence[Slot], M: int) -> int:
    """Count changes between leftward and rightward moves for each crane."""
    last_direction = [0] * M
    reversals = 0
    for slot in slots:
        if slot.state != "move":
            continue
        q = slot.crane - 1
        direction = 1 if slot.end_bay > slot.start_bay else -1
        if last_direction[q] and direction != last_direction[q]:
            reversals += 1
        last_direction[q] = direction
    return reversals


def _legal_configurations(N: int, M: int) -> list[tuple[int, ...]]:
    """Generate only safe configurations instead of filtering all combinations.

    If ``y`` is a strictly increasing M-combination from ``1..N-M+1``, then
    ``y[q] + q`` has a gap of at least two.  This is a bijection, so no legal
    configuration is lost and no illegal combination is ever materialized.
    """
    return [
        tuple(bay + q for q, bay in enumerate(compressed))
        for compressed in itertools.combinations(range(1, N - M + 2), M)
    ]


def _validate_input(
    W: list[int], M: int, S: Iterable[int]
) -> tuple[int, list[int], list[tuple[int, ...]]]:
    if not isinstance(W, list) or not W:
        raise ValueError("W 必须是非空整数列表。")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in W):
        raise ValueError("W 中每个作业量必须是非负整数。")
    if isinstance(M, bool) or not isinstance(M, int) or M <= 0:
        raise ValueError("M 必须是正整数。")

    N = len(W)
    if M > (N + 1) // 2:
        raise ValueError(f"N={N} 时最多只能停放 {(N + 1) // 2} 台互不相邻的桥吊。")

    starts = sorted(S)
    if len(starts) != len(set(starts)):
        raise ValueError("S 不能包含重复贝位。")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= N for v in starts):
        raise ValueError(f"S 中贝位必须是 1..{N} 的整数。")
    if len(starts) > M:
        raise ValueError("强制开工贝位数量不能超过桥吊数量 M。")
    if any(right - left < 2 for left, right in zip(starts, starts[1:])):
        raise ValueError("S 中任意两个贝位不能相邻。")
    if any(W[bay - 1] == 0 for bay in starts):
        raise ValueError("S 中贝位必须具有正作业量，否则 t=0 无法开工作业。")

    configurations = _legal_configurations(N, M)
    required = set(starts)
    if not any(required.issubset(config) for config in configurations):
        raise ValueError("S 无法扩充为 M 台桥吊的合法初始停位组合。")

    coverable = {bay for config in configurations for bay in config}
    unreachable = [i + 1 for i, amount in enumerate(W) if amount > 0 and i + 1 not in coverable]
    if unreachable:
        raise ValueError(
            "在当前 N、M 和安全间距下，以下有作业量的贝位永远无法被占用："
            + ", ".join(map(str, unreachable))
        )
    return N, starts, configurations


def _load_deviation(loads: Sequence[int], weights: Sequence[int], total_work: int) -> int:
    weight_sum = sum(weights)
    return sum(abs(weight_sum * load - total_work * weight) for load, weight in zip(loads, weights))


def _bay_eligibility(
    N: int,
    M: int,
    configurations: Sequence[tuple[int, ...]],
) -> list[set[int]]:
    """Crane identities that can legally occupy each bay."""
    # Crane q can occupy every bay between its leftmost and rightmost safe
    # positions.  Computing this interval directly avoids scanning all legal
    # configurations, which becomes expensive as N and M grow.
    eligible = [set() for _ in range(N)]
    for q in range(M):
        first_bay = 1 + 2 * q
        last_bay = N - 2 * (M - q - 1)
        for bay in range(first_bay, last_bay + 1):
            eligible[bay - 1].add(q)
    return eligible


def _bay_criticality(
    W: Sequence[int],
    M: int,
    eligibility: Sequence[set[int]],
) -> list[float]:
    """Combine contiguous-window pressure and crane-eligibility scarcity."""
    N = len(W)
    pressure = [float(amount) for amount in W]
    for left in range(N):
        window_work = 0
        for right in range(left, N):
            window_work += W[right]
            length = right - left + 1
            parallel_capacity = min(M, (length + 1) // 2)
            density = window_work / parallel_capacity
            for bay_index in range(left, right + 1):
                pressure[bay_index] = max(pressure[bay_index], density)
    pressure_scale = max(pressure, default=1.0) or 1.0
    result = []
    for bay_index, value in enumerate(pressure):
        scarcity = M / max(1, len(eligibility[bay_index]))
        result.append(0.70 * value / pressure_scale + 0.30 * scarcity)
    return result


def _makespan_lower_bound(
    W: Sequence[int],
    M: int,
    initial_configs: Sequence[tuple[int, ...]],
    eligibility: Sequence[set[int]],
) -> tuple[int, dict[str, int]]:
    """Return valid workload, movement, and congestion-window lower bounds."""
    total_work = sum(W)
    positive_bays = sum(amount > 0 for amount in W)
    initial_positive_capacity = max(
        (sum(W[bay - 1] > 0 for bay in config) for config in initial_configs),
        default=0,
    )
    minimum_crane_moves = max(0, positive_bays - initial_positive_capacity)
    workload_bound = math.ceil(total_work / M)
    active_time_bound = math.ceil((total_work + minimum_crane_moves) / M)

    congestion_bound = 0
    window_active_bound = 0
    N = len(W)
    for left in range(N):
        window_work = 0
        positive_in_window: set[int] = set()
        for right in range(left, N):
            window_work += W[right]
            if W[right] > 0:
                positive_in_window.add(right + 1)
            length = right - left + 1
            parallel_capacity = min(M, (length + 1) // 2)
            congestion_bound = max(
                congestion_bound,
                math.ceil(window_work / parallel_capacity),
            )
            # At most parallel_capacity cranes can end a slot inside this
            # interval. Every work slot and every first arrival at a positive
            # bay consumes one such endpoint, giving a stronger valid bound.
            initial_window_coverage = max(
                (
                    sum(bay in positive_in_window for bay in config)
                    for config in initial_configs
                ),
                default=0,
            )
            minimum_window_arrivals = max(
                0, len(positive_in_window) - initial_window_coverage
            )
            window_active_bound = max(
                window_active_bound,
                math.ceil(
                    (window_work + minimum_window_arrivals) / parallel_capacity
                ),
            )

    # Hall-type bound for every contiguous subset of crane identities. Bays
    # whose eligible cranes all lie inside the subset must consume this
    # subset's work and first-visit movement capacity.
    eligibility_bound = 0
    for first_q in range(M):
        for last_q in range(first_q, M):
            crane_count = last_q - first_q + 1
            mandatory_bays = [
                bay_index
                for bay_index, amount in enumerate(W)
                if amount > 0
                and eligibility[bay_index]
                and all(first_q <= q <= last_q for q in eligibility[bay_index])
            ]
            if not mandatory_bays:
                continue
            mandatory_set = {bay_index + 1 for bay_index in mandatory_bays}
            initial_coverage = max(
                (
                    sum(
                        first_q <= q <= last_q and bay in mandatory_set
                        for q, bay in enumerate(config)
                    )
                    for config in initial_configs
                ),
                default=0,
            )
            minimum_subset_moves = max(0, len(mandatory_bays) - initial_coverage)
            mandatory_work = sum(W[bay_index] for bay_index in mandatory_bays)
            eligibility_bound = max(
                eligibility_bound,
                math.ceil((mandatory_work + minimum_subset_moves) / crane_count),
            )

    components = {
        "single_bay": max(W, default=0),
        "total_workload": workload_bound,
        "work_plus_minimum_moves": active_time_bound,
        "contiguous_window_congestion": congestion_bound,
        "contiguous_window_work_plus_moves": window_active_bound,
        "eligible_crane_bottleneck": eligibility_bound,
    }
    return max(components.values(), default=0), components


def _remaining_lower_bound(
    remaining: Sequence[int],
    M: int,
    positions: tuple[int, ...],
    eligibility: Sequence[set[int]],
) -> int:
    """Admissible slots still needed from one partial-schedule state."""
    total_work = sum(remaining)
    if total_work == 0:
        return 0
    positive = {i + 1 for i, amount in enumerate(remaining) if amount > 0}
    initial_coverage = sum(bay in positive for bay in positions)
    bound = max(
        max(remaining),
        math.ceil(total_work / M),
        math.ceil((total_work + len(positive) - initial_coverage) / M),
    )

    N = len(remaining)
    for left in range(N):
        window_work = 0
        positive_window: set[int] = set()
        for right in range(left, N):
            window_work += remaining[right]
            if remaining[right] > 0:
                positive_window.add(right + 1)
            capacity = min(M, (right - left + 2) // 2)
            arrivals = len(positive_window) - sum(
                bay in positive_window for bay in positions
            )
            bound = max(
                bound,
                math.ceil((window_work + max(0, arrivals)) / capacity),
            )

    for first_q in range(M):
        for last_q in range(first_q, M):
            mandatory = {
                bay_index + 1
                for bay_index, amount in enumerate(remaining)
                if amount > 0
                and eligibility[bay_index]
                and all(first_q <= q <= last_q for q in eligibility[bay_index])
            }
            if not mandatory:
                continue
            crane_count = last_q - first_q + 1
            covered = sum(
                positions[q] in mandatory for q in range(first_q, last_q + 1)
            )
            mandatory_work = sum(remaining[bay - 1] for bay in mandatory)
            bound = max(
                bound,
                math.ceil(
                    (mandatory_work + len(mandatory) - covered) / crane_count
                ),
            )
    return bound


def _initial_configurations(
    W: Sequence[int],
    starts: Sequence[int],
    configurations: Sequence[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    required = set(starts)
    eligible = [config for config in configurations if required.issubset(config)]
    return sorted(
        eligible,
        key=lambda config: (
            sum(W[bay - 1] > 0 for bay in config),
            sum(min(W[bay - 1], 8) for bay in config),
            -sum(abs(2 * bay - (len(W) + 1)) for bay in config),
        ),
        reverse=True,
    )


def _greedy_owner_plan(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    initial: tuple[int, ...],
    eligibility: Sequence[set[int]],
    bay_criticality: Sequence[float],
    rng: random.Random,
    force_extra_initial: bool,
    middle_bias: float,
) -> list[int | None]:
    """Assign each positive bay to one crane before schedule decoding."""
    owner: list[int | None] = [None] * len(W)
    estimated_active = [0.0] * M
    required = set(starts)

    # A bay processed at its initial position needs no first-visit move. Strong
    # start bays are mandatory; other initial bays are optional diversification.
    for q, bay in enumerate(initial):
        bay_index = bay - 1
        if W[bay_index] <= 0:
            continue
        if bay in required or force_extra_initial:
            owner[bay_index] = q
            estimated_active[q] += W[bay_index]

    unassigned = [i for i, amount in enumerate(W) if amount > 0 and owner[i] is None]
    rng.shuffle(unassigned)
    unassigned.sort(
        key=lambda i: (len(eligibility[i]), -bay_criticality[i], -W[i])
    )
    center = (M - 1) / 2
    for bay_index in unassigned:
        candidates = sorted(eligibility[bay_index])
        best_q = min(
            candidates,
            key=lambda q: (
                estimated_active[q]
                + W[bay_index]
                + 1
                - middle_bias * (center - abs(q - center)),
                rng.random(),
            ),
        )
        owner[bay_index] = best_q
        estimated_active[best_q] += W[bay_index] + 1
    return owner


def _best_partition_owner_plan(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    initial: tuple[int, ...],
    eligibility: Sequence[set[int]],
) -> list[int | None] | None:
    """Dynamic-programming seed with contiguous crane work regions."""
    N = len(W)
    total_work = sum(W)
    weights = [min(q + 1, M - q) for q in range(M)]
    weight_sum = sum(weights)
    initial_crane = {bay: q for q, bay in enumerate(initial)}
    required = set(starts)

    @functools.lru_cache(maxsize=None)
    def dp(q: int, left: int) -> tuple[int, int, int, tuple[int, ...]] | None:
        if q == M:
            return (0, 0, 0, ()) if left == N else None
        ends = (N,) if q == M - 1 else range(left, N + 1)
        best: tuple[int, int, int, tuple[int, ...]] | None = None
        for right in ends:
            positive = [i for i in range(left, right) if W[i] > 0]
            if any(q not in eligibility[i] for i in positive):
                continue
            if any(
                bay in required and initial_crane.get(bay) != q
                for bay in range(left + 1, right + 1)
            ):
                continue
            tail = dp(q + 1, right)
            if tail is None:
                continue
            work = sum(W[i] for i in positive)
            visits = len(positive)
            starts_on_work = any(i + 1 == initial[q] for i in positive)
            active = work + max(0, visits - int(starts_on_work))
            deviation = abs(weight_sum * work - total_work * weights[q])
            candidate = (
                max(active, tail[0]),
                deviation + tail[1],
                active + tail[2],
                (right,) + tail[3],
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        return best

    result = dp(0, 0)
    if result is None:
        return None
    owner: list[int | None] = [None] * N
    left = 0
    for q, right in enumerate(result[3]):
        for bay_index in range(left, right):
            if W[bay_index] > 0:
                owner[bay_index] = q
        left = right
    return owner


def _relax_partition_boundaries(
    owner: Sequence[int | None],
    W: Sequence[int],
    radius: int,
) -> list[int | None]:
    """Open a few partition-boundary bays to controlled shared processing.

    ``None`` means that the decoder may use any physically eligible crane.
    Only positive-work bays near a change of owner are opened, keeping the
    strong contiguous seed while allowing adjacent cranes to rebalance work.
    """
    relaxed = list(owner)
    positive = [index for index, amount in enumerate(W) if amount > 0]
    boundaries = [
        ordinal
        for ordinal, (left, right) in enumerate(zip(positive, positive[1:]))
        if owner[left] is not None
        and owner[right] is not None
        and owner[left] != owner[right]
    ]
    for ordinal in boundaries:
        for positive_ordinal in range(
            max(0, ordinal - radius + 1),
            min(len(positive), ordinal + radius + 1),
        ):
            relaxed[positive[positive_ordinal]] = None
    return relaxed


def _focused_next_configurations(
    positions: tuple[int, ...],
    remaining: Sequence[int],
    configurations: Sequence[tuple[int, ...]],
    eligibility: Sequence[set[int]],
    owners: Sequence[set[int]],
    bay_criticality: Sequence[float],
    strategy: _Strategy,
    rng: random.Random,
    fixed_owner: Sequence[int | None] | None,
    preferred_owner: Sequence[int | None] | None,
    last_move_direction: Sequence[int],
    width: int,
) -> list[tuple[int, ...]]:
    """Build a small, high-value transition neighborhood.

    Moving across any distance costs one slot in this variant, so useful next
    positions are primarily the current bay, high-pressure unfinished bays,
    and an occasional extreme parking bay that can unblock another crane.
    """
    M = len(positions)
    N = len(remaining)
    choices: list[list[int]] = []
    for q in range(M):
        ranked: list[tuple[float, float, int]] = []
        for bay_index, left in enumerate(remaining):
            if left <= 0 or q not in eligibility[bay_index]:
                continue
            if (
                fixed_owner is not None
                and fixed_owner[bay_index] is not None
                and fixed_owner[bay_index] != q
            ):
                continue
            if strategy.strict_owner and owners[bay_index] and q not in owners[bay_index]:
                continue
            direction = 0
            if bay_index + 1 != positions[q]:
                direction = 1 if bay_index + 1 > positions[q] else -1
            direction_score = (
                0.05 * strategy.reversal_penalty
                if direction == 0 or last_move_direction[q] in (0, direction)
                else -0.25 * strategy.reversal_penalty
            )
            ranked.append(
                (
                    bay_criticality[bay_index]
                    + 0.08 * min(left, 10)
                    + (0.55 if preferred_owner is not None and preferred_owner[bay_index] == q else 0.0)
                    + direction_score,
                    rng.random(),
                    bay_index + 1,
                )
            )
        ranked.sort(reverse=True)
        selected = [positions[q]]
        selected.extend(item[2] for item in ranked[:width])

        # Preserve exploration without growing the Cartesian product too much.
        if len(ranked) > width:
            selected.append(rng.choice(ranked[width:])[2])
        min_bay = 1 + 2 * q
        max_bay = N - 2 * (M - q - 1)
        selected.append(min_bay if rng.random() < 0.5 else max_bay)
        selected = list(dict.fromkeys(selected))
        # Avoid an exponential neighborhood explosion with many cranes.  Keep
        # the current bay and the strongest ranked bays, then rotate one
        # exploratory/boundary option between restarts.
        option_cap = 5 if M <= 3 else 4 if M <= 5 else 3
        if len(selected) > option_cap:
            core_count = min(len(selected), 1 + width, option_cap - 1)
            core = selected[:core_count]
            tail = selected[core_count:]
            rng.shuffle(tail)
            selected = core + tail[: option_cap - core_count]
        choices.append(selected)

    result = {
        config
        for config in itertools.product(*choices)
        if all(right - left >= 2 for left, right in zip(config, config[1:]))
    }
    result.add(positions)
    return list(result) if result else list(configurations)


def _construct_schedule(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    configurations: Sequence[tuple[int, ...]],
    initial: tuple[int, ...],
    strategy: _Strategy,
    rng: random.Random,
    max_steps: int,
    bay_criticality: Sequence[float],
    eligibility: Sequence[set[int]],
    fixed_owner: Sequence[int | None] | None = None,
    preferred_owner: Sequence[int | None] | None = None,
    focused_width: int | None = None,
    deadline: float | None = None,
) -> _CandidateSchedule:
    remaining = list(W)
    remaining_total = sum(remaining)
    total_work = remaining_total
    positions = initial
    owners: list[set[int]] = [set() for _ in W]
    loads = [0] * M
    target_weights = [min(q + 1, M - q) for q in range(M)]
    weight_sum = sum(target_weights)
    if strategy.equal_load_target:
        target_loads = [total_work / M] * M
    else:
        target_loads = [total_work * weight / weight_sum for weight in target_weights]
    required = set(starts)
    mandatory_cranes = {q for q, bay in enumerate(initial) if bay in required}
    slots: list[Slot] = []
    last_move_direction = [0] * M
    no_progress_slots = 0

    for t in range(max_steps):
        if deadline is not None and time.perf_counter() >= deadline:
            raise RuntimeError("构造搜索达到本阶段时间上限。")
        if remaining_total == 0:
            break

        best_normal: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
        best_progress: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None

        next_configurations = configurations
        if focused_width is not None:
            next_configurations = _focused_next_configurations(
                positions, remaining, configurations, eligibility, owners,
                bay_criticality, strategy, rng, fixed_owner, preferred_owner,
                last_move_direction, focused_width,
            )

        for config_index, next_positions in enumerate(next_configurations):
            if (
                deadline is not None
                and config_index % 256 == 0
                and time.perf_counter() >= deadline
            ):
                raise RuntimeError("构造搜索达到本阶段时间上限。")
            if t == 0 and any(next_positions[q] != positions[q] for q in mandatory_cranes):
                continue

            working: list[int] = []
            new_splits = 0
            for q in range(M):
                bay_index = positions[q] - 1
                if next_positions[q] != positions[q] or remaining[bay_index] <= 0:
                    continue
                if (
                    fixed_owner is not None
                    and fixed_owner[bay_index] is not None
                    and fixed_owner[bay_index] != q
                ):
                    continue
                if strategy.strict_owner and owners[bay_index] and q not in owners[bay_index]:
                    continue
                working.append(q)
                if owners[bay_index] and q not in owners[bay_index]:
                    new_splits += 1

            working_set = set(working)
            ready_count = 0
            ready_volume = 0
            balance_gain = 0.0
            priority_gain = sum(bay_criticality[positions[q] - 1] for q in working)
            owner_hint_gain = 0.0
            if preferred_owner is not None:
                owner_hint_gain = sum(
                    preferred_owner[positions[q] - 1] == q for q in working
                )
            for q, bay in enumerate(next_positions):
                bay_index = bay - 1
                left = remaining[bay_index]
                if q in working_set and positions[q] == bay:
                    left -= 1
                owner_compatible = (
                    fixed_owner[bay_index] in (None, q)
                    if fixed_owner is not None
                    else not owners[bay_index] or q in owners[bay_index]
                )
                if left > 0 and (owner_compatible or not strategy.strict_owner):
                    ready_count += 1
                    ready_volume += min(left, 6)
                    balance_gain += max(0.0, target_loads[q] - loads[q])
                    priority_gain += bay_criticality[bay_index]
                    if preferred_owner is not None and preferred_owner[bay_index] == q:
                        owner_hint_gain += 0.60

            if not working and ready_count == 0:
                continue

            movement_count = sum(a != b for a, b in zip(positions, next_positions))
            reversal_count = 0
            for q, (start_bay, end_bay) in enumerate(zip(positions, next_positions)):
                if start_bay == end_bay:
                    continue
                direction = 1 if end_bay > start_bay else -1
                if last_move_direction[q] and direction != last_move_direction[q]:
                    reversal_count += 1
            score = (
                strategy.work_weight * len(working)
                + strategy.ready_weight * ready_count
                + 0.12 * ready_volume
                + strategy.balance_weight * balance_gain
                + strategy.priority_weight * priority_gain
                + 1.25 * owner_hint_gain
                - strategy.split_penalty * new_splits
                - strategy.move_penalty * movement_count
                - strategy.reversal_penalty * reversal_count
                + strategy.noise * rng.random()
            )
            item = (score, next_positions, tuple(working))
            if best_normal is None or score > best_normal[0]:
                best_normal = item
            if working and (best_progress is None or score > best_progress[0]):
                best_progress = item

        chosen = best_progress if no_progress_slots >= 1 and best_progress is not None else best_normal
        if chosen is None:
            raise RuntimeError("自定义搜索无法继续构造排程；请检查输入。")

        _, next_positions, working_tuple = chosen
        working_set = set(working_tuple)
        for q in range(M):
            start_bay = positions[q]
            end_bay = next_positions[q]
            if q in working_set:
                bay_index = start_bay - 1
                remaining[bay_index] -= 1
                remaining_total -= 1
                owners[bay_index].add(q)
                loads[q] += 1
                slots.append(Slot(t, q + 1, "work", start_bay, end_bay, start_bay))
            elif start_bay != end_bay:
                slots.append(Slot(t, q + 1, "move", start_bay, end_bay, None))
                last_move_direction[q] = 1 if end_bay > start_bay else -1
            else:
                slots.append(Slot(t, q + 1, "idle", start_bay, end_bay, None))

        no_progress_slots = 0 if working_set else no_progress_slots + 1
        positions = next_positions

    if remaining_total > 0:
        raise RuntimeError(f"在安全构造上界 max_steps={max_steps} 内未完成全部作业。")

    makespan = len(slots) // M
    assignment_count = sum(len(bay_owners) for bay_owners in owners)
    split_bay_count = sum(len(bay_owners) > 1 for bay_owners in owners)
    movement_count = sum(slot.state == "move" for slot in slots)
    reversal_count = _count_reversals(slots, M)
    return _CandidateSchedule(
        slots=slots,
        makespan=makespan,
        assignment_count=assignment_count,
        split_bay_count=split_bay_count,
        load_deviation=_load_deviation(loads, target_weights, total_work),
        reversal_count=reversal_count,
        movement_count=movement_count,
        loads=loads,
        owners=owners,
    )


def _focused_exact_neighbors(
    positions: tuple[int, ...],
    remaining: Sequence[int],
    eligibility: Sequence[set[int]],
    bay_criticality: Sequence[float],
    attempt: int,
    choice_width: int,
    guide_positions: tuple[int, ...] | None,
) -> list[tuple[int, ...]]:
    """Generate a bounded but diverse set of legal next configurations."""
    choices: list[list[int]] = []
    for q, current_bay in enumerate(positions):
        ranked = sorted(
            (
                (
                    bay_criticality[bay_index] + 0.10 * min(amount, 12),
                    amount,
                    bay_index + 1,
                )
                for bay_index, amount in enumerate(remaining)
                if amount > 0 and q in eligibility[bay_index]
            ),
            reverse=True,
        )
        best_work = [item[2] for item in ranked[:choice_width]]
        guide = guide_positions[q] if guide_positions is not None else None
        exploratory = None
        if len(ranked) > choice_width:
            pool_size = min(len(ranked), choice_width + 6)
            exploratory = ranked[(choice_width + attempt + q) % pool_size][2]
        N = len(remaining)
        parking = (
            1 + 2 * q
            if (attempt + q) % 2 == 0
            else N - 2 * (len(positions) - q - 1)
        )
        # Bound the Cartesian product when many cranes are deployed.
        max_options = 4 if len(positions) <= 3 else 3 if len(positions) == 4 else 2
        roles = list(dict.fromkeys([
            *best_work,
            guide,
            exploratory,
            parking,
        ]))
        roles = [bay for bay in roles if bay is not None and bay != current_bay]
        available = max_options - 1
        if available == 1 and roles:
            # With many cranes the Cartesian product permits only one move
            # candidate per crane. Rotate its role across layers/variants so
            # work, incumbent guidance, exploration and parking all survive.
            preferred_cycle = [
                *(best_work[:1] * 2), guide, exploratory, parking
            ]
            preferred_cycle = [
                bay for bay in preferred_cycle
                if bay is not None and bay != current_bay
            ]
            chosen = preferred_cycle[(attempt + q) % len(preferred_cycle)]
            selected = [current_bay, chosen]
        else:
            # Always keep the strongest work target; rotate the remaining
            # roles so no fixed prefix silently discards them.
            selected = [current_bay]
            if roles:
                selected.append(roles[0])
                tail = roles[1:]
                if tail and available > 1:
                    offset = (attempt + q) % len(tail)
                    rotated = tail[offset:] + tail[:offset]
                    selected.extend(rotated[: available - 1])
        selected = list(dict.fromkeys(selected))
        choices.append(selected)

    neighbors = {
        config
        for config in itertools.product(*choices)
        if all(right - left >= 2 for left, right in zip(config, config[1:]))
    }
    neighbors.add(positions)

    # Deterministic best-first order matters to both the bounded layered
    # search and DFS.  Prefer work now, then positions that are ready to work
    # in the following slot, while using the incumbent only as a tie-breaker.
    def priority(config: tuple[int, ...]) -> tuple[float, ...]:
        working = sum(
            start_bay == end_bay and remaining[start_bay - 1] > 0
            for start_bay, end_bay in zip(positions, config)
        )
        ready = sum(remaining[bay - 1] > 0 for bay in config)
        pressure = sum(
            bay_criticality[bay - 1]
            for bay in config
            if remaining[bay - 1] > 0
        )
        guide_matches = (
            sum(a == b for a, b in zip(config, guide_positions))
            if guide_positions is not None else 0
        )
        return (-working, -ready, -pressure, -guide_matches, *config)

    return sorted(neighbors, key=priority)


def _candidate_from_history(
    W: Sequence[int],
    M: int,
    history: Sequence[tuple[int, ...]],
) -> _CandidateSchedule:
    """Decode a layered/DFS configuration path into the normal result type."""
    remaining = list(W)
    owners: list[set[int]] = [set() for _ in W]
    loads = [0] * M
    slots: list[Slot] = []
    for t, (positions, next_positions) in enumerate(zip(history, history[1:])):
        for q, (start_bay, end_bay) in enumerate(zip(positions, next_positions)):
            bay_index = start_bay - 1
            if start_bay == end_bay and remaining[bay_index] > 0:
                remaining[bay_index] -= 1
                owners[bay_index].add(q)
                loads[q] += 1
                slots.append(Slot(t, q + 1, "work", start_bay, end_bay, start_bay))
            elif start_bay != end_bay:
                slots.append(Slot(t, q + 1, "move", start_bay, end_bay, None))
            else:
                slots.append(Slot(t, q + 1, "idle", start_bay, end_bay, None))
    if any(remaining):
        raise RuntimeError("精确搜索路径没有完成全部作业。")
    target_weights = [min(q + 1, M - q) for q in range(M)]
    movement_count = sum(slot.state == "move" for slot in slots)
    return _CandidateSchedule(
        slots=slots,
        makespan=len(history) - 1,
        assignment_count=sum(len(item) for item in owners),
        split_bay_count=sum(len(item) > 1 for item in owners),
        load_deviation=_load_deviation(loads, target_weights, sum(W)),
        reversal_count=_count_reversals(slots, M),
        movement_count=movement_count,
        loads=loads,
        owners=owners,
    )


def _bounded_layered_search(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    initial_configs: Sequence[tuple[int, ...]],
    eligibility: Sequence[set[int]],
    bay_criticality: Sequence[float],
    incumbent: _CandidateSchedule,
    deadline: float,
    variant: int = 0,
    cutoff: int | None = None,
) -> tuple[_CandidateSchedule | None, int]:
    """Quickly seek a schedule one slot shorter with bounded dynamic programming.

    States are merged by ``(remaining work, crane positions)`` after every
    time layer.  Keeping only the most promising states makes this an
    incomplete search, so it can improve an incumbent but never claims a
    proof.  The later complete DFS is solely responsible for proof by
    enumeration.
    """
    target = incumbent.makespan - 1
    if target < 0:
        return None, 0
    required = set(starts)
    if M <= 3:
        beam_width, choice_width = 8_000, 2
    elif M == 4:
        beam_width, choice_width = 3_000, 2
    elif M == 5:
        beam_width, choice_width = 1_000, 1
    else:
        beam_width, choice_width = 400, 1

    incumbent_history: list[tuple[int, ...]] = []
    for t in range(incumbent.makespan):
        rows = sorted(
            (slot for slot in incumbent.slots if slot.time == t),
            key=lambda slot: slot.crane,
        )
        if t == 0:
            incumbent_history.append(tuple(slot.start_bay for slot in rows))
        incumbent_history.append(tuple(slot.end_bay for slot in rows))

    def quick_bound(
        remaining: tuple[int, ...], positions: tuple[int, ...]
    ) -> int:
        total = sum(remaining)
        if total == 0:
            return 0
        positive_count = sum(amount > 0 for amount in remaining)
        covered = sum(remaining[bay - 1] > 0 for bay in positions)
        return max(
            max(remaining),
            math.ceil(total / M),
            math.ceil((total + positive_count - covered) / M),
        )

    states: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[int, ...], ...],
    ] = {}
    start_used = 0
    original_remaining = tuple(W)
    # Variants after the first one perform a ruin-and-recreate repair of the
    # incumbent tail.  The prefix remains fixed while all crane decisions in
    # the selected bottleneck tail are searched again together.
    if cutoff is not None and target >= 3:
        start_used = min(target - 2, max(1, cutoff))
        tail_remaining = list(W)
        for positions, next_positions in zip(
            incumbent_history[:start_used], incumbent_history[1:start_used + 1]
        ):
            for start_bay, end_bay in zip(positions, next_positions):
                if start_bay == end_bay and tail_remaining[start_bay - 1] > 0:
                    tail_remaining[start_bay - 1] -= 1
        remaining_tuple = tuple(tail_remaining)
        start_positions = incumbent_history[start_used]
        states[(remaining_tuple, start_positions)] = tuple(
            incumbent_history[:start_used + 1]
        )
    elif variant > 0 and target >= 6:
        fractions = (0.35, 0.55, 0.70)
        start_used = min(target - 2, max(1, int(target * fractions[(variant - 1) % 3])))
        tail_remaining = list(W)
        for positions, next_positions in zip(
            incumbent_history[:start_used], incumbent_history[1:start_used + 1]
        ):
            for start_bay, end_bay in zip(positions, next_positions):
                if start_bay == end_bay and tail_remaining[start_bay - 1] > 0:
                    tail_remaining[start_bay - 1] -= 1
        remaining_tuple = tuple(tail_remaining)
        start_positions = incumbent_history[start_used]
        states[(remaining_tuple, start_positions)] = tuple(
            incumbent_history[:start_used + 1]
        )
    else:
        for initial in initial_configs:
            if quick_bound(original_remaining, initial) <= target:
                states[(original_remaining, initial)] = (initial,)
    if len(states) > beam_width:
        states = dict(list(states.items())[:beam_width])

    evaluated = 0
    for used in range(start_used, target):
        if time.perf_counter() >= deadline:
            break
        depth_left = target - used - 1
        next_states: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[int, ...], ...],
        ] = {}
        guide = (
            incumbent_history[used + 1]
            if used + 1 < len(incumbent_history) else None
        )
        for (remaining, positions), history in states.items():
            if evaluated % 2_048 == 0 and time.perf_counter() >= deadline:
                break
            neighbors = _focused_exact_neighbors(
                positions, remaining, eligibility, bay_criticality,
                used + variant * 7, choice_width, guide,
            )
            mandatory_cranes = {
                q for q, bay in enumerate(positions) if bay in required
            } if used == 0 else set()
            for next_positions in neighbors:
                evaluated += 1
                if mandatory_cranes and any(
                    next_positions[q] != positions[q] for q in mandatory_cranes
                ):
                    continue
                new_remaining = list(remaining)
                work_count = 0
                for start_bay, end_bay in zip(positions, next_positions):
                    if (
                        start_bay == end_bay
                        and new_remaining[start_bay - 1] > 0
                    ):
                        new_remaining[start_bay - 1] -= 1
                        work_count += 1
                if work_count == 0 and next_positions == positions:
                    continue
                remaining_tuple = tuple(new_remaining)
                if quick_bound(remaining_tuple, next_positions) > depth_left:
                    continue
                key = (remaining_tuple, next_positions)
                if key not in next_states:
                    next_states[key] = history + (next_positions,)

        if not next_states:
            break
        for (remaining, _), history in next_states.items():
            if not any(remaining):
                return _candidate_from_history(W, M, history), evaluated

        def state_priority(
            item: tuple[
                tuple[tuple[int, ...], tuple[int, ...]],
                tuple[tuple[int, ...], ...],
            ]
        ) -> tuple[float, ...]:
            (remaining, positions), _ = item
            ready = sum(remaining[bay - 1] > 0 for bay in positions)
            pressure = sum(
                bay_criticality[bay - 1]
                for bay in positions
                if remaining[bay - 1] > 0
            )
            return (sum(remaining), max(remaining), -ready, -pressure)

        if len(next_states) > beam_width:
            ordered_states = sorted(next_states.items(), key=state_priority)
            # Reserve part of the beam for different spatial arrangements.
            # Without this, thousands of nearly identical states can crowd out
            # a temporary parking/letting-through maneuver needed later.
            bucket_size = max(2, math.ceil(len(W) / 6))
            diversity_limit = max(1, beam_width // 5)
            diverse = []
            used_keys = set()
            for item in ordered_states:
                (remaining, positions), _ = item
                bottleneck = max(range(len(remaining)), key=remaining.__getitem__)
                signature = (
                    tuple((bay - 1) // bucket_size for bay in positions),
                    bottleneck // bucket_size,
                )
                if signature in used_keys:
                    continue
                used_keys.add(signature)
                diverse.append(item)
                if len(diverse) >= diversity_limit:
                    break
            selected_keys = {item[0] for item in diverse}
            selected = diverse
            selected.extend(
                item for item in ordered_states
                if item[0] not in selected_keys
            )
            states = dict(selected[:beam_width])
        else:
            states = next_states
    return None, evaluated


def _blocking_repair_cutoffs(
    incumbent: _CandidateSchedule,
    M: int,
) -> list[int]:
    """Choose repair starts near late idle/move events of the finishing crane."""
    last_work = [-1] * M
    by_time: dict[int, list[Slot]] = {}
    for slot in incumbent.slots:
        by_time.setdefault(slot.time, []).append(slot)
        if slot.state == "work":
            last_work[slot.crane - 1] = slot.time
    bottleneck = max(range(M), key=last_work.__getitem__)
    involved = {bottleneck}
    if bottleneck > 0:
        involved.add(bottleneck - 1)
    if bottleneck + 1 < M:
        involved.add(bottleneck + 1)
    start_scan = max(1, incumbent.makespan // 3)
    events = [
        t
        for t in range(start_scan, incumbent.makespan)
        if any(
            slot.crane - 1 in involved and slot.state != "work"
            for slot in by_time.get(t, [])
        )
    ]
    candidates = []
    if events:
        candidates.extend((max(1, events[0] - 1), max(1, events[len(events) // 2] - 1)))
    candidates.extend((incumbent.makespan * 55 // 100, incumbent.makespan * 70 // 100))
    return list(dict.fromkeys(candidates))[:3]


def _mcts_horizon_search(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    eligibility: Sequence[set[int]],
    bay_criticality: Sequence[float],
    incumbent: _CandidateSchedule,
    deadline: float,
    seed: int,
) -> tuple[_CandidateSchedule | None, int]:
    """Use partial-schedule MCTS to seek a schedule one slot shorter.

    Unlike the strategy UCB portfolio, nodes here are actual partial schedules.
    Selection uses UCB, one new transition is expanded at a time, and a fast
    stochastic rollout evaluates the downstream consequence of that action.
    This is an improvement search only and never produces an optimality proof.
    """
    target = incumbent.makespan - 1
    if target <= 0:
        return None, 0
    rng = random.Random(seed)
    required = set(starts)
    incumbent_history: list[tuple[int, ...]] = []
    for t in range(incumbent.makespan):
        rows = sorted(
            (slot for slot in incumbent.slots if slot.time == t),
            key=lambda slot: slot.crane,
        )
        if t == 0:
            incumbent_history.append(tuple(slot.start_bay for slot in rows))
        incumbent_history.append(tuple(slot.end_bay for slot in rows))
    root = (tuple(W), incumbent_history[0], 0)
    visits: dict[tuple[tuple[int, ...], tuple[int, ...], int], int] = {}
    values: dict[tuple[tuple[int, ...], tuple[int, ...], int], float] = {}
    children: dict[
        tuple[tuple[int, ...], tuple[int, ...], int],
        list[tuple[tuple[int, ...], tuple[int, ...], int]],
    ] = {}

    def fast_bound(remaining: tuple[int, ...], positions: tuple[int, ...]) -> int:
        total = sum(remaining)
        if total == 0:
            return 0
        positive = sum(amount > 0 for amount in remaining)
        covered = sum(remaining[bay - 1] > 0 for bay in positions)
        return max(
            max(remaining),
            math.ceil(total / M),
            math.ceil((total + positive - covered) / M),
        )

    def advance(
        remaining: tuple[int, ...],
        positions: tuple[int, ...],
        next_positions: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        updated = list(remaining)
        worked = 0
        for start_bay, end_bay in zip(positions, next_positions):
            if start_bay == end_bay and updated[start_bay - 1] > 0:
                updated[start_bay - 1] -= 1
                worked += 1
        if worked == 0 and next_positions == positions:
            return None
        return tuple(updated)

    def actions(
        state: tuple[tuple[int, ...], tuple[int, ...], int],
        salt: int,
    ) -> list[tuple[int, ...]]:
        remaining, positions, used = state
        guide = incumbent_history[used + 1] if used + 1 < len(incumbent_history) else None
        result = _focused_exact_neighbors(
            positions, remaining, eligibility, bay_criticality,
            used + salt, 2 if M <= 4 else 1, guide,
        )
        if used == 0:
            mandatory = {q for q, bay in enumerate(positions) if bay in required}
            result = [
                config for config in result
                if all(config[q] == positions[q] for q in mandatory)
            ]
        return result

    iterations = 0
    while time.perf_counter() < deadline:
        iterations += 1
        state = root
        path = [state]
        history = [root[1]]

        # Select and expand one partial-schedule node.
        while state[2] < target and any(state[0]):
            remaining, positions, used = state
            candidates = []
            for config in actions(state, iterations):
                updated = advance(remaining, positions, config)
                if updated is None:
                    continue
                if fast_bound(updated, config) > target - used - 1:
                    continue
                candidates.append((updated, config, used + 1))
            if not candidates:
                break
            known = children.setdefault(state, [])
            unexpanded = [child for child in candidates if child not in known]
            if unexpanded:
                child = rng.choice(unexpanded)
                known.append(child)
                state = child
                path.append(state)
                history.append(state[1])
                break
            parent_visits = max(1, visits.get(state, 0))
            state = max(
                known,
                key=lambda child: (
                    values.get(child, 0.0) / max(1, visits.get(child, 0))
                    + 0.7 * math.sqrt(
                        math.log(parent_visits + 1) / max(1, visits.get(child, 0))
                    )
                ),
            )
            path.append(state)
            history.append(state[1])

        # Fast stochastic rollout from the newly selected node.
        remaining, positions, used = state
        while used < target and any(remaining) and time.perf_counter() < deadline:
            ranked = []
            for config in actions((remaining, positions, used), iterations + used):
                updated = advance(remaining, positions, config)
                if updated is None:
                    continue
                bound = fast_bound(updated, config)
                if bound > target - used - 1:
                    continue
                ready = sum(updated[bay - 1] > 0 for bay in config)
                ranked.append(((sum(updated), bound, -ready), updated, config))
            if not ranked:
                break
            ranked.sort(key=lambda item: item[0])
            _, remaining, positions = rng.choice(ranked[: min(3, len(ranked))])
            used += 1
            history.append(positions)
        complete = not any(remaining)
        if complete:
            return _candidate_from_history(W, M, history), iterations
        bound = fast_bound(remaining, positions)
        reward = 1.0 / (1.0 + sum(remaining) / M + bound)
        for visited_state in path:
            visits[visited_state] = visits.get(visited_state, 0) + 1
            values[visited_state] = values.get(visited_state, 0.0) + reward
    return None, iterations


class _ExactSearchTimeout(RuntimeError):
    pass


def _exact_horizon_search(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    configurations: Sequence[tuple[int, ...]],
    initial_configs: Sequence[tuple[int, ...]],
    eligibility: Sequence[set[int]],
    bay_criticality: Sequence[float],
    incumbent: _CandidateSchedule,
    deadline: float,
) -> tuple[_CandidateSchedule | None, str, int]:
    """Complete DFS for a schedule shorter than the incumbent.

    Focused configurations are visited first, but every legal configuration is
    eventually considered. Therefore an ``INFEASIBLE`` result is a proof for
    the tested horizon; timeout never produces an optimality claim.
    """
    target = incumbent.makespan - 1
    required = set(starts)
    incumbent_history: list[tuple[int, ...]] = []
    for t in range(incumbent.makespan):
        rows = sorted(
            (slot for slot in incumbent.slots if slot.time == t),
            key=lambda slot: slot.crane,
        )
        if t == 0:
            incumbent_history.append(tuple(slot.start_bay for slot in rows))
        incumbent_history.append(tuple(slot.end_bay for slot in rows))

    failed: set[tuple[tuple[int, ...], tuple[int, ...], int]] = set()
    nodes = 0

    def dfs(
        remaining: tuple[int, ...],
        positions: tuple[int, ...],
        used: int,
        history: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...] | None:
        nonlocal nodes
        nodes += 1
        if nodes % 128 == 0 and time.perf_counter() >= deadline:
            raise _ExactSearchTimeout
        if not any(remaining):
            return history
        depth_left = target - used
        if depth_left <= 0:
            return None
        if _remaining_lower_bound(remaining, M, positions, eligibility) > depth_left:
            return None
        key = (remaining, positions, depth_left)
        if key in failed:
            return None

        guide = (
            incumbent_history[used + 1]
            if used + 1 < len(incumbent_history) else None
        )
        focused = _focused_exact_neighbors(
            positions, remaining, eligibility, bay_criticality,
            used, 2 if M <= 4 else 1, guide,
        )
        focused_set = set(focused)
        ordered = itertools.chain(
            focused,
            (config for config in configurations if config not in focused_set),
        )
        mandatory_cranes = {
            q for q, bay in enumerate(positions) if bay in required
        } if used == 0 else set()

        for next_positions in ordered:
            if mandatory_cranes and any(
                next_positions[q] != positions[q] for q in mandatory_cranes
            ):
                continue
            new_remaining = list(remaining)
            work_count = 0
            for start_bay, end_bay in zip(positions, next_positions):
                if start_bay == end_bay and new_remaining[start_bay - 1] > 0:
                    new_remaining[start_bay - 1] -= 1
                    work_count += 1
            if work_count == 0 and next_positions == positions:
                continue
            remaining_tuple = tuple(new_remaining)
            result = dfs(
                remaining_tuple,
                next_positions,
                used + 1,
                history + (next_positions,),
            )
            if result is not None:
                return result
        failed.add(key)
        return None

    selected_initials = list(initial_configs)
    if incumbent_history:
        selected_initials = list(
            dict.fromkeys([incumbent_history[0], *selected_initials])
        )
    try:
        for initial in selected_initials:
            if time.perf_counter() >= deadline:
                raise _ExactSearchTimeout
            if _remaining_lower_bound(W, M, initial, eligibility) > target:
                continue
            history = dfs(tuple(W), initial, 0, (initial,))
            if history is not None:
                return _candidate_from_history(W, M, history), "FOUND", nodes
    except _ExactSearchTimeout:
        return None, "TIMEOUT", nodes
    return None, "INFEASIBLE", nodes


def solve_cwp(
    W: list[int],
    M: int,
    S: Iterable[int],
    *,
    restarts: int = 100_000,
    time_limit: float = 270.0,
    seed: int = 20260910,
    patience: int = 2_000,
) -> Solution:
    """Construct a safe schedule without calling an optimization solver."""
    search_start = time.perf_counter()
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts <= 0:
        raise ValueError("restarts 必须是正整数。")
    if time_limit <= 0:
        raise ValueError("time_limit 必须大于 0。")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError("patience 必须是正整数。")
    effective_time_limit = min(float(time_limit), 270.0)
    deadline = search_start + effective_time_limit
    _, starts, configurations = _validate_input(W, M, S)

    total_work = sum(W)
    max_steps = max(1, 3 * total_work + len(W) + 1)
    target_weights = [min(q + 1, M - q) for q in range(M)]
    initial_configs = _initial_configurations(W, starts, configurations)
    eligibility = _bay_eligibility(len(W), M, configurations)
    lower_bound, lower_bound_components = _makespan_lower_bound(
        W, M, initial_configs, eligibility
    )
    bay_criticality = _bay_criticality(W, M, eligibility)
    rng = random.Random(seed)
    use_exact_search = effective_time_limit >= 30.0
    heuristic_deadline = (
        search_start + 0.55 * effective_time_limit
        if use_exact_search else deadline
    )
    layered_deadline = (
        search_start + 0.95 * effective_time_limit
        if use_exact_search else deadline
    )
    best: _CandidateSchedule | None = None
    completed = 0
    last_improvement = 0

    # The first five rules preserve the original decoder; the second five add
    # congestion-window priority. An online UCB selector allocates more trials
    # to rules that perform well on the current instance.
    base_strategies = [
        _Strategy(True, False, 13.0, 3.0, 40.0, 0.00, 0.08, 0.0, 0.0, 0.00),
        _Strategy(True, False, 9.0, 5.0, 25.0, 0.00, 0.05, 0.0, 0.0, 0.35),
        _Strategy(False, False, 13.0, 3.0, 8.0, 0.00, 0.08, 0.0, 0.0, 0.30),
        _Strategy(False, False, 9.0, 5.0, 2.0, 0.14, 0.05, 0.0, 0.0, 0.70),
        _Strategy(False, False, 7.0, 6.0, 0.0, 0.05, 0.02, 0.0, 0.0, 1.00),
        _Strategy(False, True, 9.0, 5.0, 0.0, 0.30, 0.04, 0.0, 2.5, 0.60),
        _Strategy(False, True, 7.0, 6.0, -1.0, 0.45, 0.02, 0.0, 4.0, 0.90),
    ]
    priority_weights = (2.5, 3.5, 1.5, 3.5, 5.0, 4.0, 5.0)
    priority_strategies = [
        _Strategy(
            rule.strict_owner, rule.equal_load_target,
            rule.work_weight, rule.ready_weight,
            rule.split_penalty, rule.balance_weight, rule.move_penalty,
            rule.reversal_penalty, priority_weight, rule.noise,
        )
        for rule, priority_weight in zip(base_strategies, priority_weights)
    ]
    reversal_strategies = [
        _Strategy(False, False, 10.0, 4.5, 5.0, 0.05, 0.05, penalty, 3.0, 0.45)
        for penalty in (2.0, 6.0, 12.0)
    ]
    strategies = base_strategies + priority_strategies + reversal_strategies
    strategy_counts = [0] * len(strategies)
    strategy_rewards = [0.0] * len(strategies)
    top_pool_size = min(len(initial_configs), max(20, min(200, restarts)))
    top_pool = initial_configs[:top_pool_size]
    warmup_initial_count = min(8, len(top_pool))
    warmup_trials = warmup_initial_count * len(strategies)
    partition_plan_cache: dict[tuple[int, ...], list[int | None] | None] = {}

    for restart in range(restarts):
        if restart > 0 and time.perf_counter() >= heuristic_deadline:
            break
        if (
            best is not None
            and best.makespan == lower_bound
            and restart - last_improvement >= patience
        ):
            break
        if restart < warmup_trials:
            strategy_index = restart % len(strategies)
            initial = top_pool[restart // len(strategies)]
        elif rng.random() < 0.20:
            strategy_index = rng.randrange(len(strategies))
        else:
            # Self-adaptive UCB: exploit successful dispatch rules while the
            # incumbent is improving; gradually raise exploration after a
            # long period without improvement.
            stagnation = restart - last_improvement
            exploration = 0.12 + 0.38 * min(
                1.0, stagnation / max(50.0, 0.5 * patience)
            )
            strategy_index = max(
                range(len(strategies)),
                key=lambda index: (
                    strategy_rewards[index] / strategy_counts[index]
                    + exploration
                    * math.sqrt(math.log(restart + 1) / strategy_counts[index])
                ),
            )
        if restart >= warmup_trials:
            if rng.random() < 0.8:
                initial = rng.choice(top_pool)
            else:
                initial = rng.choice(initial_configs)
        strategy = strategies[strategy_index]
        fixed_owner = None
        partition_phase = restart % 17
        if partition_phase in (1, 6, 11):
            if initial not in partition_plan_cache:
                partition_plan_cache[initial] = _best_partition_owner_plan(
                    W, M, starts, initial, eligibility
                )
            base_partition = partition_plan_cache[initial]
            if base_partition is not None and partition_phase != 1:
                fixed_owner = _relax_partition_boundaries(
                    base_partition, W, 1 if partition_phase == 6 else 2
                )
            else:
                fixed_owner = base_partition
        elif restart >= warmup_trials and restart % 20 == 1:
            fixed_owner = _greedy_owner_plan(
                W, M, starts, initial, eligibility, bay_criticality, rng,
                force_extra_initial=(restart % 4 != 0),
                middle_bias=0.0 if restart % 5 else 0.35,
            )
        preferred_owner: list[int | None] | None = None
        if best is not None and restart >= warmup_trials and restart % 3 != 0:
            # ALNS-style destroy/repair hint: inherit most bay owners from the
            # incumbent, erase part of them, and occasionally mutate one.  The
            # hint is soft, so the decoder can still escape to a better plan.
            preferred_owner = [
                next(iter(bay_owners)) if len(bay_owners) == 1 else None
                for bay_owners in best.owners
            ]
            destroy_rate = (0.20, 0.35, 0.50)[restart % 3]
            for bay_index in range(len(W)):
                if rng.random() < destroy_rate:
                    preferred_owner[bay_index] = None
            mutable = [
                i for i, amount in enumerate(W)
                if amount > 0 and len(eligibility[i]) > 1
            ]
            if mutable and rng.random() < 0.35:
                bay_index = rng.choice(mutable)
                alternatives = list(eligibility[bay_index])
                rng.shuffle(alternatives)
                preferred_owner[bay_index] = alternatives[0]
        strategy_counts[strategy_index] += 1
        previous_best_makespan = best.makespan if best is not None else max_steps
        try:
            candidate = _construct_schedule(
                W, M, starts, configurations, initial, strategy, rng,
                min(max_steps, 2 * total_work + 1) if fixed_owner is not None else max_steps,
                bay_criticality, eligibility, fixed_owner, preferred_owner,
                None if restart == 0 or restart % 75 == 0 else 2,
                None if best is None else heuristic_deadline,
            )
        except RuntimeError:
            # Some randomized fixed-owner plans can trap the greedy decoder.
            # They are discarded; unrestricted decoders remain in the portfolio.
            continue
        completed += 1
        gap = max(0, candidate.makespan - lower_bound)
        reward = (
            1.0 / (1 + gap)
            + 0.25 * max(0, previous_best_makespan - candidate.makespan)
            + 0.05 / (1 + candidate.split_bay_count)
            + 0.01 / (1 + candidate.assignment_count)
            + 0.002 / (1 + candidate.reversal_count)
        )
        strategy_rewards[strategy_index] += reward
        if best is None or candidate.objective_key < best.objective_key:
            best = candidate
            last_improvement = restart

    assert best is not None
    improvement_deadline = search_start + 0.95 * effective_time_limit
    improvement_remaining = max(0.0, improvement_deadline - time.perf_counter())
    gap_after_construction = best.makespan - lower_bound
    mcts_share = (
        0.55 if M >= 5 or gap_after_construction <= 1
        else 0.35
    )
    mcts_deadline = time.perf_counter() + improvement_remaining * mcts_share
    mcts_iterations = 0
    mcts_improvements = 0
    if (
        use_exact_search
        and best.makespan > lower_bound
        and time.perf_counter() < mcts_deadline
    ):
        improved, evaluated = _mcts_horizon_search(
            W, M, starts, eligibility, bay_criticality, best,
            mcts_deadline, seed + 97,
        )
        mcts_iterations += evaluated
        if improved is not None and improved.objective_key < best.objective_key:
            best = improved
            mcts_improvements += 1

    layered_search_states = 0
    layered_search_improvements = 0
    repair_cutoffs = _blocking_repair_cutoffs(best, M)
    for layered_variant in range(4):
        if (
            not use_exact_search
            or best.makespan <= lower_bound
            or time.perf_counter() >= layered_deadline
        ):
            break
        # Give the global pass and each of the three tail repairs its own
        # share. Previously the first pass could consume the complete phase.
        passes_left = 4 - layered_variant
        pass_deadline = time.perf_counter() + (
            layered_deadline - time.perf_counter()
        ) / passes_left
        improved, evaluated = _bounded_layered_search(
            W, M, starts, initial_configs, eligibility,
            bay_criticality, best, pass_deadline, layered_variant,
            None
            if layered_variant == 0
            else repair_cutoffs[(layered_variant - 1) % len(repair_cutoffs)],
        )
        layered_search_states += evaluated
        if improved is not None and improved.objective_key < best.objective_key:
            best = improved
            layered_search_improvements += 1

    exact_search_nodes = 0
    exact_search_improvements = 0
    exact_search_proved_optimal = False
    # Complete enumeration is most useful when the remaining gap is small.
    # For a wide gap on a large state space, spend the budget on diverse
    # improvement passes instead of a proof attempt that cannot finish.
    exact_is_promising = (
        best.makespan - lower_bound <= 1
        and len(configurations) <= 20_000
    )
    while (
        use_exact_search
        and exact_is_promising
        and best.makespan > lower_bound
        and time.perf_counter() < deadline
    ):
        improved, exact_status, evaluated = _exact_horizon_search(
            W, M, starts, configurations, initial_configs, eligibility,
            bay_criticality, best, deadline,
        )
        exact_search_nodes += evaluated
        if improved is not None and improved.objective_key < best.objective_key:
            best = improved
            exact_search_improvements += 1
            continue
        if exact_status == "INFEASIBLE":
            exact_search_proved_optimal = True
        break

    elapsed = time.perf_counter() - search_start
    makespan_optimal = best.makespan == lower_bound or exact_search_proved_optimal
    bay_cranes = {
        i + 1: [q + 1 for q in sorted(bay_owners)]
        for i, bay_owners in enumerate(best.owners)
        if bay_owners
    }
    return Solution(
        status=(
            "MAKESPAN_OPTIMAL_BY_LOWER_BOUND"
            if best.makespan == lower_bound
            else "MAKESPAN_OPTIMAL_BY_EXACT_SEARCH"
            if exact_search_proved_optimal
            else "HEURISTIC_FEASIBLE"
        ),
        method="adaptive_search_plus_partial_mcts_blocking_repair_and_selective_exact_bb_no_solver",
        makespan=best.makespan,
        makespan_lower_bound=lower_bound,
        lower_bound_components=lower_bound_components,
        makespan_proven_optimal=makespan_optimal,
        proven_lexicographic_optimal=False,
        assignment_count=best.assignment_count,
        split_bay_count=best.split_bay_count,
        load_deviation=best.load_deviation,
        reversal_count=best.reversal_count,
        movement_count=best.movement_count,
        crane_loads=best.loads,
        target_weights=target_weights,
        bay_cranes=bay_cranes,
        slots=best.slots,
        restarts_completed=completed,
        strategy_evaluations=strategy_counts,
        layered_search_states=layered_search_states,
        layered_search_improvements=layered_search_improvements,
        mcts_iterations=mcts_iterations,
        mcts_improvements=mcts_improvements,
        exact_search_nodes=exact_search_nodes,
        exact_search_improvements=exact_search_improvements,
        exact_search_proved_optimal=exact_search_proved_optimal,
        search_seconds=round(elapsed, 6),
        max_steps=max_steps,
    )


def verify_solution(W: Sequence[int], M: int, S: Iterable[int], solution: Solution) -> None:
    """Independently verify workload, state, collision, and start constraints."""
    by_time: dict[int, list[Slot]] = {}
    work_done = [0] * len(W)
    load_done = [0] * M
    required = set(S)

    for slot in solution.slots:
        by_time.setdefault(slot.time, []).append(slot)
        if slot.state == "work":
            if slot.work_bay != slot.start_bay or slot.start_bay != slot.end_bay:
                raise AssertionError("作业槽的位置不一致。")
            assert slot.work_bay is not None
            work_done[slot.work_bay - 1] += 1
            load_done[slot.crane - 1] += 1
        elif slot.state == "move":
            if slot.start_bay == slot.end_bay or slot.work_bay is not None:
                raise AssertionError("移动槽定义错误。")
        elif slot.state == "idle":
            if slot.start_bay != slot.end_bay or slot.work_bay is not None:
                raise AssertionError("空闲槽定义错误。")
        else:
            raise AssertionError(f"未知状态：{slot.state}")

    if work_done != list(W):
        raise AssertionError(f"作业量不守恒：得到 {work_done}，要求 {list(W)}。")
    if load_done != solution.crane_loads:
        raise AssertionError("桥吊负荷汇总不一致。")
    if _count_reversals(solution.slots, M) != solution.reversal_count:
        raise AssertionError("桥吊折返次数汇总不一致。")
    if len(by_time) != solution.makespan:
        raise AssertionError("时间槽数量与完工时间不一致。")

    for t in range(solution.makespan):
        rows = sorted(by_time[t], key=lambda row: row.crane)
        if len(rows) != M:
            raise AssertionError(f"t={t} 的桥吊状态数量不是 M。")
        if any(right.start_bay - left.start_bay < 2 for left, right in zip(rows, rows[1:])):
            raise AssertionError(f"t={t} 左边界违反安全间距。")
        if any(right.end_bay - left.end_bay < 2 for left, right in zip(rows, rows[1:])):
            raise AssertionError(f"t={t} 右边界违反安全间距。")
        working_bays = [row.work_bay for row in rows if row.state == "work"]
        if len(working_bays) != len(set(working_bays)):
            raise AssertionError(f"t={t} 同一贝位有多台桥吊作业。")
        if t > 0:
            previous = sorted(by_time[t - 1], key=lambda row: row.crane)
            for before, current in zip(previous, rows):
                if before.end_bay != current.start_bay:
                    raise AssertionError(
                        f"t={t} Q{current.crane} 位置不连续："
                        f"上一槽结束于 {before.end_bay}，本槽开始于 {current.start_bay}。"
                    )

    first_slot_work = {
        row.work_bay for row in by_time.get(0, []) if row.state == "work"
    }
    if not required.issubset(first_slot_work):
        raise AssertionError("强制开工集合 S 未在 t=0 全部开工。")


def plot_schedule(
    solution: Solution,
    N: int,
    output_path: Path,
    show: bool = False,
) -> None:
    """Draw the schedule, save it as PNG, and optionally show a GUI window."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A project-local persistent cache avoids an expensive system-font rescan
    # when the user's default Matplotlib configuration directory is missing or
    # not writable (a common first-run issue in IDE/sandbox environments).
    font_cache = output_path.parent / ".matplotlib-cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(font_cache))
    import logging
    import matplotlib

    # Agg can save figures without a desktop.  When --show is supplied we keep
    # Matplotlib's GUI backend so plt.show() can open a window in PyCharm.
    if not show:
        matplotlib.use("Agg")
    # Several macOS CJK fonts do not advertise a literal "normal" weight.
    # Matplotlib falls back correctly; silence only that font-manager chatter.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"):
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    crane_count = len(solution.crane_loads)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(q % 10) for q in range(crane_count)]
    height = min(30.0, max(6.0, 0.42 * max(1, solution.makespan)))
    fig, ax = plt.subplots(figsize=(max(10.0, 0.65 * N), height))

    for slot in solution.slots:
        color = colors[slot.crane - 1]
        y0 = slot.time
        if slot.state == "work":
            rect = Rectangle(
                (slot.start_bay - 0.36, y0 + 0.06), 0.72, 0.88,
                facecolor=color, edgecolor="none", alpha=0.78,
            )
            ax.add_patch(rect)
            ax.text(slot.start_bay, y0 + 0.52, f"Q{slot.crane}", ha="center", va="center", fontsize=8)
        elif slot.state == "move":
            ax.annotate(
                "",
                xy=(slot.end_bay, y0 + 0.9),
                xytext=(slot.start_bay, y0 + 0.1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8, linestyle="--"),
            )
        else:
            ax.plot(slot.start_bay, y0 + 0.5, marker="o", markersize=5, color=color, fillstyle="none")

    ax.set_xlim(0.5, N + 0.5)
    ax.set_ylim(solution.makespan if solution.makespan else 1, 0)
    ax.set_xticks(range(1, N + 1))
    ax.set_xticklabels([str(i) for i in range(1, N + 1)])
    if solution.makespan <= 30:
        ax.set_yticks(range(solution.makespan + 1))
    else:
        step = max(1, math.ceil(solution.makespan / 20))
        ax.set_yticks(range(0, solution.makespan + 1, step))
    ax.set_xlabel("贝位序号 / Bay index (1-based)")
    ax.set_ylabel("Time")
    proof_text = "makespan optimal" if solution.makespan_proven_optimal else "heuristic"
    ax.set_title(
        f"CWP schedule — {proof_text}, makespan {solution.makespan}, "
        f"reversals {solution.reversal_count}, moves {solution.movement_count}, "
        f"split bays {solution.split_bay_count}"
    )
    ax.grid(True, which="major", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)

    crane_legend = [
        Line2D([0], [0], color=colors[q], lw=5, label=f"Q{q + 1} load={solution.crane_loads[q]}")
        for q in range(crane_count)
    ]
    state_legend = [
        Patch(facecolor="0.55", alpha=0.78, label="work: filled cell"),
        Line2D([0], [0], color="0.35", lw=1.8, linestyle="--", marker=">", label="move: dashed arrow"),
        Line2D([0], [0], color="0.35", marker="o", fillstyle="none", linestyle="None", label="idle: hollow dot"),
    ]
    first = ax.legend(handles=crane_legend, title="Cranes", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(first)
    ax.legend(handles=state_legend, title="States", loc="lower left", bbox_to_anchor=(1.01, 0.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_worker(solution: Solution, N: int, output_path: Path) -> None:
    """Process entry point for time-bounded PNG generation."""
    plot_schedule(solution, N, output_path, show=False)


def _show_saved_schedule(output_path: Path) -> None:
    """Display an already generated PNG; waiting for the user is outside runtime."""
    import matplotlib.pyplot as plt

    image = plt.imread(output_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _print_summary(solution: Solution) -> None:
    if solution.status == "MAKESPAN_OPTIMAL_BY_LOWER_BOUND":
        proof = "达到理论下界，完工时间已证明最优"
    elif solution.status == "MAKESPAN_OPTIMAL_BY_EXACT_SEARCH":
        proof = "精确分支定界已排除所有更短工期，完工时间已证明最优"
    else:
        proof = "当前最好可行解，未证明全局最优"
    print(f"状态: {solution.status}（{proof}）")
    print(f"方法: {solution.method}")
    print(f"完工时间: {solution.makespan}；理论下界: {solution.makespan_lower_bound}")
    print(f"下界组成: {solution.lower_bound_components}")
    print(f"桥吊-贝位分配数: {solution.assignment_count}")
    print(f"被多吊拆分的贝位数: {solution.split_bay_count}")
    print(f"各桥吊作业负荷: {solution.crane_loads}")
    print(f"中间重载目标权重: {solution.target_weights}")
    print(f"折返次数: {solution.reversal_count}")
    print(f"移动次数: {solution.movement_count}")
    print(f"完成搜索轮数: {solution.restarts_completed}；耗时: {solution.search_seconds:.3f}s")
    print(f"各优先规则评估次数: {solution.strategy_evaluations}")
    print(
        f"分层搜索评估状态数: {solution.layered_search_states}；"
        f"工期改进次数: {solution.layered_search_improvements}"
    )
    print(
        f"部分排程 MCTS 迭代数: {solution.mcts_iterations}；"
        f"工期改进次数: {solution.mcts_improvements}"
    )
    print(
        f"精确搜索节点数: {solution.exact_search_nodes}；"
        f"工期改进次数: {solution.exact_search_improvements}；"
        f"是否由精确搜索证明: {solution.exact_search_proved_optimal}"
    )


def main() -> None:
    program_start = time.perf_counter()
    overall_deadline = program_start + 300.0
    parser = argparse.ArgumentParser(description="Solve the discrete one-rail CWP without an optimization solver.")
    parser.add_argument(
        "input", nargs="?", type=Path, default=Path("example_input.json"),
        help="JSON file containing W, M, and S (default: example_input.json)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--restarts", type=int, default=100_000, help="Maximum number of custom-search restarts")
    parser.add_argument("--time-limit", type=float, default=270.0, help="Search time limit in seconds; capped at 270")
    parser.add_argument("--seed", type=int, default=20260910, help="Random seed for reproducible schedules")
    parser.add_argument("--patience", type=int, default=2_000, help="Stop after this many non-improving trials once the makespan lower bound is reached")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the figure window (default); use --no-show for headless runs",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    W, M, S = payload["W"], payload["M"], payload.get("S", [])
    # Reserve 30 seconds for validation, JSON output and bounded PNG creation.
    solve_budget = min(
        args.time_limit,
        max(1.0, overall_deadline - time.perf_counter() - 30.0),
    )
    solution = solve_cwp(
        W, M, S,
        restarts=args.restarts,
        time_limit=solve_budget,
        seed=args.seed,
        patience=args.patience,
    )
    verify_solution(W, M, S, solution)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "schedule.json"
    png_path = args.output_dir / "schedule.png"
    json_path.write_text(json.dumps(solution.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    plot_timeout = max(0.0, min(25.0, overall_deadline - time.perf_counter()))
    plot_completed = False
    if plot_timeout > 0:
        context = multiprocessing.get_context("spawn")
        plot_process = context.Process(
            target=_plot_worker,
            args=(solution, len(W), png_path),
        )
        plot_process.start()
        plot_process.join(plot_timeout)
        if plot_process.is_alive():
            plot_process.terminate()
            plot_process.join(2.0)
        plot_completed = plot_process.exitcode == 0 and png_path.exists()
    _print_summary(solution)
    print(f"调度明细: {json_path.resolve()}")
    if plot_completed:
        print(f"甘特图: {png_path.resolve()}")
    else:
        print("甘特图未在总时限内生成；调度 JSON 已正常保存。")
    print(f"程序计算耗时: {time.perf_counter() - program_start:.3f}s")
    if args.show and plot_completed:
        _show_saved_schedule(png_path)


if __name__ == "__main__":
    main()
