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
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class Slot:
    time: int
    crane: int
    state: str
    start_bay: int | float
    end_bay: int | float
    work_bay: int | None
    move_id: int | None = None
    move_step: int | None = None
    move_steps: int | None = None


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
    trajectory_repair_iterations: int = 0
    trajectory_repair_improvements: int = 0
    elite_pool_size: int = 0
    elite_repairs: int = 0
    critical_repair_iterations: int = 0
    critical_repair_improvements: int = 0
    phase_seconds: dict[str, float] = field(default_factory=dict)
    operator_calls: dict[str, int] = field(default_factory=dict)
    move_time: int = 1

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
    move_time: int = 1

    @property
    def objective_key(self) -> tuple[int, int, int, int]:
        """User priorities: duration, single-crane bays, central load, moves."""
        return (
            self.makespan,
            self.split_bay_count,
            self.load_deviation,
            self.movement_count,
        )


@dataclass(frozen=True)
class RepairWindow:
    """A bounded, explicit repair contract.

    ``start`` and ``end`` are position-row indices in the target trajectory;
    transitions in ``[start, end)`` may be rebuilt.  Rows outside this range
    and cranes outside ``active_cranes`` are copied from the source schedule.
    Keeping this contract as data prevents the caller and a repair operator
    from silently choosing different random windows.
    """

    start: int
    end: int
    active_cranes: tuple[int, ...]

    def normalized(self, horizon: int, crane_count: int) -> "RepairWindow | None":
        active = tuple(sorted({q for q in self.active_cranes if 0 <= q < crane_count}))
        start = max(1, min(horizon - 1, int(self.start)))
        end = max(start + 1, min(horizon, int(self.end)))
        if not active or end <= start:
            return None
        return RepairWindow(start, end, active)


@dataclass
class RepairState:
    """Serializable-in-memory continuation point for trajectory repair."""

    context_key: tuple[Any, ...]
    paths: list[list[int]] | None
    counts: list[int] | None
    loss: float
    best_paths: list[list[int]] | None
    best_loss: float
    iterations: int
    rng_state: object


def _candidate_position_rows(
    candidate: _CandidateSchedule, M: int,
) -> list[list[int]]:
    """Recover one position row per zero-time work boundary."""
    rows = [[0] * M for _ in range(candidate.makespan + 1)]
    for slot in candidate.slots:
        rows[slot.time][slot.crane - 1] = int(slot.start_bay)
        rows[slot.time + 1][slot.crane - 1] = int(slot.end_bay)
    if any(any(position == 0 for position in row) for row in rows):
        raise ValueError("排程缺少完整的桥吊位置轨迹。")
    return rows


def _shortening_potential(
    W: Sequence[int],
    M: int,
    candidate: _CandidateSchedule,
) -> tuple[tuple[int, int, int, int, int, int, int], int | None, tuple[int, ...]]:
    """Score how close a complete zero-time schedule is to losing one row.

    The first components are deliberately not the user objective.  They are
    an intermediate search signal: missing work after the best single-row
    deletion plus cranes whose current load cannot fit in ``H-1``.  A local
    preparation may temporarily worsen moves or split bays when it makes a
    later legal shortening more likely.
    """
    if candidate.move_time != 0 or candidate.makespan < 2:
        fallback = (10**9, 10**9, 10**9, 10**9, 10**9,
                    candidate.split_bay_count, candidate.movement_count)
        return fallback, None, tuple(W)
    rows = _candidate_position_rows(candidate, M)
    best: tuple[tuple[int, int, int, int, int, int, int], int, tuple[int, ...]] | None = None
    for remove_at in range(2, len(rows)):
        shortened = rows[:remove_at] + rows[remove_at + 1:]
        capacity = [0] * len(W)
        for row in shortened[:-1]:
            for bay in row:
                if 1 <= bay <= len(W):
                    capacity[bay - 1] += 1
        deficits = tuple(max(0, W[i] - capacity[i]) for i in range(len(W)))
        total_deficit = sum(deficits)
        overload = sum(
            max(0, load - (candidate.makespan - 1))
            for load in candidate.loads
        )
        deficit_difficulty = sum(
            amount * (
                M + 1 - sum(
                    1
                    for q in range(M)
                    if 1 + 2 * q <= bay <= len(W) - 2 * (M - q - 1)
                )
            )
            for bay, amount in enumerate(deficits, 1)
        )
        score = (
            total_deficit + overload,
            total_deficit,
            overload,
            deficit_difficulty,
            sum(value > 0 for value in deficits),
            candidate.split_bay_count,
            candidate.movement_count,
        )
        item = (score, remove_at, deficits)
        if best is None or item < best:
            best = item
    if best is None:
        fallback = (10**9, 10**9, 10**9, 10**9, 10**9,
                    candidate.split_bay_count, candidate.movement_count)
        return fallback, None, tuple(W)
    return best


def _decode_best_shortening(
    W: Sequence[int], M: int, candidate: _CandidateSchedule,
) -> _CandidateSchedule | None:
    """Decode the best prepared one-row deletion when it has zero deficit."""
    score, remove_at, _ = _shortening_potential(W, M, candidate)
    if remove_at is None or score[1] != 0:
        return None
    rows = _candidate_position_rows(candidate, M)
    shortened = rows[:remove_at] + rows[remove_at + 1:]
    try:
        result = _candidate_from_history(W, M, shortened, candidate.move_time)
    except RuntimeError:
        return None
    return result if result.makespan < candidate.makespan else None


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


def apply_completed_edge_exits(
    slots: Sequence[Slot],
    M: int,
    N: int,
    makespan: int,
    move_time: int,
    force_exit: bool = False,
) -> list[Slot]:
    """Legalize completed edge cranes without needlessly losing capacity.

    With zero-time relocation, a completed prefix/suffix may move outward to
    restore the two-bay safety distance.  By default it remains on the rail
    whenever space exists, so Step 8 may assign it new work later.  A crane is
    put ``offrail`` only when no legal on-rail position remains.  ``force_exit``
    retains the old eager-exit behaviour for explicit ablation tests.
    """
    if move_time != 0:
        raise ValueError("自动边界退出目前只支持 move_time=0。")
    by_time: dict[int, list[Slot]] = {}
    last_work = [-1] * M
    for slot in slots:
        by_time.setdefault(slot.time, []).append(slot)
        if slot.state == "work":
            last_work[slot.crane - 1] = max(last_work[slot.crane - 1], slot.time)
    if any(len(by_time.get(t, ())) != M for t in range(makespan)):
        raise ValueError("自动边界退出要求每个时间槽都包含 M 条桥吊记录。")

    result: list[Slot] = []
    for t in range(makespan):
        rows = sorted(by_time[t], key=lambda row: row.crane)
        left_count = 0
        while left_count < M and last_work[left_count] < t:
            left_count += 1
        right_first = M
        while right_first > left_count and last_work[right_first - 1] < t:
            right_first -= 1
        positions = [int(slot.start_bay) for slot in rows]
        if not force_exit:
            # Keep the still-working middle fixed.  Completed edge cranes are
            # shifted only as far outward as necessary.  This turns "may
            # leave" into a capacity-preserving option instead of treating a
            # source schedule's last work time as an irrevocable exit time.
            if left_count:
                right_limit = positions[left_count] - 2 if left_count < M else N
                for q in range(left_count - 1, -1, -1):
                    positions[q] = min(positions[q], right_limit)
                    right_limit = positions[q] - 2
            if right_first < M:
                left_limit = positions[right_first - 1] + 2 if right_first else 1
                for q in range(right_first, M):
                    positions[q] = max(positions[q], left_limit)
                    left_limit = positions[q] + 2
        for q, slot in enumerate(rows):
            if q < left_count and (force_exit or positions[q] < 1):
                position = 1 - 2 * (left_count - q)
                result.append(Slot(t, q + 1, "offrail", position, position, None))
            elif q >= right_first and (force_exit or positions[q] > N):
                position = N + 2 * (q - right_first + 1)
                result.append(Slot(t, q + 1, "offrail", position, position, None))
            elif q < left_count or q >= right_first:
                position = positions[q]
                result.append(Slot(t, q + 1, "idle", position, position, None))
            else:
                result.append(slot)
    return result


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


def _weighted_initial_positions(weights, M, starts):
    """Maximum-weight size-M independent set on a path, containing S.

    dp[b][k] selects k separated positions from bays 1..b. Required bays
    cannot be skipped. No joint crane configurations are enumerated.
    """
    required = set(starts)
    n = len(weights)
    dp = [[None] * (M + 1) for _ in range(n + 1)]
    dp[0][0] = (0.0, ())
    for b in range(1, n + 1):
        for k in range(M + 1):
            best = dp[b - 1][k] if b not in required else None
            if k and b - 1 not in required:
                previous = dp[max(0, b - 2)][k - 1]
                if previous is not None:
                    candidate = (previous[0] + weights[b - 1], previous[1] + (b,))
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            dp[b][k] = best
    return dp[n][M]


def _validate_input(
    W: list[int], M: int, S: Iterable[int], move_time: int = 1,
) -> tuple[int, list[int], list[tuple[int, ...]]]:
    if not isinstance(W, list) or not W:
        raise ValueError("W 必须是非空整数列表。")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in W):
        raise ValueError("W 中每个作业量必须是非负整数。")
    if isinstance(M, bool) or not isinstance(M, int) or M <= 0:
        raise ValueError("M 必须是正整数。")
    if isinstance(move_time, bool) or not isinstance(move_time, int) or move_time < 0:
        raise ValueError("move_time 必须是非负整数。")

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

    if math.comb(N - M + 1, M) <= 20_000:
        configurations = _legal_configurations(N, M)
    else:
        # A bounded pool is used ONLY for heuristic initial seeds. Lower
        # bounds optimize initial coverage independently over the full domain.
        pool = set()
        sample_rng = random.Random(193)
        for trial in range(128):
            weights = [min(w, 8) if trial == 0 else sample_rng.random()
                       * (1 + min(w, 8) * (trial % 3 == 0)) for w in W]
            result = _weighted_initial_positions(weights, M, starts)
            if result is None:
                break
            pool.add(result[1])
        configurations = sorted(pool)
    required = set(starts)
    if not any(required.issubset(config) for config in configurations):
        raise ValueError("S 无法扩充为 M 台桥吊的合法初始停位组合。")

    coverable = {bay for q in range(M)
                 for bay in range(1 + 2 * q, N - 2 * (M - q - 1) + 1)}
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
    starts: Sequence[int] | None = None,
    move_time: int = 1,
) -> tuple[int, dict[str, int]]:
    """Return valid workload, movement, and congestion-window lower bounds."""
    total_work = sum(W)
    positive_bays = sum(amount > 0 for amount in W)
    def coverage(bays):
        if starts is not None:
            result = _weighted_initial_positions(
                [int(b + 1 in bays) for b in range(len(W))], M, starts)
            return int(result[0])
        return max((sum(b in bays for b in config) for config in initial_configs), default=0)

    initial_positive_capacity = coverage({i + 1 for i, w in enumerate(W) if w})
    minimum_crane_moves = max(0, positive_bays - initial_positive_capacity)
    workload_bound = math.ceil(total_work / M)
    active_time_bound = math.ceil(
        (total_work + move_time * minimum_crane_moves) / M
    )

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
            initial_window_coverage = coverage(positive_in_window)
            minimum_window_arrivals = max(
                0, len(positive_in_window) - initial_window_coverage
            )
            window_active_bound = max(
                window_active_bound,
                math.ceil(
                    (window_work + move_time * minimum_window_arrivals)
                    / parallel_capacity
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
            initial_coverage = coverage(mandatory_set)
            minimum_subset_moves = max(0, len(mandatory_bays) - initial_coverage)
            mandatory_work = sum(W[bay_index] for bay_index in mandatory_bays)
            eligibility_bound = max(
                eligibility_bound,
                math.ceil(
                    (mandatory_work + move_time * minimum_subset_moves)
                    / crane_count
                ),
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


def _construct_schedule_legacy(
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


def _best_safe_dispatch(cells):
    """Exact additive dispatch DP; flags idle=0, ready=1, working=2.

    Prefix maxima impose gap >= 2 in O(M N) states. This optimizes
    one dispatch score, not the complete scheduling objective.
    """
    previous = None
    for row in cells:
        current = [[None] * 3 for _ in row]
        prefix = [None] * 3
        for i, cell in enumerate(row):
            if previous is not None and i >= 2:
                for flag, item in enumerate(previous[i - 2]):
                    if item is not None and (prefix[flag] is None or item[0] > prefix[flag][0]):
                        prefix[flag] = item
            if cell is None:
                continue
            score, flag = cell
            if previous is None:
                current[i][flag] = (score, (i + 1,))
                continue
            for old_flag, item in enumerate(prefix):
                if item is None:
                    continue
                new_flag = max(flag, old_flag)
                proposal = (item[0] + score, item[1] + (i + 1,))
                incumbent = current[i][new_flag]
                if incumbent is None or proposal[0] > incumbent[0]:
                    current[i][new_flag] = proposal
        previous = current
    normal = progress = None
    for row in previous or []:
        for flag in (1, 2):
            item = row[flag]
            if item is not None:
                if normal is None or item[0] > normal[0]:
                    normal = item
                if flag == 2 and (progress is None or item[0] > progress[0]):
                    progress = item
    return normal, progress


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

        cells = []
        for q in range(M):
            row = []
            for bay in range(1, len(W) + 1):
                i = bay - 1
                if q not in eligibility[i] or (
                    t == 0 and q in mandatory_cranes and bay != positions[q]
                ):
                    row.append(None)
                    continue
                allowed = (
                    (fixed_owner is None or fixed_owner[i] in (None, q))
                    and (not strategy.strict_owner or not owners[i] or q in owners[i])
                )
                work = bay == positions[q] and remaining[i] > 0 and allowed
                left = remaining[i] - int(work)
                ready = left > 0 and allowed
                move = bay != positions[q]
                direction = 1 if bay > positions[q] else -1
                reversal = move and last_move_direction[q] not in (0, direction)
                hint = (
                    (int(work) + 0.6 * int(ready))
                    if preferred_owner is not None and preferred_owner[i] == q else 0.0
                )
                score = (
                    strategy.work_weight * work
                    + strategy.ready_weight * ready
                    + 0.12 * min(left, 6) * ready
                    + strategy.balance_weight * max(0.0, target_loads[q] - loads[q]) * ready
                    + strategy.priority_weight * bay_criticality[i] * (int(work) + int(ready))
                    + 1.25 * hint
                    - strategy.split_penalty * bool(work and owners[i] and q not in owners[i])
                    - strategy.move_penalty * move
                    - strategy.reversal_penalty * reversal
                    + strategy.noise * rng.random() / M
                )
                row.append((score, 2 if work else 1 if ready else 0))
            cells.append(row)
        normal, progress = _best_safe_dispatch(cells)
        for result, is_progress in ((normal, False), (progress, True)):
            if result is None:
                continue
            score, next_positions = result
            working = tuple(
                q for q, bay in enumerate(next_positions)
                if cells[q][bay - 1][1] == 2
            )
            item = (score, next_positions, working)
            if is_progress:
                best_progress = item
            else:
                best_normal = item
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


def _trajectory_repair(
    W,
    M,
    starts,
    incumbent,
    deadline,
    seed,
    active_cranes: Sequence[int] | None = None,
    window: tuple[int, int] | None = None,
    repair_state: RepairState | None = None,
    return_state: bool = False,
    attempt_trace: list[dict[str, Any]] | None = None,
    move_time: int = 1,
    preserve_horizon: bool = False,
    preparation_mode: bool = False,
):
    """Annealed interval repair of a shortened complete position trajectory.

    Infeasible states may lack work, but every rail position remains safe.
    Capacity counts stationary edges; excess capacity decodes as idle.
    Only a zero-deficit trajectory is returned. No optimality claim.
    """
    horizon = incumbent.makespan if preserve_horizon else incumbent.makespan - 1
    if horizon < 1 or horizon > 2000:
        result = (None, 0, None) if return_state else (None, 0)
        return result
    rng = random.Random(seed)
    rows = [[0] * M for _ in range(incumbent.makespan + 1)]
    for slot in incumbent.slots:
        rows[slot.time][slot.crane - 1] = slot.start_bay
        rows[slot.time + 1][slot.crane - 1] = slot.end_bay
    required = set(starts)
    pinned = {q for q, bay in enumerate(rows[0]) if bay in required}
    active = set(range(M)) if active_cranes is None else set(active_cranes)
    active &= set(range(M))
    if not active:
        return (None, 0, None) if return_state else (None, 0)
    window_start, window_end = window or (0, horizon)
    window_start = max(0, min(horizon, window_start))
    window_end = max(window_start, min(horizon, window_end))
    # An exited crane is no longer part of a local on-rail neighbourhood.
    # Keep its virtual trajectory frozen and repair only cranes that remain
    # physically on the rail throughout this window.
    active = {
        q for q in active
        if all(1 <= rows[t][q] <= len(W) for t in range(window_start, window_end + 1))
    }
    if not active:
        return (None, 0, None) if return_state else (None, 0)
    if preparation_mode and (not preserve_horizon or move_time != 0):
        raise ValueError("局部准备阶段要求 preserve_horizon=True 且 move_time=0。")
    preparation_reference, _, preparation_deficits = _shortening_potential(
        W, M, incumbent
    ) if preparation_mode else ((0, 0, 0, 0, 0, 0, 0), None, tuple())
    preparation_targets = {
        bay for bay, deficit in enumerate(preparation_deficits, 1) if deficit > 0
    }
    context_key = (
        tuple(W), M, horizon, move_time, preserve_horizon, preparation_mode,
        tuple(sorted(active)), window_start, window_end,
        hash(tuple(tuple(row) for row in rows)),
    )
    continuation = repair_state if (
        repair_state is not None and repair_state.context_key == context_key
    ) else None
    if continuation is not None:
        rng.setstate(continuation.rng_state)
    best_paths = (
        [path[:] for path in continuation.best_paths]
        if continuation is not None and continuation.best_paths is not None else None
    )
    best_loss = continuation.best_loss if continuation is not None else float('inf')
    iterations = continuation.iterations if continuation is not None else 0
    continued_paths = (
        [path[:] for path in continuation.paths]
        if continuation is not None and continuation.paths is not None else None
    )
    continued_counts = (
        list(continuation.counts)
        if continuation is not None and continuation.counts is not None else None
    )
    continued_loss = continuation.loss if continuation is not None else float('inf')

    def result(candidate, saved_state=None):
        if return_state:
            return candidate, iterations, saved_state
        return candidate, iterations

    def save_state(paths, counts, loss):
        return RepairState(
            context_key=context_key,
            paths=[path[:] for path in paths] if paths is not None else None,
            counts=list(counts) if counts is not None else None,
            loss=float(loss),
            best_paths=[path[:] for path in best_paths] if best_paths is not None else None,
            best_loss=float(best_loss),
            iterations=iterations,
            rng_state=rng.getstate(),
        )

    def capacity_counts(paths_to_count):
        counts_to_return = [0] * len(W)
        for path in paths_to_count:
            if move_time == 0:
                for bay in path[:-1]:
                    if 1 <= bay <= len(W):
                        counts_to_return[int(bay) - 1] += 1
            else:
                for left, right in zip(path, path[1:]):
                    if left == right and 1 <= left <= len(W):
                        counts_to_return[int(left) - 1] += 1
        return counts_to_return
    while time.perf_counter() < deadline:
        # Delete a boundary, not a job. The resulting missing work is measured
        # explicitly. Keep both endpoints of the mandatory first work slot.
        if continued_paths is not None:
            paths = [path[:] for path in continued_paths]
            counts = list(continued_counts)
            loss = continued_loss
            continued_paths = None
            continued_counts = None
        elif best_paths is not None and rng.random() < 0.75:
            paths = [path[:] for path in best_paths]
            counts = capacity_counts(paths)
            loss = sum(
                max(0, W[i] - count) + 0.015 * max(0, W[i] - count) ** 2
                for i, count in enumerate(counts)
            )
        else:
            # When a local repair is requested, the deleted boundary must be
            # inside that same window.  Deleting an unrelated row creates a
            # work deficit that the supposedly frozen neighbourhood cannot
            # repair.  General repair keeps the original broad range.
            if preserve_horizon:
                removed = None
                shortened = rows[:]
            else:
                remove_low = 2
                remove_high = len(rows) - 1
                if window is not None:
                    remove_low = max(remove_low, int(window[0]) + 1)
                    remove_high = min(remove_high, int(window[1]))
                if remove_low > remove_high:
                    return result(None, None)
                removed = rng.randrange(remove_low, remove_high + 1)
            if attempt_trace is not None:
                attempt_trace.append({"remove_at": removed, "window": (window_start, window_end)})
            shortened = rows if preserve_horizon else rows[:removed] + rows[removed + 1:]
            paths = [[row[q] for row in shortened] for q in range(M)]
            counts = capacity_counts(paths)
            loss = sum(
                max(0, W[i] - count) + 0.015 * max(0, W[i] - count) ** 2
                for i, count in enumerate(counts)
            )

        def penalty(i, count):
            deficit = max(0, W[i] - count)
            return deficit + 0.015 * deficit * deficit

        for attempt in range(4000):
            iterations += 1
            if iterations % 128 == 0 and time.perf_counter() >= deadline:
                return result(None, save_state(paths, counts, loss))
            if loss < 1e-8:
                history = list(zip(*paths))
                candidate = _candidate_from_history(W, M, history, move_time)
                if preparation_mode:
                    candidate_potential, _, _ = _shortening_potential(W, M, candidate)
                    if (
                        candidate.makespan == incumbent.makespan
                        and candidate_potential[:5] < preparation_reference[:5]
                    ):
                        return result(candidate, None)
                elif (
                    not preserve_horizon
                    or candidate.makespan == incumbent.makespan
                    and candidate.objective_key < incumbent.objective_key
                ):
                    return result(candidate, None)
            if loss < best_loss - 1e-8:
                best_loss = loss
                best_paths = [path[:] for path in paths]
            q = rng.choice(tuple(active))
            path = paths[q]
            if q in pinned and horizon < 2:
                continue
            lower_a = max(window_start, 2 if q in pinned else 0)
            if lower_a > window_end:
                continue
            a = rng.randrange(lower_a, window_end + 1)
            if rng.random() < 0.65:
                # Shift an existing work block boundary, including the case
                # where two blocks must merge to remove a movement slot.
                b = a
                while b < horizon and path[b + 1] == path[a]:
                    b += 1
                if rng.random() < 0.5:
                    b = min(b, a + rng.randrange(1, 4))
            else:
                b = min(horizon, a + rng.randrange(1, max(2, horizon // 3)))
            b = min(b, window_end)
            targeted = preparation_mode and preparation_targets and rng.random() < 0.70
            coordinated = targeted or (
                M >= 4 and len(active) >= 2 and rng.random() < 0.25
            )
            low = 1 + 2 * q if coordinated else (max(paths[q - 1][a:b + 1]) + 2 if q else 1)
            high = len(W) - 2 * (M - q - 1) if coordinated else (min(paths[q + 1][a:b + 1]) - 2 if q + 1 < M else len(W))
            if low > high:
                continue
            target_choices = [
                bay for bay in sorted(preparation_targets)
                if low <= bay <= high
                and not all(position == bay for position in path[a:b + 1])
            ]
            choices = []
            if a:
                choices.append(path[a - 1])
            if b < horizon:
                choices.append(path[b + 1])
            choices.extend(target_choices)
            choices.extend(i + 1 for i in range(low - 1, high) if counts[i] < W[i])
            choices.append(rng.randint(low, high))
            bay = rng.choice(target_choices if targeted and target_choices else choices)
            if not low <= bay <= high or all(p == bay for p in path[a:b + 1]):
                continue
            replacements = {q: [bay] * (b - a + 1)}
            target_left = targeted and bay < min(path[a:b + 1])
            target_right = targeted and bay > max(path[a:b + 1])
            if target_left:
                # Insert the missing bay and relay each vacated trajectory to
                # the next active crane on the right.  This preserves useful
                # capacity through a Qk -> Qk+1 hand-off instead of merely
                # pushing neighbours away from a randomly moved crane.
                previous = paths[q][a:b + 1]
                for other in range(q + 1, M):
                    if other not in active:
                        break
                    replacements[other] = previous
                    previous = paths[other][a:b + 1]
            elif target_right:
                following = paths[q][a:b + 1]
                for other in range(q - 1, -1, -1):
                    if other not in active:
                        break
                    replacements[other] = following
                    following = paths[other][a:b + 1]
            elif coordinated:
                # Propagate only the minimum necessary displacement to
                # neighboring cranes. Every boundary remains safe; this
                # permits escaping blocks that no single crane can leave.
                for other in range(q + 1, M):
                    if other not in active:
                        continue
                    previous = replacements.get(other - 1, paths[other - 1][a:b + 1])
                    replacements[other] = [max(p, left + 2) for p, left in
                                           zip(paths[other][a:b + 1], previous)]
                for other in range(q - 1, -1, -1):
                    if other not in active:
                        continue
                    following = replacements.get(other + 1, paths[other + 1][a:b + 1])
                    replacements[other] = [min(p, right - 2) for p, right in
                                           zip(paths[other][a:b + 1], following)]
            if any(p < 1 or p > len(W) for row in replacements.values() for p in row):
                continue
            if a < 2 and any(
                k in replacements
                and replacements[k][:2-a] != paths[k][a:min(b+1, 2)]
                for k in pinned
            ):
                continue
            for t in range(a, b + 1):
                row = [
                    replacements.get(k, paths[k][t])[t - a]
                    if k in replacements else paths[k][t]
                    for k in range(M)
                ]
                if any(right - left < 2 for left, right in zip(row, row[1:])):
                    replacements = {}
                    break
            if not replacements:
                continue
            delta = {}
            proposed_counts = None
            if move_time == 0:
                proposed_paths = [path[:] for path in paths]
                for k, replacement in replacements.items():
                    proposed_paths[k][a:b + 1] = replacement
                proposed_counts = capacity_counts(proposed_paths)
                delta = {
                    i: proposed_counts[i] - counts[i]
                    for i in range(len(W))
                    if proposed_counts[i] != counts[i]
                }
            for k, replacement in replacements.items():
                if move_time == 0:
                    continue
                trajectory = paths[k]
                for t in range(max(0, a - 1), min(horizon, b + 1)):
                    old_a, old_b = trajectory[t], trajectory[t + 1]
                    new_a = replacement[t-a] if a <= t <= b else old_a
                    new_b = replacement[t+1-a] if a <= t + 1 <= b else old_b
                    if old_a == old_b and 1 <= old_a <= len(W):
                        index = int(old_a) - 1
                        delta[index] = delta.get(index, 0) - 1
                    if new_a == new_b and 1 <= new_a <= len(W):
                        index = int(new_a) - 1
                        delta[index] = delta.get(index, 0) + 1
            change = sum(penalty(i, counts[i] + d) - penalty(i, counts[i]) for i, d in delta.items())
            temperature = 0.08 + 0.65 * (1.0 - attempt / 4000) ** 2
            if change <= 0 or rng.random() < math.exp(-change / temperature):
                for k, replacement in replacements.items():
                    paths[k][a:b + 1] = replacement
                if proposed_counts is not None:
                    counts = proposed_counts
                else:
                    for i, d in delta.items():
                        counts[i] += d
                loss += change
    saved = save_state(paths, counts, loss) if "paths" in locals() else None
    return result(None, saved)


def _candidate_from_history(
    W: Sequence[int],
    M: int,
    history: Sequence[tuple[int, ...]],
    move_time: int = 1,
) -> _CandidateSchedule:
    """Decode a configuration path using the configured relocation duration.

    ``move_time == 0`` means a relocation occurs at the boundary between two
    work periods.  It therefore creates no ``move`` slot.  Positive durations
    are expanded into that many unit slots; positions during a simultaneous
    move are linearly interpolated, which preserves crane order and the
    two-bay separation whenever both endpoint configurations are safe.
    """
    if isinstance(move_time, bool) or not isinstance(move_time, int) or move_time < 0:
        raise ValueError("move_time 必须是非负整数。")
    remaining = list(W)
    owners: list[set[int]] = [set() for _ in W]
    loads = [0] * M
    slots: list[Slot] = []
    movement_count = 0
    movement_directions: list[list[int]] = [[] for _ in range(M)]
    output_time = 0
    move_id = 0
    for positions, next_positions in zip(history, history[1:]):
        moving = [a != b for a, b in zip(positions, next_positions)]
        movement_count += sum(moving)
        for q, (a, b) in enumerate(zip(positions, next_positions)):
            if a != b:
                movement_directions[q].append(1 if b > a else -1)

        if move_time == 0:
            # Work is performed at the current configuration; relocation to
            # next_positions then happens instantaneously at the period edge.
            work_here = [
                1 <= bay <= len(W) and remaining[int(bay) - 1] > 0
                for bay in positions
            ]
            if not any(work_here):
                continue
            for q, bay in enumerate(positions):
                if work_here[q]:
                    bay_index = int(bay) - 1
                    remaining[bay_index] -= 1
                    owners[bay_index].add(q)
                    loads[q] += 1
                    slots.append(Slot(output_time, q + 1, "work", bay, bay, int(bay)))
                elif not 1 <= bay <= len(W):
                    slots.append(Slot(output_time, q + 1, "offrail", bay, bay, None))
                else:
                    slots.append(Slot(output_time, q + 1, "idle", bay, bay, None))
            output_time += 1
            continue

        duration = move_time if any(moving) else 1
        current_move_id = move_id if any(moving) and move_time > 1 else None
        if current_move_id is not None:
            move_id += 1
        for step in range(duration):
            for q, (start_bay, end_bay) in enumerate(zip(positions, next_positions)):
                if moving[q]:
                    left = start_bay + (end_bay - start_bay) * step / duration
                    right = start_bay + (end_bay - start_bay) * (step + 1) / duration
                    slots.append(Slot(
                        output_time, q + 1, "move", left, right, None,
                        current_move_id, step + 1 if current_move_id is not None else None,
                        duration if current_move_id is not None else None,
                    ))
                else:
                    bay_index = start_bay - 1
                    if step == 0 and remaining[bay_index] > 0:
                        remaining[bay_index] -= 1
                        owners[bay_index].add(q)
                        loads[q] += 1
                        slots.append(Slot(
                            output_time, q + 1, "work",
                            start_bay, start_bay, start_bay,
                        ))
                    else:
                        slots.append(Slot(
                            output_time, q + 1, "idle",
                            start_bay, start_bay, None,
                        ))
            output_time += 1
    if any(remaining):
        raise RuntimeError("精确搜索路径没有完成全部作业。")
    target_weights = [min(q + 1, M - q) for q in range(M)]
    if move_time == 0:
        visible = [
            tuple(
                int(slot.start_bay)
                for slot in sorted(
                    (item for item in slots if item.time == t),
                    key=lambda item: item.crane,
                )
            )
            for t in range(output_time)
        ]
        visible_directions: list[list[int]] = [[] for _ in range(M)]
        movement_count = 0
        for before, after in zip(visible, visible[1:]):
            for q, (a, b) in enumerate(zip(before, after)):
                if a != b:
                    movement_count += 1
                    visible_directions[q].append(1 if b > a else -1)
        reversal_count = sum(
            previous != current
            for directions in visible_directions
            for previous, current in zip(directions, directions[1:])
        )
    else:
        reversal_count = sum(
            previous != current
            for directions in movement_directions
            for previous, current in zip(directions, directions[1:])
        )
    return _CandidateSchedule(
        slots=slots,
        makespan=output_time,
        assignment_count=sum(len(item) for item in owners),
        split_bay_count=sum(len(item) > 1 for item in owners),
        load_deviation=_load_deviation(loads, target_weights, sum(W)),
        reversal_count=reversal_count,
        movement_count=movement_count,
        loads=loads,
        owners=owners,
        move_time=move_time,
    )


def _unit_history(candidate: _CandidateSchedule, M: int) -> list[tuple[int, ...]]:
    """Recover the old unit-transition path from a legacy/core candidate."""
    rows = [[0] * M for _ in range(candidate.makespan + 1)]
    for slot in candidate.slots:
        rows[slot.time][slot.crane - 1] = int(slot.start_bay)
        rows[slot.time + 1][slot.crane - 1] = int(slot.end_bay)
    if any(any(bay == 0 for bay in row) for row in rows):
        raise ValueError("无法从非单位移动排程恢复核心轨迹。")
    return [tuple(row) for row in rows]


def _retime_candidate(
    W: Sequence[int], M: int, candidate: _CandidateSchedule, move_time: int,
) -> _CandidateSchedule:
    """Apply movement timing to a candidate produced by the legacy core."""
    if move_time == candidate.move_time:
        return candidate
    return _candidate_from_history(W, M, _unit_history(candidate, M), move_time)


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


def _critical_repair_windows(
    incumbent: _CandidateSchedule,
    M: int,
) -> list[tuple[tuple[int, ...], tuple[int, int]]]:
    """Return valid chains and windows derived from the same incumbent.

    The previous implementation always chose a chain ending at the
    bottleneck and could emit crane indices outside ``range(M)`` for small
    fleets.  Build chains around both neighbours of the latest-working crane
    and clamp their widths before they reach the repair operator.
    """
    if M <= 0 or incumbent.makespan < 4:
        return []
    last_work = [-1] * M
    by_time: dict[int, list[Slot]] = {}
    for slot in incumbent.slots:
        by_time.setdefault(slot.time, []).append(slot)
        if slot.state == "work":
            last_work[slot.crane - 1] = max(last_work[slot.crane - 1], slot.time)
    bottleneck = max(range(M), key=last_work.__getitem__)
    centres = list(dict.fromkeys(
        sorted(range(M), key=lambda q: (abs(q - bottleneck), -last_work[q]))
    ))
    chains: list[tuple[int, ...]] = []
    for width in range(1, min(4, M) + 1):
        for centre in centres[:min(3, len(centres))]:
            left = max(0, min(centre - width // 2, M - width))
            chain = tuple(range(left, left + width))
            if chain not in chains:
                chains.append(chain)
        # Keep an outer-neighbour chain as well.  A saturated inner crane may
        # need a relay that reaches a completed edge crane even though that
        # edge crane is far from the latest-working bottleneck.
        outer = tuple(range(max(0, M - width), M))
        if outer not in chains:
            chains.append(outer)

    horizon = incumbent.makespan - 1
    width = max(3, horizon // 4)
    rows = {
        t: tuple(slot.start_bay for slot in sorted(by_time.get(t, ()), key=lambda x: x.crane))
        for t in range(incumbent.makespan)
    }
    events = [
        t for t in range(1, incumbent.makespan)
        if rows.get(t) and rows.get(t - 1) and rows[t] != rows[t - 1]
    ]
    events.extend(last + 1 for last in last_work if 1 <= last + 1 < horizon)
    max_start = max(1, horizon - width)
    candidates = {
        max(1, min(max_start, event - width // 2)) for event in events
    }
    candidates.update(
        max(1, min(max_start, horizon * fraction // 100))
        for fraction in (35, 50, 65)
    )
    candidates.update((1, max_start))
    # Keep five well-spread event windows.  With the bounded crane chains this
    # stays below the 48-call Step 8 cap while covering early hand-offs, middle
    # relocations, and the tail.
    starts: list[int] = []
    if candidates:
        starts = [min(candidates)]
        if max(candidates) != starts[0]:
            starts.append(max(candidates))
        while len(starts) < min(5, len(candidates)):
            remaining = [value for value in candidates if value not in starts]
            starts.append(max(
                remaining,
                key=lambda value: (min(abs(value - chosen) for chosen in starts), -value),
            ))
        starts.sort()
    windows = []
    for start in starts:
        end = min(horizon, start + width)
        if end > start:
            windows.append((start, end))
    return [(chain, window) for chain in chains for window in windows]


def _targeted_local_preparation(
    W: Sequence[int],
    M: int,
    candidate: _CandidateSchedule,
    chain: Sequence[int],
    window: tuple[int, int],
) -> tuple[_CandidateSchedule | None, int]:
    """Enumerate one-row hand-off cascades aimed at current deficit bays."""
    if candidate.move_time != 0:
        return None, 0
    reference, _, deficits = _shortening_potential(W, M, candidate)
    targets = [bay for bay, amount in enumerate(deficits, 1) if amount]
    if not targets:
        return _decode_best_shortening(W, M, candidate), 0
    active = tuple(sorted({q for q in chain if 0 <= q < M}))
    active_set = set(active)
    if not active:
        return None, 0
    rows = _candidate_position_rows(candidate, M)
    capacity = [0] * len(W)
    for row in rows[:-1]:
        for bay in row:
            if 1 <= bay <= len(W):
                capacity[bay - 1] += 1
    start = max(1, window[0])
    end = min(candidate.makespan - 1, window[1])
    evaluated = 0
    best: _CandidateSchedule | None = None
    best_key = reference[:5]
    for t in range(start, end):
        original = rows[t]
        for q in active:
            for target in targets:
                if target == original[q]:
                    continue
                variants: list[list[int]] = []
                direct = original[:]
                direct[q] = target
                variants.append(direct)
                cascade = original[:]
                cascade[q] = target
                if target < original[q]:
                    previous = original[q]
                    for other in range(q + 1, M):
                        if other not in active_set:
                            break
                        displaced = original[other]
                        cascade[other] = previous
                        previous = displaced
                else:
                    following = original[q]
                    for other in range(q - 1, -1, -1):
                        if other not in active_set:
                            break
                        displaced = original[other]
                        cascade[other] = following
                        following = displaced
                variants.append(cascade)
                for replacement in variants:
                    evaluated += 1
                    if any(position < 1 or position > len(W) for position in replacement):
                        continue
                    if any(right - left < 2 for left, right in zip(replacement, replacement[1:])):
                        continue
                    proposed_capacity = capacity[:]
                    for old, new in zip(original, replacement):
                        if old == new:
                            continue
                        proposed_capacity[old - 1] -= 1
                        proposed_capacity[new - 1] += 1
                    if any(
                        proposed_capacity[i] < W[i] for i in range(len(W))
                    ):
                        continue
                    proposed_rows = rows[:]
                    proposed_rows[t] = replacement
                    try:
                        proposed = _candidate_from_history(
                            W, M, [tuple(row) for row in proposed_rows], 0
                        )
                    except RuntimeError:
                        continue
                    if proposed.makespan < candidate.makespan:
                        return proposed, evaluated
                    if proposed.makespan != candidate.makespan:
                        continue
                    potential, _, _ = _shortening_potential(W, M, proposed)
                    if potential[:5] < best_key:
                        best = proposed
                        best_key = potential[:5]
    return best, evaluated


def _cumulative_local_trajectory_repair(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    incumbent: _CandidateSchedule,
    deadline: float,
    seed: int,
    move_time: int = 1,
    attempt_trace: list[dict[str, Any]] | None = None,
) -> tuple[_CandidateSchedule | None, int, _CandidateSchedule]:
    """Prepare and shorten through cumulative bounded local windows.

    Every mutation remains inside one ordinary critical window.  Accepted
    same-horizon preparations become the source of the next window, allowing
    two distant local repairs to cooperate without introducing a global
    window.  The returned third value is the best prepared H schedule even if
    no H-1 schedule was found, so callers can retain it as an elite.
    """
    current = incumbent
    evaluated_total = 0
    if move_time != 0:
        return None, evaluated_total, current
    current_potential, _, _ = _shortening_potential(W, M, current)
    cycle = 0
    stale_cycles = 0
    while time.perf_counter() < deadline and cycle < 4 and stale_cycles < 2:
        windows = _critical_repair_windows(current, M)
        if not windows:
            break
        progress = False
        for index, (chain, window) in enumerate(windows):
            if time.perf_counter() >= deadline:
                break
            targeted, evaluated = _targeted_local_preparation(
                W, M, current, chain, window
            )
            evaluated_total += evaluated
            if targeted is not None:
                if targeted.makespan < current.makespan:
                    if attempt_trace is not None:
                        attempt_trace.append({
                            "cycle": cycle, "window_index": index,
                            "chain": list(chain), "window": list(window),
                            "phase": "targeted", "evaluated": evaluated,
                            "accepted": True, "shortened": True,
                        })
                    return targeted, evaluated_total, current
                targeted_potential, remove_at, deficits = _shortening_potential(
                    W, M, targeted
                )
                if attempt_trace is not None:
                    attempt_trace.append({
                        "cycle": cycle, "window_index": index,
                        "chain": list(chain), "window": list(window),
                        "phase": "targeted", "evaluated": evaluated,
                        "accepted": True,
                        "before_potential": list(current_potential),
                        "after_potential": list(targeted_potential),
                        "best_remove_at": remove_at,
                        "deficit_bays": [
                            bay for bay, amount in enumerate(deficits, 1) if amount
                        ],
                    })
                current = targeted
                current_potential = targeted_potential
                progress = True
                shortened = _decode_best_shortening(W, M, current)
                if shortened is not None:
                    return shortened, evaluated_total, current
                break
            remaining_calls = max(1, len(windows) - index)
            slice_deadline = min(
                deadline,
                time.perf_counter() + max(
                    0.05,
                    (deadline - time.perf_counter()) / remaining_calls,
                ),
            )
            prepare_deadline = time.perf_counter() + 0.65 * (
                slice_deadline - time.perf_counter()
            )
            prepared, evaluated, _ = _trajectory_repair(
                W, M, starts, current, prepare_deadline,
                seed + cycle * 1009 + index,
                active_cranes=chain,
                window=window,
                return_state=True,
                move_time=move_time,
                preserve_horizon=True,
                preparation_mode=True,
            )
            evaluated_total += evaluated
            record = {
                "cycle": cycle,
                "window_index": index,
                "chain": list(chain),
                "window": list(window),
                "phase": "prepare",
                "before_potential": list(current_potential),
                "evaluated": evaluated,
                "accepted": prepared is not None,
            }
            if prepared is not None:
                prepared_potential, remove_at, deficits = _shortening_potential(
                    W, M, prepared
                )
                record.update({
                    "after_potential": list(prepared_potential),
                    "best_remove_at": remove_at,
                    "deficit_bays": [
                        bay for bay, amount in enumerate(deficits, 1) if amount
                    ],
                })
                current = prepared
                current_potential = prepared_potential
                progress = True
                shortened = _decode_best_shortening(W, M, current)
                if attempt_trace is not None:
                    attempt_trace.append(record)
                if shortened is not None:
                    return shortened, evaluated_total, current
                # Recompute event windows from the newly prepared trajectory.
                break
            if attempt_trace is not None:
                attempt_trace.append(record)
            if time.perf_counter() < slice_deadline:
                shortened, evaluated, _ = _trajectory_repair(
                    W, M, starts, current, slice_deadline,
                    seed + 500_003 + cycle * 1009 + index,
                    active_cranes=chain,
                    window=window,
                    return_state=True,
                    move_time=move_time,
                )
                evaluated_total += evaluated
                if attempt_trace is not None:
                    attempt_trace.append({
                        "cycle": cycle,
                        "window_index": index,
                        "chain": list(chain),
                        "window": list(window),
                        "phase": "shorten",
                        "before_potential": list(current_potential),
                        "evaluated": evaluated,
                        "accepted": shortened is not None,
                    })
                if shortened is not None:
                    return shortened, evaluated_total, current
        if progress:
            stale_cycles = 0
        else:
            stale_cycles += 1
        cycle += 1
    return None, evaluated_total, current


def _critical_window_beam_repair(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    incumbent: _CandidateSchedule,
    chain: Sequence[int],
    deadline: float,
    seed: int,
    window: tuple[int, int] | None = None,
    attempt_trace: list[dict[str, Any]] | None = None,
    move_time: int = 1,
    preserve_horizon: bool = False,
) -> tuple[_CandidateSchedule | None, int]:
    """Rebuild one explicit ``[start,end)`` window for a valid crane chain.

    A target row is removed only inside the requested window.  Prefix work is
    consumed once, the beam must reach the complete window end, and suffix
    work is consumed only after the active cranes are joined to a safe exit
    boundary.  The source exit is ranked first, while nearby exits remain
    available.  This prevents the old implementation from joining an
    incomplete beam layer to a suffix and claiming a candidate.
    """
    target = incumbent.makespan if preserve_horizon else incumbent.makespan - 1
    if target < 3:
        return None, 0
    active = tuple(sorted({q for q in chain if 0 <= q < M}))
    if not active:
        return None, 0
    if window is None:
        window = (max(1, target // 3), min(target, max(2, target // 3 + max(3, target // 4))))
    spec = RepairWindow(window[0], window[1], active).normalized(target, M)
    if spec is None:
        return None, 0
    rng = random.Random(seed)
    eligible_by_bay = _bay_eligibility(len(W), M, [])
    base_rows = [[0] * M for _ in range(incumbent.makespan + 1)]
    for slot in incumbent.slots:
        base_rows[slot.time][slot.crane - 1] = slot.start_bay
        base_rows[slot.time + 1][slot.crane - 1] = slot.end_bay
    active = tuple(
        q for q in active
        if all(
            1 <= base_rows[t][q] <= len(W)
            for t in range(spec.start, spec.end + 1)
        )
    )
    if not active:
        return None, 0
    required = set(starts)
    evaluated = 0
    attempts = 0
    beam_width = 500 if len(active) <= 2 else 280 if len(active) <= 3 else 160

    def safe(row: Sequence[int]) -> bool:
        return all(right - left >= 2 for left, right in zip(row, row[1:]))

    def consume(remaining: list[int], current: Sequence[int], following: Sequence[int]) -> None:
        for start_bay, end_bay in zip(current, following):
            if not 1 <= start_bay <= len(W):
                continue
            if (move_time == 0 or start_bay == end_bay) and remaining[int(start_bay) - 1] > 0:
                remaining[int(start_bay) - 1] -= 1

    while time.perf_counter() < deadline and attempts < 24:
        attempts += 1
        # Shortening deletes one boundary. Multi-objective refinement keeps
        # the full horizon and rebuilds the selected window in place.
        remove_low = spec.start + 1
        remove_high = min(spec.end, target)
        if remove_low > remove_high:
            return None, evaluated
        remove_at = None if preserve_horizon else rng.randint(remove_low, remove_high)
        if attempt_trace is not None:
            attempt_trace.append({"remove_at": remove_at, "window": tuple(window)})
        rows = base_rows[:] if preserve_horizon else base_rows[:remove_at] + base_rows[remove_at + 1:]
        if any(not safe(row) for row in rows):
            continue

        remaining = list(W)
        for t in range(spec.start):
            consume(remaining, rows[t], rows[t + 1])
        initial_active = tuple(rows[spec.start][q] for q in active)
        exit_active = tuple(rows[spec.end][q] for q in active)
        # A state value is (remaining work, active positions) -> active rows.
        states: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], ...]] = {
            (tuple(remaining), initial_active): (initial_active,)
        }
        complete_depth = True
        for t in range(spec.start, spec.end):
            if time.perf_counter() >= deadline:
                return None, evaluated
            next_states: dict[
                tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], ...]
            ] = {}
            for (state_remaining, current_active), active_history in states.items():
                choices: list[list[int]] = []
                local_remaining = list(state_remaining)
                for q_index, q in enumerate(active):
                    allowed = [
                        bay for bay in range(1, len(W) + 1)
                        if q in eligible_by_bay[bay - 1]
                    ]
                    ranked = sorted(
                        allowed,
                        key=lambda bay: (
                            local_remaining[bay - 1], W[bay - 1],
                            -abs(bay - current_active[q_index]),
                        ),
                        reverse=True,
                    )
                    values = [
                        exit_active[q_index], current_active[q_index], rows[t + 1][q],
                        *ranked[:3],
                    ]
                    if allowed:
                        values.extend(rng.sample(allowed, min(2, len(allowed))))
                    choices.append(list(dict.fromkeys(values)))
                for next_active in itertools.product(*choices):
                    evaluated += 1
                    if evaluated % 128 == 0 and time.perf_counter() >= deadline:
                        return None, evaluated
                    full_current = list(rows[t])
                    full_next = list(rows[t + 1])
                    for q, bay in zip(active, current_active):
                        full_current[q] = bay
                    for q, bay in zip(active, next_active):
                        full_next[q] = bay
                    if not safe(full_current) or not safe(full_next):
                        continue
                    new_remaining = list(state_remaining)
                    consume(new_remaining, full_current, full_next)
                    if not any(new_remaining) and t + 1 < target:
                        # It is still legal to idle, but keep the state only
                        # if the fixed suffix has no work left to consume.
                        pass
                    slots_left = target - (t + 1)
                    if sum(new_remaining) > slots_left * M:
                        continue
                    if max(new_remaining, default=0) > slots_left:
                        continue
                    key = (tuple(new_remaining), tuple(next_active))
                    if key not in next_states:
                        next_states[key] = active_history + (tuple(next_active),)
            if not next_states:
                complete_depth = False
                break

            def state_key(item):
                (remaining_state, positions), _ = item
                ready = sum(remaining_state[bay - 1] > 0 for bay in positions)
                return sum(remaining_state), max(remaining_state, default=0), -ready

            states = dict(sorted(next_states.items(), key=state_key)[:beam_width])
        if not complete_depth:
            continue

        for (state_remaining, _), active_history in sorted(
            states.items(), key=lambda item: (sum(item[0][0]), max(item[0][0], default=0))
        ):
            candidate_rows = [row[:] for row in rows]
            for offset, active_row in enumerate(active_history):
                t = spec.start + offset
                for q, bay in zip(active, active_row):
                    candidate_rows[t][q] = bay
            rem = list(state_remaining)
            for t in range(spec.end, target):
                consume(rem, candidate_rows[t], candidate_rows[t + 1])
            if any(rem) or not all(safe(row) for row in candidate_rows):
                continue
            if not required.issubset(set(candidate_rows[0])):
                continue
            try:
                candidate = _candidate_from_history(
                    W, M, [tuple(row) for row in candidate_rows], move_time
                )
            except RuntimeError:
                continue
            if candidate.makespan <= target and (
                not preserve_horizon
                or candidate.makespan == incumbent.makespan
                and candidate.objective_key < incumbent.objective_key
            ):
                return candidate, evaluated
    return None, evaluated


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
    critical_mode: str = "both",
    checkpoint: _CandidateSchedule | None = None,
    skip_general_repair: bool = False,
    stop_before_critical: bool = False,
    move_time: int = 1,
    on_incumbent: Callable[[Solution], None] | None = None,
) -> Solution:
    """Construct a safe schedule without calling an optimization solver."""
    search_start = time.perf_counter()
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts <= 0:
        raise ValueError("restarts 必须是正整数。")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit 必须大于 0。")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError("patience 必须是正整数。")
    if critical_mode not in {"off_reallocate", "off_reserved", "beam", "trajectory", "both"}:
        raise ValueError("critical_mode 必须是 off_reallocate、off_reserved、beam、trajectory 或 both。")
    effective_time_limit = min(float(time_limit), 270.0)
    deadline = search_start + effective_time_limit
    _, starts, configurations = _validate_input(W, M, S, move_time)

    total_work = sum(W)
    max_steps = max(1, 3 * total_work + len(W) + 1)
    target_weights = [min(q + 1, M - q) for q in range(M)]
    initial_configs = _initial_configurations(W, starts, configurations)
    eligibility = _bay_eligibility(len(W), M, configurations)
    lower_bound, lower_bound_components = _makespan_lower_bound(
        W, M, initial_configs, eligibility, starts, move_time
    )
    bay_criticality = _bay_criticality(W, M, eligibility)
    rng = random.Random(seed)
    # Exact pruning is written for one-slot transitions.  It remains a useful
    # heuristic source for other durations, but cannot certify optimality.
    use_exact_search = effective_time_limit >= 2.0 and move_time == 1
    heuristic_deadline = (
        search_start + (
            min(15.0, 0.90 * effective_time_limit) if M <= 3
            # Larger fleets need more independent constructions before a
            # local repair can be useful.  Scale this phase with the budget,
            # but keep at least the old five-second slice and never cross the
            # overall deadline; the later phases retain the remaining time.
            else min(
                12.0,
                0.90 * effective_time_limit,
                max(5.0, 0.20 * effective_time_limit),
            )
        )
        if use_exact_search else deadline
    )
    layered_deadline = (
        search_start + 0.95 * effective_time_limit
        if use_exact_search else deadline
    )
    best: _CandidateSchedule | None = None
    completed = 0
    last_improvement = 0
    published_key = None
    elite_pool: list[_CandidateSchedule] = []
    elite_signatures: set[tuple[Any, ...]] = set()
    elite_repairs = 0
    critical_repair_iterations = 0
    critical_repair_improvements = 0
    repair_states: dict[tuple[Any, ...], RepairState] = {}
    phase_seconds: dict[str, float] = {}
    operator_calls: dict[str, int] = {
        "construction": 0,
        "trajectory": 0,
        "critical_beam": 0,
        "mcts": 0,
        "layered": 0,
        "exact": 0,
    }

    def schedule_signature(candidate: _CandidateSchedule) -> tuple[Any, ...]:
        owner_signature = tuple(
            tuple(sorted(owner_set)) for owner_set in candidate.owners
        )
        move_signature = tuple(
            tuple(
                (slot.start_bay, slot.end_bay)
                for slot in candidate.slots
                if slot.crane == q + 1 and slot.state == "move"
            )
            for q in range(M)
        )
        return owner_signature, move_signature

    def remember_elite(candidate: _CandidateSchedule) -> None:
        """Keep bounded structural alternatives, replacing same-shape losers.

        Timing is deliberately absent from the structural signature: when a
        later candidate has the same ownership/move shape but a better timing
        objective, it must replace the old pool member instead of being
        silently discarded.
        """
        if best is not None and candidate.makespan > best.makespan + 2 and candidate is not best:
            return
        signature = schedule_signature(candidate)
        for index, old in enumerate(elite_pool):
            if schedule_signature(old) != signature:
                continue
            if candidate.objective_key < old.objective_key:
                elite_pool[index] = candidate
                elite_pool.sort(key=lambda item: item.objective_key)
            return
        if signature in elite_signatures:
            return
        elite_signatures.add(signature)
        elite_pool.append(candidate)
        elite_pool.sort(key=lambda item: item.objective_key)
        if len(elite_pool) > 16:
            removed = elite_pool.pop()
            elite_signatures.discard(schedule_signature(removed))

    def refresh_elite_pool() -> None:
        """Pin the current incumbent and remove stale far-worse members."""
        if best is None:
            return
        remember_elite(best)
        if len(elite_pool) <= 1:
            return
        kept = [item for item in elite_pool if item is best or item.makespan <= best.makespan + 2]
        elite_pool[:] = sorted(kept, key=lambda item: item.objective_key)[:16]
        elite_signatures.clear()
        elite_signatures.update(schedule_signature(item) for item in elite_pool)

    def publish(candidate):
        nonlocal published_key
        if on_incumbent is None:
            return
        if published_key is not None and candidate.objective_key >= published_key:
            return
        timed_candidate = _retime_candidate(W, M, candidate, move_time)
        optimal = move_time == 1 and timed_candidate.makespan == lower_bound
        snapshot = Solution(
            status="MAKESPAN_OPTIMAL_BY_LOWER_BOUND" if optimal else "HEURISTIC_FEASIBLE",
            method="dp_dispatch_priority_evolution_trajectory_repair_mcts_exact_no_solver",
            makespan=timed_candidate.makespan, makespan_lower_bound=lower_bound,
            lower_bound_components=lower_bound_components,
            makespan_proven_optimal=optimal, proven_lexicographic_optimal=False,
            assignment_count=timed_candidate.assignment_count, split_bay_count=timed_candidate.split_bay_count,
            load_deviation=timed_candidate.load_deviation, reversal_count=timed_candidate.reversal_count,
            movement_count=timed_candidate.movement_count, crane_loads=timed_candidate.loads,
            target_weights=target_weights,
            bay_cranes={i + 1: [q + 1 for q in sorted(owners)]
                        for i, owners in enumerate(timed_candidate.owners) if owners},
            slots=timed_candidate.slots, restarts_completed=completed,
            strategy_evaluations=list(evaluation_totals),
            layered_search_states=0, layered_search_improvements=0,
            mcts_iterations=0, mcts_improvements=0, exact_search_nodes=0,
            exact_search_improvements=0, exact_search_proved_optimal=False,
            search_seconds=round(time.perf_counter() - search_start, 6), max_steps=max_steps,
            elite_pool_size=len(elite_pool), elite_repairs=elite_repairs,
            critical_repair_iterations=critical_repair_iterations,
            critical_repair_improvements=critical_repair_improvements,
            phase_seconds=dict(phase_seconds),
            operator_calls=dict(operator_calls),
            move_time=move_time,
        )
        verify_solution(W, M, starts, snapshot)
        on_incumbent(snapshot)
        published_key = candidate.objective_key

    # Experimental checkpoint entry: resume exactly at the boundary before
    # the critical-window phase.  The checkpoint is an already validated
    # complete candidate and is intentionally kept private to the ablation
    # harness; normal construction remains unchanged when it is absent.
    if checkpoint is not None:
        best = checkpoint
        remember_elite(checkpoint)
        published_key = checkpoint.objective_key

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
    evaluation_totals = [0] * len(strategies)
    strategy_rewards = [0.0] * len(strategies)
    top_pool_size = min(len(initial_configs), max(20, min(200, restarts)))
    top_pool = initial_configs[:top_pool_size]
    warmup_initial_count = min(8, len(top_pool))
    warmup_trials = warmup_initial_count * len(strategies)
    partition_plan_cache: dict[tuple[int, ...], list[int | None] | None] = {}
    elite_priorities = list(bay_criticality)
    dp_incumbent = None
    decoder_offset = 0

    for round_index in range(0 if checkpoint is not None else restarts):
        if M <= 3 and round_index == 300:
            if best is not None and best.makespan == lower_bound:
                break
            # Restart the original random stream and rule portfolio too.
            # Merely swapping decoders midway lets the first decoder's UCB
            # rewards starve rules that are effective in the second decoder.
            dp_incumbent = best
            best = None
            rng = random.Random(seed)
            strategy_counts = [0] * len(strategies)
            strategy_rewards = [0.0] * len(strategies)
            last_improvement = 0
            decoder_offset = 300
        restart = round_index - decoder_offset
        if round_index > 0 and time.perf_counter() >= heuristic_deadline:
            break
        if (
            best is not None
            and best.makespan == lower_bound
            and restart - last_improvement >= min(patience, 64)
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
        evaluation_totals[strategy_index] += 1
        # Search the priority representation, retaining successful ordering
        # hints instead of repeating the same fixed dispatch rules forever.
        trial_priorities = list(bay_criticality)
        if M > 3 and restart >= warmup_trials and restart % 4:
            trial_priorities = list(elite_priorities)
            for i in range(len(W)):
                if W[i] and rng.random() < (0.2 if restart % 4 == 1 else 0.5):
                    trial_priorities[i] = max(0.0, trial_priorities[i] + rng.uniform(-3.0, 3.0))
            if restart % 4 == 3:
                trial_priorities = [rng.uniform(0.0, 8.0) for _ in W]
        previous_best_makespan = best.makespan if best is not None else max_steps
        # With <= 3 cranes the old focused product is already small and its
        # joint random noise supplies useful diversity. Retain that decoder;
        # DP removes the expensive product for larger crane fleets.
        use_legacy = decoder_offset > 0
        constructor = _construct_schedule_legacy if use_legacy else _construct_schedule
        decoder_options = (
            {'focused_width': None if restart == 0 or restart % 75 == 0 else 2}
            if use_legacy else {}
        )
        try:
            candidate = constructor(
                W, M, starts, configurations, initial, strategy, rng,
                min(max_steps, 2 * total_work + 1) if fixed_owner is not None else max_steps,
                trial_priorities, eligibility, fixed_owner, preferred_owner,
                deadline=None if best is None else heuristic_deadline,
                **decoder_options,
            )
        except RuntimeError:
            # Some randomized fixed-owner plans can trap the greedy decoder.
            # They are discarded; unrestricted decoders remain in the portfolio.
            continue
        completed += 1
        operator_calls["construction"] += 1
        remember_elite(candidate)
        gap = max(0, candidate.makespan - lower_bound)
        reward = (
            1.0 / (1 + gap)
            + 0.25 * max(0, previous_best_makespan - candidate.makespan)
            + 0.05 / (1 + candidate.split_bay_count)
            + 0.01 / (1 + candidate.load_deviation)
            + 0.002 / (1 + candidate.movement_count)
        )
        strategy_rewards[strategy_index] += reward
        if best is None or candidate.objective_key < best.objective_key:
            best = candidate
            remember_elite(candidate)
            refresh_elite_pool()
            elite_priorities = trial_priorities
            last_improvement = restart
            publish(best)

    if dp_incumbent is not None and (best is None or dp_incumbent.objective_key < best.objective_key):
        best = dp_incumbent
    assert best is not None
    phase_seconds["construction"] = round(time.perf_counter() - search_start, 6)
    repair_iterations = repair_improvements = 0
    repair_phase_deadline = time.perf_counter() + max(
        0.0, search_start + 0.95 * effective_time_limit - time.perf_counter()
    ) * 0.65
    general_fraction = 1.0 if effective_time_limit <= 15.0 else 0.55
    general_repair_deadline = (
        time.perf_counter()
        if checkpoint is not None or skip_general_repair
        else time.perf_counter() + max(
            0.0, repair_phase_deadline - time.perf_counter()
        ) * general_fraction
    )
    repair_round = 0
    general_repair_started = time.perf_counter()
    while best.makespan > lower_bound and time.perf_counter() < general_repair_deadline:
        if not elite_pool:
            elite_pool.append(best)
        source = elite_pool[repair_round % len(elite_pool)]
        source_key = (
            "general", schedule_signature(source), source.makespan,
        )
        remaining_budget = general_repair_deadline - time.perf_counter()
        slice_deadline = min(
            general_repair_deadline,
            time.perf_counter() + max(0.20, min(8.0, remaining_budget / 3.0)),
        )
        improved, evaluated, continuation = _trajectory_repair(
            W, M, starts, source, slice_deadline, seed + 211 + repair_round,
            repair_state=repair_states.get(source_key), return_state=True,
        )
        operator_calls["trajectory"] += 1
        if continuation is None:
            repair_states.pop(source_key, None)
        else:
            repair_states[source_key] = continuation
        repair_iterations += evaluated
        if improved is not None:
            elite_repairs += 1
            remember_elite(improved)
            if improved.objective_key < best.objective_key:
                best = improved
                repair_improvements += 1
                refresh_elite_pool()
                publish(best)
        repair_round += 1
    phase_seconds["general_repair"] = round(
        time.perf_counter() - general_repair_started, 6
    )

    # A second neighborhood focuses on the crane that finishes last and a
    # contiguous chain of neighbours.  The source and its window are selected
    # together; a window computed for ``best`` is never applied to a different
    # elite schedule with another horizon.
    critical_index = 0
    critical_limit = max(1, min(48, max(1, len(elite_pool)) * 6))
    critical_reserve = min(
        max(0.0, repair_phase_deadline - time.perf_counter()),
        7.0 * critical_limit,
    )
    reserved_deadline = max(search_start, repair_phase_deadline - critical_reserve)
    critical_started = time.perf_counter()
    if (
        not stop_before_critical
        and move_time == 0
        and critical_mode in {"trajectory", "both"}
        and best.makespan > lower_bound
        and time.perf_counter() < repair_phase_deadline
    ):
        cumulative_deadline = repair_phase_deadline
        if critical_mode == "both":
            cumulative_deadline = time.perf_counter() + 0.65 * (
                repair_phase_deadline - time.perf_counter()
            )
        operator_calls["trajectory"] += 1
        cumulative, evaluated, prepared = _cumulative_local_trajectory_repair(
            W, M, starts, best, cumulative_deadline, seed,
            move_time=move_time,
        )
        critical_repair_iterations += evaluated
        if prepared is not best:
            elite_repairs += 1
            remember_elite(prepared)
        if cumulative is not None:
            elite_repairs += 1
            remember_elite(cumulative)
            if cumulative.objective_key < best.objective_key:
                best = cumulative
                critical_repair_improvements += 1
                refresh_elite_pool()
                publish(best)
    while (
        not stop_before_critical
        and critical_mode in {"beam", "trajectory", "both"}
        and
        best.makespan > lower_bound
        and critical_index < critical_limit
        and time.perf_counter() < repair_phase_deadline
    ):
        source = elite_pool[critical_index % len(elite_pool)] if elite_pool else best
        source_windows = _critical_repair_windows(source, M)
        if not source_windows:
            break
        chain, window = source_windows[
            (critical_index // max(1, len(elite_pool))) % len(source_windows)
        ]
        remaining_budget = repair_phase_deadline - time.perf_counter()
        slice_deadline = min(
            repair_phase_deadline,
            time.perf_counter() + max(0.20, min(7.0, remaining_budget / 3.0)),
        )
        if critical_mode == "beam" or (
            critical_mode == "both" and critical_index % 2 == 1
        ):
            operator_calls["critical_beam"] += 1
            improved, evaluated = _critical_window_beam_repair(
                W, M, starts, source, chain, slice_deadline,
                seed + 1009 + critical_index,
                window=window,
                move_time=move_time,
            )
        else:
            operator_calls["trajectory"] += 1
            source_key = (
                "critical", schedule_signature(source), source.makespan,
                tuple(chain), tuple(window),
            )
            improved, evaluated, continuation = _trajectory_repair(
                W, M, starts, source, slice_deadline,
                seed + 1009 + critical_index,
                active_cranes=chain,
                window=window,
                repair_state=repair_states.get(source_key), return_state=True,
                move_time=move_time,
            )
            if continuation is None:
                repair_states.pop(source_key, None)
            else:
                repair_states[source_key] = continuation
        critical_repair_iterations += evaluated
        if improved is not None:
            elite_repairs += 1
            remember_elite(improved)
            if improved.objective_key < best.objective_key:
                best = improved
                critical_repair_improvements += 1
                refresh_elite_pool()
                publish(best)
        critical_index += 1
    phase_seconds["critical_repair"] = round(
        time.perf_counter() - critical_started, 6
    )
    improvement_deadline = (
        time.perf_counter()
        if stop_before_critical
        else reserved_deadline
        if critical_mode == "off_reserved"
        else search_start + 0.95 * effective_time_limit
    )
    if critical_mode == "off_reserved" or stop_before_critical:
        layered_deadline = min(layered_deadline, improvement_deadline)
    improvement_remaining = max(0.0, improvement_deadline - time.perf_counter())
    gap_after_construction = best.makespan - lower_bound
    mcts_share = (
        0.55 if M >= 5 or gap_after_construction <= 1
        else 0.35
    )
    mcts_phase_started = time.perf_counter()
    mcts_deadline = time.perf_counter() + improvement_remaining * mcts_share
    mcts_iterations = 0
    mcts_improvements = 0
    if (
        use_exact_search
        and best.makespan > lower_bound
        and time.perf_counter() < mcts_deadline
    ):
        operator_calls["mcts"] += 1
        improved, evaluated = _mcts_horizon_search(
            W, M, starts, eligibility, bay_criticality, best,
            mcts_deadline, seed + 97,
        )
        mcts_iterations += evaluated
        if improved is not None and improved.objective_key < best.objective_key:
            best = improved
            mcts_improvements += 1
            publish(best)
    phase_seconds["mcts"] = round(time.perf_counter() - mcts_phase_started, 6)

    layered_search_states = 0
    layered_search_improvements = 0
    repair_cutoffs = _blocking_repair_cutoffs(best, M)
    layered_started = time.perf_counter()
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
        operator_calls["layered"] += 1
        layered_search_states += evaluated
        if improved is not None and improved.objective_key < best.objective_key:
            best = improved
            layered_search_improvements += 1
            publish(best)
    phase_seconds["layered"] = round(time.perf_counter() - layered_started, 6)

    exact_search_nodes = 0
    exact_search_improvements = 0
    exact_search_proved_optimal = False
    # Complete enumeration is most useful when the remaining gap is small.
    # For a wide gap on a large state space, spend the budget on diverse
    # improvement passes instead of a proof attempt that cannot finish.
    exact_is_promising = (
        best.makespan - lower_bound <= 1
        and len(configurations) <= 20_000
        and math.comb(len(W) - M + 1, M) <= 20_000
    )
    exact_started = time.perf_counter()
    while (
        use_exact_search
        and exact_is_promising
        and best.makespan > lower_bound
        and time.perf_counter() < improvement_deadline
    ):
        operator_calls["exact"] += 1
        improved, exact_status, evaluated = _exact_horizon_search(
            W, M, starts, configurations, initial_configs, eligibility,
            bay_criticality, best, improvement_deadline,
        )
        exact_search_nodes += evaluated
        if improved is not None and improved.objective_key < best.objective_key:
            best = improved
            exact_search_improvements += 1
            publish(best)
            continue
        if exact_status == "INFEASIBLE":
            exact_search_proved_optimal = True
        break
    phase_seconds["exact"] = round(time.perf_counter() - exact_started, 6)

    elapsed = time.perf_counter() - search_start
    best = _retime_candidate(W, M, best, move_time)
    makespan_optimal = move_time == 1 and (
        best.makespan == lower_bound or exact_search_proved_optimal
    )
    bay_cranes = {
        i + 1: [q + 1 for q in sorted(bay_owners)]
        for i, bay_owners in enumerate(best.owners)
        if bay_owners
    }
    return Solution(
        status=(
            "MAKESPAN_OPTIMAL_BY_LOWER_BOUND"
            if makespan_optimal and best.makespan == lower_bound
            else "MAKESPAN_OPTIMAL_BY_EXACT_SEARCH"
            if exact_search_proved_optimal
            else "HEURISTIC_FEASIBLE"
        ),
        method=(
            "dp_dispatch_priority_construction_only_no_solver"
            if stop_before_critical
            else "dp_dispatch_priority_evolution_trajectory_repair_mcts_exact_no_solver"
        ),
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
        strategy_evaluations=evaluation_totals,
        layered_search_states=layered_search_states,
        layered_search_improvements=layered_search_improvements,
        mcts_iterations=mcts_iterations,
        mcts_improvements=mcts_improvements,
        exact_search_nodes=exact_search_nodes,
        exact_search_improvements=exact_search_improvements,
        exact_search_proved_optimal=exact_search_proved_optimal,
        search_seconds=round(elapsed, 6),
        max_steps=max_steps,
        trajectory_repair_iterations=repair_iterations,
        trajectory_repair_improvements=repair_improvements,
        elite_pool_size=len(elite_pool),
        elite_repairs=elite_repairs,
        critical_repair_iterations=critical_repair_iterations,
        critical_repair_improvements=critical_repair_improvements,
        phase_seconds=dict(phase_seconds),
        operator_calls=dict(operator_calls),
        move_time=move_time,
    )


def verify_solution(W: Sequence[int], M: int, S: Iterable[int], solution: Solution) -> None:
    """Independently verify workload, state, collision, and start constraints."""
    move_time = getattr(solution, "move_time", 1)
    if isinstance(move_time, bool) or not isinstance(move_time, int) or move_time < 0:
        raise AssertionError("move_time 必须是非负整数。")
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
            if not 1 <= slot.work_bay <= len(W):
                raise AssertionError("作业贝位超出 1..N。")
            work_done[slot.work_bay - 1] += 1
            load_done[slot.crane - 1] += 1
        elif slot.state == "move":
            if move_time == 0:
                raise AssertionError("move_time=0 时不应占用移动时间槽。")
            if slot.start_bay == slot.end_bay or slot.work_bay is not None:
                raise AssertionError("移动槽定义错误。")
        elif slot.state == "idle":
            if slot.start_bay != slot.end_bay or slot.work_bay is not None:
                raise AssertionError("空闲槽定义错误。")
            if not 1 <= slot.start_bay <= len(W):
                raise AssertionError("轨道内空闲位置超出 1..N。")
        elif slot.state == "offrail":
            if slot.start_bay != slot.end_bay or slot.work_bay is not None:
                raise AssertionError("退场槽定义错误。")
            if 1 <= slot.start_bay <= len(W):
                raise AssertionError("offrail 状态必须位于工作轨道之外。")
        else:
            raise AssertionError(f"未知状态：{slot.state}")

    if work_done != list(W):
        raise AssertionError(f"作业量不守恒：得到 {work_done}，要求 {list(W)}。")
    if load_done != solution.crane_loads:
        raise AssertionError("桥吊负荷汇总不一致。")
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
        if t > 0 and move_time > 0:
            previous = sorted(by_time[t - 1], key=lambda row: row.crane)
            for before, current in zip(previous, rows):
                if before.end_bay != current.start_bay:
                    raise AssertionError(
                        f"t={t} Q{current.crane} 位置不连续："
                        f"上一槽结束于 {before.end_bay}，本槽开始于 {current.start_bay}。"
                    )

    if move_time == 0:
        directions: list[list[int]] = [[] for _ in range(M)]
        visible_moves = 0
        for t in range(1, solution.makespan):
            previous = sorted(by_time[t - 1], key=lambda row: row.crane)
            current = sorted(by_time[t], key=lambda row: row.crane)
            for q, (before, after) in enumerate(zip(previous, current)):
                if before.end_bay != after.start_bay:
                    visible_moves += 1
                    directions[q].append(1 if after.start_bay > before.end_bay else -1)
        reversals = sum(
            a != b
            for crane_directions in directions
            for a, b in zip(crane_directions, crane_directions[1:])
        )
        if visible_moves != solution.movement_count:
            raise AssertionError("瞬时移动次数汇总不一致。")
    else:
        reversals = _count_reversals(solution.slots, M)
        if move_time == 1:
            movement_count = sum(slot.state == "move" for slot in solution.slots)
        else:
            groups: dict[tuple[int, int], list[Slot]] = {}
            for slot in solution.slots:
                if slot.state != "move":
                    continue
                if slot.move_id is None:
                    raise AssertionError("多时段移动缺少 move_id。")
                groups.setdefault((slot.crane, slot.move_id), []).append(slot)
            for group in groups.values():
                steps = sorted(slot.move_step for slot in group)
                if len(group) != move_time or steps != list(range(1, move_time + 1)):
                    raise AssertionError("移动持续时间与 move_time 不一致。")
                if any(slot.move_steps != move_time for slot in group):
                    raise AssertionError("move_steps 与 move_time 不一致。")
            movement_count = len(groups)
        if movement_count != solution.movement_count:
            raise AssertionError("移动次数汇总不一致。")
    if reversals != solution.reversal_count:
        raise AssertionError("桥吊折返次数汇总不一致。")

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
    diagnostic_title: str | None = None,
    highlight_window: tuple[int, int] | None = None,
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

    if highlight_window is not None:
        window_start, window_end = highlight_window
        ax.axhspan(
            window_start,
            window_end,
            facecolor="#ffd54f",
            edgecolor="#f9a825",
            linewidth=1.2,
            alpha=0.18,
            zorder=0,
        )

    for slot in solution.slots:
        color = colors[slot.crane - 1]
        y0 = slot.time
        if slot.state == "offrail":
            continue
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

    if getattr(solution, "move_time", 1) == 0:
        # Instantaneous relocations live between periods and therefore have
        # no Slot of their own.  Draw them at the shared time boundary.
        for q in range(1, crane_count + 1):
            crane_slots = sorted(
                (slot for slot in solution.slots if slot.crane == q),
                key=lambda slot: slot.time,
            )
            for previous, current in zip(crane_slots, crane_slots[1:]):
                if previous.state == "offrail" or current.state == "offrail":
                    continue
                if previous.end_bay == current.start_bay:
                    continue
                boundary = current.time
                ax.annotate(
                    "",
                    xy=(current.start_bay, boundary + 0.06),
                    xytext=(previous.end_bay, boundary - 0.06),
                    arrowprops=dict(
                        arrowstyle="->", color=colors[q - 1],
                        lw=1.6, linestyle="--",
                    ),
                )

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
    title = (
        f"CWP schedule — {proof_text}, makespan {solution.makespan}, "
        f"reversals {solution.reversal_count}, moves {solution.movement_count}, "
        f"split bays {solution.split_bay_count}"
    )
    if diagnostic_title:
        title += f"\n{diagnostic_title}"
    ax.set_title(title)
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
    print("目标优先级: 完工时间 → 拆分贝位数 → 中间重载偏差 → 移动次数")
    print(f"桥吊-贝位分配数（统计）: {solution.assignment_count}")
    print(f"被多吊拆分的贝位数: {solution.split_bay_count}")
    print(f"各桥吊作业负荷: {solution.crane_loads}")
    print(f"中间重载目标权重: {solution.target_weights}")
    print(f"中间重载偏差: {solution.load_deviation}")
    print(f"折返次数（统计）: {solution.reversal_count}")
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
        f"完整轨迹修复迭代数: {solution.trajectory_repair_iterations}；"
        f"工期改进次数: {solution.trajectory_repair_improvements}"
    )
    print(
        f"精英方案池: {solution.elite_pool_size}；"
        f"精英修复次数: {solution.elite_repairs}；"
        f"关键桥吊修复迭代数: {solution.critical_repair_iterations}；"
        f"关键桥吊工期改进次数: {solution.critical_repair_improvements}"
    )
    print(
        f"精确搜索节点数: {solution.exact_search_nodes}；"
        f"工期改进次数: {solution.exact_search_improvements}；"
        f"是否由精确搜索证明: {solution.exact_search_proved_optimal}"
    )


def _solve_worker(payload, options, checkpoint, error_path):
    """Private worker: publish validated incumbents with atomic replacement."""
    def save(solution):
        temporary = Path(str(checkpoint) + '.tmp')
        temporary.write_text(json.dumps(solution.to_dict()), encoding='utf-8')
        os.replace(temporary, checkpoint)

    try:
        result = solve_cwp(**payload, **options, on_incumbent=save)
        verify_solution(payload['W'], payload['M'], payload['S'], result)
        save(result)
    except Exception as exc:
        Path(error_path).write_text(f'{type(exc).__name__}: {exc}', encoding='utf-8')


def solve_cwp_bounded(W, M, S, *, time_limit=270.0, **options):
    """Process-enforced search budget, including preprocessing.

    Direct solve_cwp remains cooperative; this entry point is used by the CLI.
    If interrupted, return only a previously validated complete incumbent.
    """
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit 必须是有限正数。')
    budget = min(float(time_limit), 270.0)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='cwp-search-') as folder:
        checkpoint = Path(folder) / 'incumbent.json'
        error_path = Path(folder) / 'error.txt'
        process = multiprocessing.get_context('spawn').Process(
            target=_solve_worker,
            args=({'W': W, 'M': M, 'S': list(S)},
                  dict(options, time_limit=max(0.001, budget - 0.15)), checkpoint, error_path),
        )
        process.start()
        try:
            process.join(max(0.0, budget - (time.perf_counter() - started)))
        finally:
            if process.is_alive():
                process.terminate()
                process.join(0.5)
            if process.is_alive():
                process.kill()
                process.join(0.5)
        if error_path.exists():
            raise RuntimeError(error_path.read_text(encoding='utf-8'))
        if not checkpoint.exists():
            raise TimeoutError('时限内未产生完整可行排程；请增加预算或缩小输入。')
        data = json.loads(checkpoint.read_text(encoding='utf-8'))
        data['slots'] = [Slot(**slot) for slot in data['slots']]
        data['bay_cranes'] = {int(bay): cranes for bay, cranes in data['bay_cranes'].items()}
        solution = Solution(**data)
        solution.search_seconds = round(time.perf_counter() - started, 6)
        return solution


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
        "--critical-mode",
        choices=("off_reallocate", "off_reserved", "beam", "trajectory", "both"),
        default="both",
        help="Critical-window ablation mode",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the figure window (default); use --no-show for headless runs",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    W, M, S = payload["W"], payload["M"], payload.get("S", [])
    move_time = payload.get("move_time", 1)
    # Reserve 30 seconds for validation, JSON output and bounded PNG creation.
    solve_budget = min(
        args.time_limit,
        max(1.0, overall_deadline - time.perf_counter() - 30.0),
    )
    solution = solve_cwp_bounded(
        W, M, S,
        restarts=args.restarts,
        time_limit=solve_budget,
        seed=args.seed,
        patience=args.patience,
        critical_mode=args.critical_mode,
        move_time=move_time,
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
