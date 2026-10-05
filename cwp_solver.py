#!/usr/bin/env python3
"""Solver-free heuristic for the discrete one-rail CWP variant.

The scheduling algorithm is implemented entirely in Python and does not call
OR-Tools, Gurobi, CPLEX, SCIP, PuLP, or any other optimization solver.
Matplotlib is used only to draw the final schedule.
"""

from __future__ import annotations

import argparse
import bisect
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


def _work_completion_time(slots: Sequence[Slot]) -> int:
    """Return the boundary immediately after the last unit of actual work."""
    return max(
        (int(slot.time) + 1 for slot in slots if slot.state == "work"),
        default=0,
    )


def _movement_count_until_completion(
    slots: Sequence[Slot],
    M: int,
    move_time: int,
    completion_time: int,
) -> int:
    """Count distinct crane relocations that start before all work finishes."""
    if completion_time <= 0:
        return 0
    if move_time == 0:
        by_time: dict[int, dict[int, Slot]] = {}
        for slot in slots:
            if slot.time < completion_time:
                by_time.setdefault(int(slot.time), {})[int(slot.crane) - 1] = slot
        return sum(
            by_time[t - 1][q].end_bay != by_time[t][q].start_bay
            for t in range(1, completion_time)
            for q in range(M)
            if q in by_time.get(t - 1, {}) and q in by_time.get(t, {})
        )
    moves = [
        slot for slot in slots
        if slot.state == "move" and slot.time < completion_time
    ]
    if move_time == 1:
        return len(moves)
    return len({(int(slot.crane), int(slot.move_id)) for slot in moves})


def _trajectory_reversal_count(
    slots: Sequence[Slot], M: int, move_time: int,
) -> int:
    """Count direction changes in the visible trajectory for either move model."""
    if move_time != 0:
        return _count_reversals(slots, M)
    by_time: dict[int, dict[int, Slot]] = {}
    for slot in slots:
        by_time.setdefault(int(slot.time), {})[int(slot.crane) - 1] = slot
    directions: list[list[int]] = [[] for _ in range(M)]
    for t in range(1, max(by_time, default=-1) + 1):
        for q in range(M):
            before = by_time.get(t - 1, {}).get(q)
            current = by_time.get(t, {}).get(q)
            if before is None or current is None or before.end_bay == current.start_bay:
                continue
            directions[q].append(1 if current.start_bay > before.end_bay else -1)
    return sum(
        previous != current
        for crane_directions in directions
        for previous, current in zip(crane_directions, crane_directions[1:])
    )


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
    schedule_horizon: int | None = None
    schedule_movement_count: int | None = None

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
    completion_time: int = field(init=False)
    completion_movement_count: int = field(init=False)
    normalization_source_horizon: int | None = field(default=None, init=False)
    normalization_trimmed_slots: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.completion_time = _work_completion_time(self.slots)
        self.completion_movement_count = _movement_count_until_completion(
            self.slots, len(self.loads), self.move_time, self.completion_time
        )

    @property
    def schedule_horizon(self) -> int:
        """Length of the stored trajectory, which may include trailing idle."""
        return self.makespan

    @property
    def objective_key(self) -> tuple[int, int]:
        """User priorities: actual work completion time, then crane moves."""
        return (self.completion_time, self.completion_movement_count)


def _normalize_completed_candidate(
    candidate: _CandidateSchedule,
) -> _CandidateSchedule:
    """Drop a zero-time trajectory's all-idle tail without changing its work."""
    completion = candidate.completion_time
    if candidate.move_time != 0 or completion >= candidate.makespan:
        return candidate
    slots = [slot for slot in candidate.slots if slot.time < completion]
    movements = _movement_count_until_completion(
        slots, len(candidate.loads), candidate.move_time, completion
    )
    normalized = _CandidateSchedule(
        slots=slots,
        makespan=completion,
        assignment_count=candidate.assignment_count,
        split_bay_count=candidate.split_bay_count,
        load_deviation=candidate.load_deviation,
        reversal_count=_trajectory_reversal_count(
            slots, len(candidate.loads), candidate.move_time
        ),
        movement_count=movements,
        loads=list(candidate.loads),
        owners=[set(owner) for owner in candidate.owners],
        move_time=candidate.move_time,
    )
    normalized.normalization_source_horizon = (
        candidate.normalization_source_horizon
        if candidate.normalization_source_horizon is not None
        else candidate.schedule_horizon
    )
    normalized.normalization_trimmed_slots = (
        candidate.normalization_trimmed_slots
        + candidate.schedule_horizon - completion
    )
    return normalized


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


def _short_excursion_details(
    candidate: _CandidateSchedule,
    M: int,
    short_visit_limit: int = 2,
) -> list[tuple[int, int, int, int, int, int]]:
    """Return short ``A -> B -> A`` work visits in a zero-time trajectory.

    The returned tuples are ``(crane, start, end, base, excursion, work)``;
    ``start`` and ``end`` are inclusive time rows.  A visit counts only when
    it performs work, so an idle position block is not mislabeled as an
    operational detour.
    """
    if candidate.move_time != 0 or candidate.makespan <= 0:
        return []
    if isinstance(short_visit_limit, bool) or short_visit_limit < 1:
        raise ValueError("short_visit_limit 必须是正整数。")
    rows = _candidate_position_rows(candidate, M)
    horizon = candidate.makespan
    work = {
        (slot.time, slot.crane - 1)
        for slot in candidate.slots
        if slot.state == "work" and 0 <= slot.time < horizon
    }
    details: list[tuple[int, int, int, int, int, int]] = []
    for q in range(M):
        start = 0
        while start < horizon:
            position = rows[start][q]
            end = start
            while end + 1 < horizon and rows[end + 1][q] == position:
                end += 1
            length = end - start + 1
            if (
                start > 0
                and end + 1 <= horizon
                and rows[start - 1][q] == rows[end + 1][q]
                and rows[start - 1][q] != position
                and length <= short_visit_limit
            ):
                work_count = sum((t, q) in work for t in range(start, end + 1))
                if work_count:
                    details.append((
                        q, start, end, rows[start - 1][q], position, work_count,
                    ))
            start = end + 1
    return details


def _trajectory_smoothness(
    candidate: _CandidateSchedule,
    M: int,
    short_visit_limit: int = 2,
) -> tuple[int, int, int]:
    """Return diagnostics for short visits and fragmented position blocks."""
    if candidate.move_time != 0:
        return (0, 0, 0)
    rows = _candidate_position_rows(candidate, M)
    horizon = candidate.makespan
    blocks = 0
    for q in range(M):
        blocks += 1
        for t in range(1, horizon):
            if rows[t][q] != rows[t - 1][q]:
                blocks += 1
    details = _short_excursion_details(candidate, M, short_visit_limit)
    return (
        len(details),
        sum(item[-1] for item in details),
        blocks,
    )


def _continuity_diagnostics(
    candidate: _CandidateSchedule,
    M: int,
    starts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Describe real work fragmentation and crane revisits.

    ``_trajectory_smoothness`` is intentionally retained as the historical
    compatibility metric.  It only counts short work visits, however, so it
    cannot see a seven-period return or the same crane doing one bay at the
    beginning and again at the end.  This report works on the complete
    absolute-time schedule and distinguishes work visits from idle/yielding
    visits.
    """
    if candidate.makespan < 0:
        raise ValueError("candidate.makespan 必须是非负整数。")
    horizon = candidate.makespan
    work_by_time_crane: dict[tuple[int, int], int] = {}
    work_times_by_bay: list[list[int]] = [[] for _ in candidate.owners]
    work_slots_by_bay: list[list[tuple[int, int]]] = [
        [] for _ in candidate.owners
    ]
    for slot in candidate.slots:
        if slot.state != "work" or slot.work_bay is None:
            continue
        q = slot.crane - 1
        bay = int(slot.work_bay)
        if 0 <= slot.time < horizon and 0 <= q < M and 1 <= bay <= len(candidate.owners):
            work_by_time_crane[(slot.time, q)] = bay
            work_times_by_bay[bay - 1].append(slot.time)
            work_slots_by_bay[bay - 1].append((slot.time, q))

    def interval_records(times: Sequence[int], bay: int) -> list[dict[str, Any]]:
        if not times:
            return []
        unique = sorted(set(times))
        records: list[dict[str, Any]] = []
        start = previous = unique[0]
        for value in unique[1:] + [None]:
            if value is not None and value == previous + 1:
                previous = value
                continue
            end = previous + 1
            owners = sorted(
                q + 1
                for t, q in work_slots_by_bay[bay - 1]
                if start <= t < end
            )
            records.append({
                "bay": bay,
                "start": start,
                "end_exclusive": end,
                "length": end - start,
                "work_count": end - start,
                "cranes": sorted(set(owners)),
            })
            if value is not None:
                start = previous = value
        return records

    work_blocks_by_bay: dict[str, list[dict[str, Any]]] = {}
    bay_gaps_by_bay: dict[str, list[dict[str, Any]]] = {}
    bay_fragmentation = 0
    for bay, times in enumerate(work_times_by_bay, 1):
        blocks = interval_records(times, bay)
        work_blocks_by_bay[str(bay)] = blocks
        bay_fragmentation += max(0, len(blocks) - 1)
        gaps: list[dict[str, Any]] = []
        for left, right in zip(blocks, blocks[1:]):
            if right["start"] > left["end_exclusive"]:
                gaps.append({
                    "start": left["end_exclusive"],
                    "end_exclusive": right["start"],
                    "length": right["start"] - left["end_exclusive"],
                })
        bay_gaps_by_bay[str(bay)] = gaps

    # Work-segment diagnostics are crane/bay specific.  A bay-level block
    # count alone cannot see that a local exchange has moved a one-slot
    # fragment from one crane to another (the H=208 Q1/Q2 regression did
    # exactly that), so keep the full segment identity here.
    work_blocks_by_crane_bay: dict[str, list[dict[str, Any]]] = {}
    for q in range(M):
        by_bay: dict[int, list[int]] = {}
        for (time, crane), bay in work_by_time_crane.items():
            if crane == q:
                by_bay.setdefault(int(bay), []).append(int(time))
        for bay, times in sorted(by_bay.items()):
            unique = sorted(set(times))
            start = previous = unique[0]
            for value in unique[1:] + [None]:
                if value is not None and value == previous + 1:
                    previous = value
                    continue
                work_blocks_by_crane_bay.setdefault(str(q + 1), []).append({
                    "crane": q + 1,
                    "bay": bay,
                    "start": start,
                    "end_exclusive": previous + 1,
                    "length": previous - start + 1,
                })
                if value is not None:
                    start = previous = value

    work_block_count_by_crane_bay: dict[str, dict[str, int]] = {
        crane: {
            str(bay): sum(
                int(block["bay"]) == bay for block in blocks
            )
            for bay in sorted({int(block["bay"]) for block in blocks})
        }
        for crane, blocks in work_blocks_by_crane_bay.items()
    }
    extra_work_blocks_total = sum(
        max(0, count - 1)
        for by_bay in work_block_count_by_crane_bay.values()
        for count in by_bay.values()
    )

    short_excursions = _short_excursion_details(candidate, M)
    short_excursion_records = [
        {
            "crane": crane + 1,
            "start": start,
            "end_exclusive": end + 1,
            "base_position": base,
            "excursion_position": excursion,
            "work_count": work_count,
            "length": end - start + 1,
        }
        for crane, start, end, base, excursion, work_count in short_excursions
    ]
    if candidate.move_time == 0:
        rows = _candidate_position_rows(candidate, M)
    else:
        # The detailed continuity search is defined for instantaneous moves.
        # Keep the report useful for other callers without pretending that a
        # fractional move is a work position.
        rows = [
            [int(slot.start_bay) for slot in sorted(
                (item for item in candidate.slots if item.time == t),
                key=lambda item: item.crane,
            )]
            for t in range(horizon)
        ]

    movement_arcs_by_crane: dict[str, list[dict[str, int]]] = {}
    movement_count_by_crane: dict[str, int] = {}
    for q in range(M):
        arcs: list[dict[str, int]] = []
        if rows:
            for t, (before, after) in enumerate(zip(rows, rows[1:])):
                if before[q] != after[q]:
                    arcs.append({
                        "time": t,
                        "from": int(before[q]),
                        "to": int(after[q]),
                        "direction": 1 if after[q] > before[q] else -1,
                    })
        movement_arcs_by_crane[str(q + 1)] = arcs
        movement_count_by_crane[str(q + 1)] = len(arcs)

    crane_position_blocks: list[dict[str, Any]] = []
    crane_work_revisits: list[dict[str, Any]] = []
    pure_yielding_revisits: list[dict[str, Any]] = []
    position_revisit_count = 0
    work_revisit_count = 0
    for q in range(M):
        if not rows or len(rows[0]) <= q:
            continue
        blocks: list[dict[str, Any]] = []
        start = 0
        while start < horizon:
            position = rows[start][q]
            end = start + 1
            while end < horizon and rows[end][q] == position:
                end += 1
            work_bays = sorted({
                work_by_time_crane[(t, q)]
                for t in range(start, end)
                if (t, q) in work_by_time_crane
            })
            block = {
                "crane": q + 1,
                "start": start,
                "end_exclusive": end,
                "length": end - start,
                "position": position,
                "work_count": sum(
                    (t, q) in work_by_time_crane for t in range(start, end)
                ),
                "work_bays": work_bays,
                "state": "work" if work_bays else "idle_or_offrail",
            }
            blocks.append(block)
            start = end
        crane_position_blocks.extend(blocks)
        seen_work: dict[int, dict[str, Any]] = {}
        first_work: dict[int, dict[str, Any]] = {}
        seen_any: dict[int, dict[str, Any]] = {}
        for block_index, block in enumerate(blocks):
            position = int(block["position"])
            previous_any = seen_any.get(position)
            if previous_any is not None:
                position_revisit_count += 1
                if not block["work_count"]:
                    pure_yielding_revisits.append({
                        "crane": q + 1,
                        "position": position,
                        "start": block["start"],
                        "end_exclusive": block["end_exclusive"],
                        "length": block["length"],
                        "previous_block": {
                            "start": previous_any["start"],
                            "end_exclusive": previous_any["end_exclusive"],
                        },
                    })
            if block["work_count"]:
                previous_work = seen_work.get(position)
                if previous_work is not None:
                    primary_work = first_work[position]
                    gap = max(
                        0, int(block["start"])
                        - int(previous_work["end_exclusive"]),
                    )
                    previous_position = (
                        blocks[block_index - 1]["position"]
                        if block_index > 0 else None
                    )
                    next_position = (
                        blocks[block_index + 1]["position"]
                        if block_index + 1 < len(blocks) else None
                    )
                    work_revisit_count += 1
                    crane_work_revisits.append({
                        "crane": q + 1,
                        "bay": position,
                        "start": block["start"],
                        "end_exclusive": block["end_exclusive"],
                        "length": block["length"],
                        "work_count": block["work_count"],
                        "gap": gap,
                        "residual_length": block["work_count"],
                        "primary_block": {
                            "start": primary_work["start"],
                            "end_exclusive": primary_work["end_exclusive"],
                            "length": primary_work["length"],
                            "work_count": primary_work["work_count"],
                        },
                        "residual_block": {
                            "start": block["start"],
                            "end_exclusive": block["end_exclusive"],
                            "length": block["length"],
                            "work_count": block["work_count"],
                        },
                        "previous_block": {
                            "start": previous_work["start"],
                            "end_exclusive": previous_work["end_exclusive"],
                            "length": previous_work["length"],
                            "work_count": previous_work["work_count"],
                        },
                        "movement_before": (
                            previous_position is not None
                            and previous_position != position
                        ),
                        "movement_after": (
                            next_position is not None
                            and next_position != position
                        ),
                        "total_work_on_crane_bay": sum(
                            int(item["length"])
                            for item in work_blocks_by_crane_bay.get(
                                str(q + 1), []
                            )
                            if int(item["bay"]) == position
                        ),
                        "prefix_completion_end": int(primary_work["start"])
                        + sum(
                            int(item["length"])
                            for item in work_blocks_by_crane_bay.get(
                                str(q + 1), []
                            )
                            if int(item["bay"]) == position
                        ),
                    })
                else:
                    first_work[position] = block
                seen_work[position] = block
            seen_any[position] = block

    # Attach the adjacent conflict chain to each revisit after all crane
    # position blocks are available.  This is diagnostic metadata only; the
    # phase operator still derives its active band from the crane index and
    # safety closure rather than from a hard-coded Q3/bay13 exception.
    for revisit in crane_work_revisits:
        gap_start = int(revisit["primary_block"]["end_exclusive"])
        gap_end = int(revisit["residual_block"]["start"])
        residual_end = int(revisit["residual_block"]["end_exclusive"])
        focus_crane = int(revisit["crane"])
        interfering_blocks = [
            {
                "crane": int(block["crane"]),
                "start": int(block["start"]),
                "end_exclusive": int(block["end_exclusive"]),
                "position": int(block["position"]),
                "state": block["state"],
            }
            for block in crane_position_blocks
            if int(block["crane"]) != focus_crane
            and int(block["start"]) < residual_end
            and int(block["end_exclusive"]) > gap_start
        ]
        revisit["interfering_cranes"] = sorted({
            int(block["crane"]) for block in interfering_blocks
        })
        revisit["interfering_blocks"] = interfering_blocks
        revisit["gap_interval"] = {
            "start": gap_start,
            "end_exclusive": gap_end,
            "length": max(0, gap_end - gap_start),
        }

    terminal_returns = [
        item for item in crane_work_revisits
        if int(item["end_exclusive"]) >= horizon - 8
    ]
    required_starts = {int(bay) for bay in (starts or ())}
    forced_prefix_interruptions: list[dict[str, Any]] = []
    for revisit in crane_work_revisits:
        primary = revisit["primary_block"]
        residual = revisit["residual_block"]
        crane = int(revisit["crane"])
        bay = int(revisit["bay"])
        prefix_start = int(primary["start"])
        prefix_end = int(primary["end_exclusive"])
        gap_start = prefix_end
        gap_end = int(residual["start"])
        prefix_work = int(primary["work_count"])
        residual_work = int(residual["work_count"])
        if (
            bay not in required_starts
            or prefix_start != 0
            or prefix_work > 2
            or int(revisit["gap"]) > 8
            or residual_work < 8
        ):
            continue
        interruption_slots = [
            (time, work_bay)
            for (time, q), work_bay in work_by_time_crane.items()
            if q == crane - 1
            and gap_start <= time < gap_end
            and work_bay != bay
        ]
        if not interruption_slots:
            continue
        forced_prefix_interruptions.append({
            "crane": crane,
            "forced_bay": bay,
            "prefix": [prefix_start, prefix_end],
            "gap": [gap_start, gap_end],
            "residual": [int(residual["start"]), int(residual["end_exclusive"])],
            "interrupt_bays": sorted({int(work_bay) for _, work_bay in interruption_slots}),
            "interrupt_work": len(interruption_slots),
            "merged_length": int(revisit["total_work_on_crane_bay"]),
            "movement_penalty": int(bool(revisit.get("movement_before"))),
        })
    micro_work_blocks = [
        block
        for blocks in work_blocks_by_crane_bay.values()
        for block in blocks
        if int(block["length"]) <= 2
    ]
    long_revisit_count = sum(
        int(item["gap"]) >= 8 for item in crane_work_revisits
    )
    multi_slot_revisits = [
        item for item in crane_work_revisits
        if int(item.get("residual_length", 0)) > 2
    ]
    multi_slot_terminal_revisits = [
        item for item in multi_slot_revisits
        if int(item["end_exclusive"]) >= horizon - 8
    ]
    forced_start_revisits = [
        item for item in crane_work_revisits
        if int(item.get("primary_block", {}).get("start", -1)) == 0
    ]
    max_crane_movement_count = max(
        movement_count_by_crane.values(), default=0
    )
    return_move_count_by_crane = {
        str(q + 1): sum(
            int(item.get("movement_before", False))
            for item in crane_work_revisits
            if int(item["crane"]) == q + 1
        )
        for q in range(M)
    }

    return {
        "bay_fragmentation": bay_fragmentation,
        "work_blocks_by_bay": work_blocks_by_bay,
        "bay_gaps_by_bay": bay_gaps_by_bay,
        "work_blocks_by_crane_bay": work_blocks_by_crane_bay,
        "work_block_count_by_crane_bay": work_block_count_by_crane_bay,
        "extra_work_blocks_total": extra_work_blocks_total,
        "micro_work_blocks": micro_work_blocks,
        "short_excursions": short_excursion_records,
        "short_excursion_count": len(short_excursion_records),
        "terminal_returns": terminal_returns,
        "terminal_return_count": len(terminal_returns),
        "forced_prefix_interruptions": forced_prefix_interruptions,
        "forced_prefix_interruption_count": len(forced_prefix_interruptions),
        "movement_arcs_by_crane": movement_arcs_by_crane,
        "movement_count_by_crane": movement_count_by_crane,
        "return_move_count_by_crane": return_move_count_by_crane,
        "max_crane_movement_count": max_crane_movement_count,
        "crane_position_blocks": crane_position_blocks,
        "position_revisit_count": position_revisit_count,
        "work_revisit_count": work_revisit_count,
        "crane_work_revisits": crane_work_revisits,
        "multi_slot_revisits": multi_slot_revisits,
        "multi_slot_terminal_revisits": multi_slot_terminal_revisits,
        "forced_start_revisits": forced_start_revisits,
        "max_work_revisit_gap": max(
            (int(item["gap"]) for item in crane_work_revisits),
            default=0,
        ),
        "long_revisit_count": long_revisit_count,
        "pure_yielding_revisits": pure_yielding_revisits,
        "movement_count": candidate.movement_count,
        "reversal_count": candidate.reversal_count,
        "load_deviation": candidate.load_deviation,
        "short_excursion": list(_trajectory_smoothness(candidate, M)),
        "continuity_key": [
            work_revisit_count,
            extra_work_blocks_total,
            bay_fragmentation,
            len(short_excursion_records),
            len(terminal_returns),
            candidate.reversal_count,
            candidate.movement_count,
            candidate.load_deviation,
        ],
    }


def _idle_diagnostics(candidate: _CandidateSchedule, M: int) -> dict[str, Any]:
    """Separate waiting, movement, off-rail time, and completed suffixes.

    A fixed-horizon schedule always contains a constant amount of non-work
    capacity.  Therefore this report never treats the total number of idle
    slots as an improvement target.  It identifies idle slots between a
    crane's first and last work slot, which are the only idle slots that may
    be removable through a legal hand-off.
    """
    by_crane: list[list[Slot]] = [[] for _ in range(M)]
    for slot in candidate.slots:
        if 1 <= slot.crane <= M:
            by_crane[slot.crane - 1].append(slot)

    def intervals(values: Sequence[int]) -> list[dict[str, int]]:
        if not values:
            return []
        ordered = sorted(set(values))
        result: list[dict[str, int]] = []
        start = previous = ordered[0]
        for value in ordered[1:] + [None]:
            if value is not None and value == previous + 1:
                previous = value
                continue
            result.append({
                "start": start,
                "end_exclusive": previous + 1,
                "length": previous - start + 1,
            })
            if value is not None:
                start = previous = value
        return result

    details: list[dict[str, Any]] = []
    total_internal = 0
    total_idle = 0
    total_moves = 0
    total_offrail = 0
    max_internal = 0
    total_internal_blocks = 0
    max_internal_blocks = 0
    for q, slots in enumerate(by_crane, 1):
        slots = sorted(slots, key=lambda item: item.time)
        work_times = [slot.time for slot in slots if slot.state == "work"]
        idle_times = [slot.time for slot in slots if slot.state == "idle"]
        move_times = [slot.time for slot in slots if slot.state == "move"]
        offrail_times = [slot.time for slot in slots if slot.state == "offrail"]
        first_work = min(work_times) if work_times else None
        last_work = max(work_times) if work_times else None
        internal_times = (
            [t for t in idle_times if first_work is not None and last_work is not None
             and first_work < t < last_work]
        )
        internal_blocks = intervals(internal_times)
        internal_total = sum(item["length"] for item in internal_blocks)
        max_internal_for_crane = max(
            (item["length"] for item in internal_blocks), default=0
        )
        internal_block_count = len(internal_blocks)
        total_internal += internal_total
        total_internal_blocks += internal_block_count
        max_internal_blocks = max(max_internal_blocks, internal_block_count)
        total_idle += len(idle_times)
        total_moves += len(move_times)
        total_offrail += len(offrail_times)
        max_internal = max(max_internal, max_internal_for_crane)
        details.append({
            "crane": q,
            "first_work": first_work,
            "last_work": last_work,
            "work_slots": len(work_times),
            "leading_idle": len([t for t in idle_times if first_work is not None and t < first_work]),
            "internal_idle": internal_total,
            "internal_idle_blocks": internal_blocks,
            "internal_idle_block_count": internal_block_count,
            "max_internal_idle": max_internal_for_crane,
            "trailing_idle": len([t for t in idle_times if last_work is not None and t > last_work]),
            "idle_slots": len(idle_times),
            "move_slots": len(move_times),
            "offrail_slots": len(offrail_times),
            "state_intervals": {
                "idle": intervals(idle_times),
                "move": intervals(move_times),
                "offrail": intervals(offrail_times),
            },
        })
    return {
        "per_crane": details,
        "total_idle": total_idle,
        "total_internal_idle": total_internal,
        "total_internal_idle_blocks": total_internal_blocks,
        "max_internal_idle_blocks": max_internal_blocks,
        "max_internal_idle": max_internal,
        "total_move_slots": total_moves,
        "total_offrail_slots": total_offrail,
    }


def _balanced_schedule_metrics(
    candidate: _CandidateSchedule, M: int,
) -> dict[str, Any]:
    """Describe genuine work-capacity balance without hiding idle time."""
    completion = int(candidate.completion_time)
    idle = _idle_diagnostics(candidate, M)
    per_crane = idle["per_crane"]
    finishes = [
        int(item["last_work"]) + 1 if item["last_work"] is not None else 0
        for item in per_crane
    ]
    normalized_idle = [
        {
            **item,
            "trailing_idle": max(0, completion - finishes[index]),
        }
        for index, item in enumerate(per_crane)
    ]
    loads = [int(value) for value in candidate.loads]
    total_work = sum(loads)
    nonwork_capacity = [max(0, completion - load) for load in loads]
    balance_deviation = sum(
        (M * load - total_work) ** 2 for load in loads
    )
    leading_internal_idle = sum(
        int(item["leading_idle"]) + int(item["internal_idle"])
        for item in per_crane
    )
    max_trailing_idle = max(
        (int(item["trailing_idle"]) for item in normalized_idle),
        default=0,
    )
    return {
        "completion_time": completion,
        "loads": loads,
        "finish_times": finishes,
        "nonwork_capacity_by_crane": nonwork_capacity,
        "total_nonwork_capacity": sum(nonwork_capacity),
        "max_nonwork_capacity": max(nonwork_capacity, default=0),
        "load_balance_deviation": balance_deviation,
        "finish_gap": max(finishes, default=0) - min(finishes, default=0),
        "max_trailing_idle": max_trailing_idle,
        "max_internal_idle": int(idle["max_internal_idle"]),
        "total_internal_idle_blocks": int(idle["total_internal_idle_blocks"]),
        "max_internal_idle_blocks": int(idle["max_internal_idle_blocks"]),
        "leading_plus_internal_idle": leading_internal_idle,
        "per_crane_idle": normalized_idle,
        "key": (
            completion,
            max(nonwork_capacity, default=0),
            balance_deviation,
            max_trailing_idle,
            max(finishes, default=0) - min(finishes, default=0),
            int(idle["max_internal_idle"]),
            int(candidate.completion_movement_count),
        ),
    }


def _recommended_schedule_rank(
    candidate: _CandidateSchedule, M: int,
) -> tuple[int, ...]:
    """Rank Step 8 output by completion, finish balance and work continuity.

    A repeated ``(crane, bay)`` work block necessarily creates a return move,
    so eliminate those structural path defects before comparing secondary
    load-balance improvements.  Synchronized completion is still protected:
    among candidates with the same revisit burden, a schedule with trailing
    idle cannot beat one in which all cranes finish together.
    """
    balance = _balanced_schedule_metrics(candidate, M)
    continuity = _continuity_diagnostics(candidate, M)
    return (
        int(candidate.completion_time),
        int(continuity["work_revisit_count"]),
        int(balance["max_nonwork_capacity"]),
        int(balance["max_trailing_idle"]),
        int(balance["finish_gap"]),
        int(continuity["position_revisit_count"]),
        int(candidate.reversal_count),
        int(candidate.completion_movement_count),
        int(continuity["extra_work_blocks_total"]),
        int(continuity["max_work_revisit_gap"]),
        int(balance["max_internal_idle_blocks"]),
        int(balance["total_internal_idle_blocks"]),
        int(balance["max_internal_idle"]),
    )


def _candidate_passes_independent_verifier(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    candidate: _CandidateSchedule,
) -> bool:
    """Validate a private candidate through the public schedule verifier."""
    view = type("_CandidateVerifierView", (), {})()
    view.slots = candidate.slots
    view.makespan = candidate.completion_time
    view.schedule_horizon = candidate.schedule_horizon
    view.crane_loads = candidate.loads
    view.reversal_count = candidate.reversal_count
    view.movement_count = candidate.completion_movement_count
    view.schedule_movement_count = candidate.movement_count
    view.move_time = candidate.move_time
    try:
        verify_solution(W, M, starts, view)
    except AssertionError:
        return False
    return True


def _trajectory_segment_neighbors(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    candidate: _CandidateSchedule,
    max_candidates: int = 64,
) -> list[tuple[str, list[tuple[int, ...]]]]:
    """Generate bounded block-level trajectory proposals for polishing.

    These proposals edit complete position blocks rather than isolated random
    cells.  They are deliberately conservative: the caller still rebuilds
    the candidate, checks the fixed horizon, and invokes the independent
    verifier before accepting anything.
    """
    if candidate.move_time != 0 or candidate.makespan < 2:
        return []
    rows = [tuple(row) for row in _candidate_position_rows(candidate, M)]
    proposals: list[tuple[str, list[tuple[int, ...]]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = {tuple(rows)}

    def add(name: str, changed: list[list[int]]) -> None:
        if len(proposals) >= max_candidates:
            return
        history = tuple(tuple(row) for row in changed)
        if history in seen:
            return
        seen.add(history)
        proposals.append((name, [tuple(row) for row in changed]))

    excursions = _short_excursion_details(candidate, M)
    for q, start, end, base, _excursion, _work in excursions:
        changed = [list(row) for row in rows]
        for t in range(start, end + 1):
            changed[t][q] = base
        add("short_visit_eliminate", changed)

    # A simultaneous batch removes adjacent short hand-offs as one coupled
    # proposal, so every row is checked with all participating cranes moved.
    ordered = sorted(excursions, key=lambda item: (item[1], item[2], item[0]))
    clusters: list[list[tuple[int, int, int, int, int, int]]] = []
    for detail in ordered:
        if not clusters or detail[1] > max(item[2] for item in clusters[-1]) + 1:
            clusters.append([detail])
        else:
            clusters[-1].append(detail)
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        changed = [list(row) for row in rows]
        for q, start, end, base, _excursion, _work in cluster:
            for t in range(start, end + 1):
                changed[t][q] = base
        add("continuous_handoff_batch", changed)

    # Slide a complete visit boundary by one row.  A proposal can be rejected
    # later if the row becomes unsafe or loses the fixed horizon.
    for q in range(M):
        for t in range(1, candidate.makespan):
            if rows[t - 1][q] == rows[t][q]:
                continue
            delayed = [list(row) for row in rows]
            delayed[t][q] = rows[t - 1][q]
            add("visit_boundary_slide_delay", delayed)
            if t > 1:
                advanced = [list(row) for row in rows]
                advanced[t - 1][q] = rows[t][q]
                add("visit_boundary_slide_advance", advanced)

    # Merge separated blocks at the same position by filling the bounded gap
    # with that position.  This is an interval proposal, not a point edit.
    for q in range(M):
        blocks: list[tuple[int, int, int]] = []
        start = 0
        while start < candidate.makespan:
            end = start
            while end + 1 < len(rows) and rows[end + 1][q] == rows[start][q]:
                end += 1
            blocks.append((start, min(end, candidate.makespan - 1), rows[start][q]))
            start = end + 1
        for left_index, left in enumerate(blocks):
            for right in blocks[left_index + 1:]:
                if left[2] != right[2] or right[0] <= left[1] + 1:
                    continue
                if right[0] - left[1] > 8:
                    break
                changed = [list(row) for row in rows]
                for t in range(left[1] + 1, right[0]):
                    changed[t][q] = left[2]
                add("same_bay_segment_merge", changed)

    # For split bays, first try removing only a short visit by one of the
    # owners.  This keeps the ownership simplification local and lets the
    # decoder reassign the released work to an existing compatible visit.
    split_bays = {
        bay for bay, owners in enumerate(candidate.owners, 1)
        if len(owners) > 1
    }
    work_at = {
        (slot.time, slot.crane - 1, int(slot.work_bay))
        for slot in candidate.slots
        if slot.state == "work" and slot.work_bay is not None
    }
    for q, start, end, base, excursion, _work in excursions:
        if excursion not in split_bays:
            continue
        if not any((t, q, excursion) in work_at for t in range(start, end + 1)):
            continue
        changed = [list(row) for row in rows]
        for t in range(start, end + 1):
            changed[t][q] = base
        add("split_bay_simplify", changed)

    return proposals


def _history_diff_regions(
    before: Sequence[tuple[int, ...]],
    after: Sequence[tuple[int, ...]],
) -> list[dict[str, Any]]:
    """Return contiguous changed rows, grouped by crane.

    The records are deliberately expressed in absolute position-row indices so
    a caller can verify the local-window contract without looking at a plot.
    """
    if len(before) != len(after):
        raise ValueError("轨迹提案的时间轴长度不能改变。")
    regions: list[dict[str, Any]] = []
    width = len(before[0]) if before else 0
    for q in range(width):
        changed = [
            t for t, (left, right) in enumerate(zip(before, after))
            if left[q] != right[q]
        ]
        start = 0
        while start < len(changed):
            end = start
            while end + 1 < len(changed) and changed[end + 1] == changed[end] + 1:
                end += 1
            first = changed[start]
            last = changed[end] + 1
            regions.append({
                "crane": q + 1,
                "start": first,
                "end_exclusive": last,
                "length": last - first,
                "changed_rows": list(range(first, last)),
            })
            start = end + 1
    return regions


def _candidate_from_history_with_frozen_work(
    W: Sequence[int],
    M: int,
    history: Sequence[tuple[int, ...]],
    source: _CandidateSchedule,
    declared_regions: Sequence[dict[str, Any]],
) -> _CandidateSchedule:
    """Decode a paired local proposal while freezing work outside its regions.

    Position-only decoding is useful for broad trajectory search, but it can
    silently reassign work in rows that the proposal did not declare.  This
    decoder keeps every source work slot outside the declared crane/time
    regions and lets only the local cells absorb the remaining work.  It is a
    zero-time, fixed-horizon transaction by design.
    """
    if source.move_time != 0 or len(history) != source.makespan + 1:
        raise RuntimeError("冻结工作事务只支持零移动时间的固定工期。")
    if len(history) != len(_candidate_position_rows(source, M)):
        raise RuntimeError("冻结工作事务的时间轴长度不一致。")
    source_rows = _candidate_position_rows(source, M)
    proposed_rows = [tuple(int(value) for value in row) for row in history]
    if any(len(row) != M for row in proposed_rows):
        raise RuntimeError("冻结工作事务的桥吊数量不一致。")
    horizon = source.makespan
    editable: set[tuple[int, int]] = set()
    for region in declared_regions:
        q = int(region.get("crane", 0)) - 1
        if not 0 <= q < M:
            raise RuntimeError("冻结工作事务包含非法桥吊。")
        start = max(0, int(region.get("start", 0)))
        end = min(horizon, int(region.get("end_exclusive", start)))
        if end - start > 8:
            raise RuntimeError("冻结工作事务超过8个时间单位。")
        editable.update((t, q) for t in range(start, end))
    for t in range(horizon):
        for q in range(M):
            if proposed_rows[t][q] != source_rows[t][q] and (t, q) not in editable:
                raise RuntimeError("候选修改超出声明工作区域。")

    source_work: dict[tuple[int, int], int] = {
        (slot.time, slot.crane - 1): int(slot.work_bay)
        for slot in source.slots
        if slot.state == "work" and slot.work_bay is not None
    }
    fixed_future = [0] * len(W)
    for (t, q), bay in source_work.items():
        if (t, q) not in editable:
            fixed_future[bay - 1] += 1
    remaining = list(W)
    work_plan: dict[tuple[int, int], int] = {}
    for t in range(horizon):
        for q in range(M):
            key = (t, q)
            if key not in editable:
                source_bay = source_work.get(key)
                if source_bay is not None:
                    if proposed_rows[t][q] != source_bay:
                        raise RuntimeError("窗口外源作业位置发生变化。")
                    if remaining[source_bay - 1] <= 0:
                        raise RuntimeError("固定作业量重复使用。")
                    remaining[source_bay - 1] -= 1
                    fixed_future[source_bay - 1] -= 1
                    work_plan[key] = source_bay
                elif proposed_rows[t][q] != source_rows[t][q]:
                    raise RuntimeError("窗口外空闲槽位置发生变化。")
                continue
            bay = proposed_rows[t][q]
            if 1 <= bay <= len(W) and remaining[bay - 1] > fixed_future[bay - 1]:
                remaining[bay - 1] -= 1
                work_plan[key] = bay
    if any(remaining):
        raise RuntimeError("局部冻结工作事务无法守恒全部作业。")

    slots: list[Slot] = []
    owners: list[set[int]] = [set() for _ in W]
    loads = [0] * M
    movement_directions: list[list[int]] = [[] for _ in range(M)]
    for t, (positions, next_positions) in enumerate(zip(proposed_rows, proposed_rows[1:])):
        for q, (start_bay, end_bay) in enumerate(zip(positions, next_positions)):
            if start_bay != end_bay:
                movement_directions[q].append(1 if end_bay > start_bay else -1)
            bay = work_plan.get((t, q))
            if bay is not None:
                owners[bay - 1].add(q)
                loads[q] += 1
                slots.append(Slot(t, q + 1, "work", bay, bay, bay))
            elif not 1 <= start_bay <= len(W):
                slots.append(Slot(t, q + 1, "offrail", start_bay, start_bay, None))
            else:
                slots.append(Slot(t, q + 1, "idle", start_bay, start_bay, None))
    movement_count = sum(
        before != after
        for row_before, row_after in zip(proposed_rows, proposed_rows[1:])
        for before, after in zip(row_before, row_after)
    )
    reversal_count = sum(
        previous != current
        for directions in movement_directions
        for previous, current in zip(directions, directions[1:])
    )
    target_weights = [min(q + 1, M - q) for q in range(M)]
    return _CandidateSchedule(
        slots=slots,
        makespan=horizon,
        assignment_count=sum(len(item) for item in owners),
        split_bay_count=sum(len(item) > 1 for item in owners),
        load_deviation=_load_deviation(loads, target_weights, sum(W)),
        reversal_count=reversal_count,
        movement_count=movement_count,
        loads=loads,
        owners=owners,
        move_time=0,
    )


def _candidate_from_rows_and_work_plan(
    W: Sequence[int],
    M: int,
    history: Sequence[tuple[int, ...]],
    work_plan: dict[tuple[int, int], int | None],
) -> _CandidateSchedule:
    """Build a zero-time candidate from an explicit position/work ledger.

    This is deliberately separate from :func:`_candidate_from_history`.  The
    latter is a useful construction decoder, but it is allowed to assign the
    next unfinished bay greedily at a position.  A Step 8 transaction must
    never do that: every cell in the local transaction has an explicit
    ``work_bay`` (or ``None`` for idle/offrail), and the complete ledger is
    checked before a candidate is returned.
    """
    if len(history) < 2:
        raise RuntimeError("显式工作事务至少需要一个时间槽。")
    horizon = len(history) - 1
    rows = [tuple(int(value) for value in row) for row in history]
    if any(len(row) != M for row in rows):
        raise RuntimeError("显式工作事务的桥吊数量不一致。")
    expected_keys = {
        (t, q) for t in range(horizon) for q in range(M)
    }
    if set(work_plan) != expected_keys:
        raise RuntimeError("显式工作账本必须覆盖每个时间槽和桥吊。")
    for row in rows:
        if any(right - left < 2 for left, right in zip(row, row[1:])):
            raise RuntimeError("显式工作事务违反桥吊安全间距。")

    owners: list[set[int]] = [set() for _ in W]
    loads = [0] * M
    movement_directions: list[list[int]] = [[] for _ in range(M)]
    slots: list[Slot] = []
    for t in range(horizon):
        positions = rows[t]
        next_positions = rows[t + 1]
        for q, (start_bay, end_bay) in enumerate(
            zip(positions, next_positions)
        ):
            bay = work_plan[(t, q)]
            if bay is not None:
                bay = int(bay)
                if not 1 <= bay <= len(W):
                    raise RuntimeError("显式工作账本包含非法贝位。")
                if start_bay != bay:
                    raise RuntimeError(
                        "显式工作账本与桥吊位置不一致："
                        f"t={t}, Q{q + 1}, position={start_bay}, work={bay}。"
                    )
                if bay in {
                    other.work_bay for other in slots
                    if other.time == t and other.state == "work"
                }:
                    raise RuntimeError(f"t={t} 同一贝位被重复作业。")
                owners[bay - 1].add(q)
                loads[q] += 1
                slots.append(Slot(t, q + 1, "work", bay, bay, bay))
            elif 1 <= start_bay <= len(W):
                slots.append(Slot(t, q + 1, "idle", start_bay, start_bay, None))
            else:
                slots.append(Slot(t, q + 1, "offrail", start_bay, start_bay, None))

    if loads is None or any(
        sum(
            slot.state == "work" and slot.work_bay == bay
            for slot in slots
        ) != required
        for bay, required in enumerate(W, 1)
    ):
        raise RuntimeError("显式工作账本没有守恒全部作业量。")
    # ``verify_solution`` counts visible zero-time moves only between two
    # emitted work slots.  A final row is an end boundary, not another
    # visible slot, so a move into row H is intentionally not counted here.
    for before_row, after_row in zip(rows, rows[1:horizon]):
        for q, (before, after) in enumerate(zip(before_row, after_row)):
            if before != after:
                movement_directions[q].append(1 if after > before else -1)
    movement_count = sum(
        before != after
        for before_row, after_row in zip(rows, rows[1:horizon])
        for before, after in zip(before_row, after_row)
    )
    reversal_count = sum(
        previous != current
        for directions in movement_directions
        for previous, current in zip(directions, directions[1:])
    )
    target_weights = [min(q + 1, M - q) for q in range(M)]
    return _CandidateSchedule(
        slots=slots,
        makespan=horizon,
        assignment_count=sum(len(item) for item in owners),
        split_bay_count=sum(len(item) > 1 for item in owners),
        load_deviation=_load_deviation(loads, target_weights, sum(W)),
        reversal_count=reversal_count,
        movement_count=movement_count,
        loads=loads,
        owners=owners,
        move_time=0,
    )


def _normalize_work_transfer_regions(
    regions: Sequence[dict[str, Any]],
    horizon: int,
    M: int,
    *,
    max_segments: int = 2,
    max_region_length: int = 8,
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    """Validate the strict Step 8 local-transaction range contract."""
    if not regions:
        raise RuntimeError("工作事务没有声明局部区域。")
    by_segment: dict[int, tuple[int, int]] = {}
    normalized: list[dict[str, Any]] = []
    for raw in regions:
        try:
            q = int(raw["crane"]) - 1
            start = int(raw["start"])
            end = int(raw["end_exclusive"])
            segment = int(raw.get("segment", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("工作事务区域字段不完整。") from exc
        if not 0 <= q < M:
            raise RuntimeError("工作事务包含非法桥吊。")
        if not 0 <= start < end <= horizon:
            raise RuntimeError("工作事务时间区域超出有效范围。")
        if end - start > max_region_length:
            raise RuntimeError(
                f"单个工作事务区域不能超过{max_region_length}个时间槽。"
            )
        old = by_segment.get(segment)
        if old is not None and old != (start, end):
            raise RuntimeError("同一事务段的桥吊区域必须使用相同时间边界。")
        by_segment[segment] = (start, end)
        normalized.append({
            "crane": q + 1,
            "start": start,
            "end_exclusive": end,
            "length": end - start,
            "segment": segment,
        })
    if len(by_segment) > max_segments:
        raise RuntimeError(
            f"一个工作事务最多包含{max_segments}个时间区段。"
        )
    intervals = sorted(by_segment.values())
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise RuntimeError("工作事务区段不能重叠。")
    active = sorted({item["crane"] - 1 for item in normalized})
    if len(active) > 3 or active != list(range(active[0], active[-1] + 1)):
        raise RuntimeError("工作事务最多使用三台相邻桥吊。")
    editable = {
        (t, q)
        for item in normalized
        for t in range(item["start"], item["end_exclusive"])
        for q in (item["crane"] - 1,)
    }
    return normalized, editable


def _candidate_from_explicit_work_transaction(
    W: Sequence[int],
    M: int,
    history: Sequence[tuple[int, ...]],
    source: _CandidateSchedule,
    work_plan: dict[tuple[int, int], int | None],
    regions: Sequence[dict[str, Any]],
    *,
    source_hash: str | None = None,
    transaction_source_hash: str | None = None,
    max_segments: int = 4,
    max_region_length: int = 8,
) -> _CandidateSchedule:
    """Apply one complete local work/position transaction.

    ``source_hash`` is the current source artifact hash and
    ``transaction_source_hash`` is the hash captured when the proposal was
    generated.  The solver has no filesystem dependency, so the complete
    source trajectory signature remains the in-memory binding as well.  The
    returned candidate is still independently verified by the caller with
    the real ``S``.
    """
    if transaction_source_hash != source_hash:
        raise RuntimeError("工作事务 source_hash 与当前源方案不匹配。")
    if source.move_time != 0:
        raise RuntimeError("严格工作事务暂只支持 move_time=0。")
    if len(history) != source.makespan + 1:
        raise RuntimeError("工作事务不能改变固定H的时间轴。")
    normalized, editable = _normalize_work_transfer_regions(
        regions, source.makespan, M,
        max_segments=max_segments,
        max_region_length=max_region_length,
    )
    source_rows = [tuple(row) for row in _candidate_position_rows(source, M)]
    rows = [tuple(int(value) for value in row) for row in history]
    if len(rows) != len(source_rows) or any(
        len(row) != M for row in rows
    ):
        raise RuntimeError("工作事务轨迹尺寸不一致。")
    expected_keys = {
        (t, q) for t in range(source.makespan) for q in range(M)
    }
    if set(work_plan) != expected_keys:
        raise RuntimeError("工作事务账本没有覆盖完整时间轴。")
    source_work = {
        (slot.time, slot.crane - 1): int(slot.work_bay)
        for slot in source.slots
        if slot.state == "work" and slot.work_bay is not None
    }

    # A row is allowed to change only when one of its incident slots belongs
    # to the declared transaction.  This makes boundary moves explicit while
    # preserving every field of all outside slots.
    allowed_rows = {
        (row_t, q)
        for slot_t, q in editable
        for row_t in (slot_t, slot_t + 1)
    }
    for t, (before, after) in enumerate(zip(source_rows, rows)):
        for q, (left, right) in enumerate(zip(before, after)):
            if left != right and (t, q) not in allowed_rows:
                raise RuntimeError(
                    f"位置变化超出声明事务区域：t={t}, Q{q + 1}。"
                )
    for q, (left, right) in enumerate(zip(source_rows[-1], rows[-1])):
        if left != right and (source.makespan - 1, q) not in editable:
            raise RuntimeError(f"末端边界位置变化未声明：Q{q + 1}。")

    for t in range(source.makespan):
        for q in range(M):
            key = (t, q)
            if key not in editable and work_plan[key] != source_work.get(key):
                raise RuntimeError(
                    f"窗口外作业发生变化：t={t}, Q{q + 1}, "
                    f"source={source_work.get(key)}, new={work_plan[key]}。"
                )

    candidate = _candidate_from_rows_and_work_plan(W, M, rows, work_plan)
    source_slot_map = {(slot.time, slot.crane - 1): slot for slot in source.slots}
    candidate_slot_map = {
        (slot.time, slot.crane - 1): slot for slot in candidate.slots
    }
    for t in range(source.makespan):
        for q in range(M):
            if (t, q) in editable:
                continue
            before = source_slot_map[(t, q)]
            after = candidate_slot_map[(t, q)]
            if (
                before.state, before.start_bay, before.end_bay, before.work_bay,
                before.move_id, before.move_step, before.move_steps,
            ) != (
                after.state, after.start_bay, after.end_bay, after.work_bay,
                after.move_id, after.move_step, after.move_steps,
            ):
                raise RuntimeError(
                    f"窗口外槽位字段发生变化：t={t}, Q{q + 1}。"
                )
    return candidate


def _candidate_from_explicit_phase_transaction(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    history: Sequence[tuple[int, ...]],
    source: _CandidateSchedule,
    work_plan: dict[tuple[int, int], int | None],
    active_cranes: Sequence[int],
    *,
    source_hash: str | None = None,
    transaction_source_hash: str | None = None,
) -> _CandidateSchedule:
    """Decode a full-horizon, block-level phase transaction.

    Local Step 8 transactions deliberately cap each edited region at eight
    slots.  A long residual revisit cannot be represented by that contract:
    its work phases and the safety-conflict chain may span the complete
    horizon.  This decoder keeps the same explicit work-ledger guarantees but
    freezes every inactive crane for the whole horizon.
    """
    if transaction_source_hash != source_hash:
        raise RuntimeError("阶段事务 source_hash 与当前源方案不匹配。")
    if source.move_time != 0:
        raise RuntimeError("阶段事务暂只支持 move_time=0。")
    if len(history) != source.makespan + 1:
        raise RuntimeError("阶段事务不能改变固定H的时间轴。")
    active = {int(q) for q in active_cranes}
    if not active or any(q < 0 or q >= M for q in active):
        raise RuntimeError("阶段事务 active_cranes 非法。")
    source_rows = [tuple(row) for row in _candidate_position_rows(source, M)]
    rows = [tuple(int(value) for value in row) for row in history]
    if len(rows) != len(source_rows) or any(len(row) != M for row in rows):
        raise RuntimeError("阶段事务轨迹尺寸不一致。")
    expected_keys = {
        (t, q) for t in range(source.makespan) for q in range(M)
    }
    if set(work_plan) != expected_keys:
        raise RuntimeError("阶段事务账本没有覆盖完整时间轴。")
    source_work = {
        (slot.time, slot.crane - 1): int(slot.work_bay)
        for slot in source.slots
        if slot.state == "work" and slot.work_bay is not None
    }
    for t in range(source.makespan):
        for q in range(M):
            if q in active:
                continue
            if rows[t][q] != source_rows[t][q]:
                raise RuntimeError(
                    f"阶段事务修改了非 active crane：t={t}, Q{q + 1}。"
                )
            if work_plan[(t, q)] != source_work.get((t, q)):
                raise RuntimeError(
                    f"阶段事务改变了非 active crane 作业：t={t}, Q{q + 1}。"
                )
    # The final row is a boundary row.  It is still frozen for inactive
    # cranes so an active phase cannot hide a terminal position change.
    for q in range(M):
        if q not in active and rows[-1][q] != source_rows[-1][q]:
            raise RuntimeError(f"阶段事务末端修改了非 active crane：Q{q + 1}。")
    candidate = _candidate_from_rows_and_work_plan(W, M, rows, work_plan)
    if not _candidate_passes_independent_verifier(W, M, starts, candidate):
        raise RuntimeError("阶段事务未通过独立 verifier。")
    return candidate


def _transaction_rows_work_map(
    candidate: _CandidateSchedule,
    M: int,
) -> tuple[list[tuple[int, ...]], dict[tuple[int, int], int | None]]:
    rows = [tuple(row) for row in _candidate_position_rows(candidate, M)]
    work = {
        (t, q): None
        for t in range(candidate.makespan)
        for q in range(M)
    }
    for slot in candidate.slots:
        if slot.state == "work" and slot.work_bay is not None:
            work[(slot.time, slot.crane - 1)] = int(slot.work_bay)
    return rows, work


def _forced_prefix_consolidation_exchange(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    source: _CandidateSchedule,
    *,
    max_neighbor_phase_permutations: int = 64,
    max_focus_phase_permutations: int = 16,
    max_idle_placements: int = 16,
    state_limit: int = 4_096,
    max_candidates: int = 32,
    deadline: float | None = None,
    source_hash: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge an interrupted forced-start work prefix using adjacent phases.

    The focus crane's forced-start bay is merged into one indivisible phase.
    One adjacent crane at a time is allowed to reorder its own complete bay
    phases to make room; every other crane is frozen.  The event search only
    branches when a phase can start or wait at its current safe position.
    """
    stats: dict[str, Any] = {
        "status": "SEARCH_COMPLETE",
        "focus_interruptions": 0,
        "focuses_started": 0,
        "focuses_completed": 0,
        "neighbor_bands_generated": 0,
        "focus_phase_orders": 0,
        "neighbor_phase_orders": 0,
        "idle_placements_tested": 0,
        "states_expanded": 0,
        "ledger_closed": 0,
        "decoded": 0,
        "verified": 0,
        "accepted": 0,
        "rejected_safety": 0,
        "rejected_ledger": 0,
        "rejected_burden_migration": 0,
        "timeout": 0,
        "state_limit": 0,
    }
    if source.move_time != 0:
        stats.update({"status": "UNSUPPORTED", "reason": "nonzero_move_time"})
        return [], stats
    if state_limit <= 0 or max_candidates <= 0:
        stats.update({"status": "UNKNOWN_STATE_LIMIT", "state_limit": 1})
        return [], stats

    horizon = source.makespan
    source_rows, source_work = _transaction_rows_work_map(source, M)
    report = _continuity_diagnostics(source, M, starts)
    focus_items = sorted(
        report["forced_prefix_interruptions"],
        key=lambda item: (
            int(item["movement_penalty"]),
            item["gap"][1] - item["gap"][0],
            item["prefix"][1] - item["prefix"][0],
            -(item["residual"][1] - item["residual"][0]),
            int(item["crane"]),
            int(item["forced_bay"]),
        ),
    )
    stats["focus_interruptions"] = len(focus_items)
    if not focus_items:
        stats["status"] = "NOT_APPLICABLE"
        return [], stats

    def expired() -> bool:
        if deadline is not None and time.perf_counter() >= deadline:
            stats["status"] = "UNKNOWN_DEADLINE"
            stats["timeout"] += 1
            return True
        return False

    def phase_units(crane: int, forced_bay: int | None = None) -> tuple[tuple[int, int], ...]:
        order: list[int] = []
        lengths: dict[int, int] = {}
        for time_index in range(horizon):
            bay = source_work.get((time_index, crane))
            if bay is None:
                continue
            bay = int(bay)
            if bay not in lengths:
                order.append(bay)
                lengths[bay] = 0
            lengths[bay] += 1
        first_bay = source_work.get((0, crane))
        if first_bay is None:
            return ()
        if forced_bay is not None:
            lengths[forced_bay] = sum(
                1 for time_index in range(horizon)
                if source_work.get((time_index, crane)) == forced_bay
            )
        order = [int(first_bay), *[bay for bay in order if bay != first_bay]]
        return tuple((bay, lengths[bay]) for bay in order if lengths.get(bay, 0) > 0)

    def bounded_orders(
        units: tuple[tuple[int, int], ...], limit: int,
    ) -> list[tuple[tuple[int, int], ...]]:
        if not units:
            return []
        first, rest = units[0], list(units[1:])
        start = tuple(rest)
        queue = [start]
        seen: set[tuple[tuple[int, int], ...]] = {start}
        result: list[tuple[tuple[int, int], ...]] = []
        while queue and len(result) < max(1, limit):
            permutation = queue.pop(0)
            result.append((first, *permutation))
            for index in range(len(permutation) - 1):
                swapped = list(permutation)
                swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
                value = tuple(swapped)
                if value not in seen:
                    seen.add(value)
                    queue.append(value)
        return result

    proposals: list[dict[str, Any]] = []
    seen_proposals: set[tuple[Any, ...]] = set()
    focus_state_budget = max(1, state_limit // len(focus_items))
    for focus in focus_items:
        if expired():
            break
        stats["focuses_started"] += 1
        focus_q = int(focus["crane"]) - 1
        focus_bay = int(focus["forced_bay"])
        neighbor_cranes = [q for q in (focus_q - 1, focus_q + 1) if 0 <= q < M]
        if not neighbor_cranes:
            continue
        focus_orders = bounded_orders(
            phase_units(focus_q, focus_bay), max_focus_phase_permutations,
        )
        if not focus_orders:
            continue
        stats["focus_phase_orders"] += len(focus_orders)
        used_for_focus = 0

        for neighbor_q in neighbor_cranes:
            if expired() or len(proposals) >= max_candidates:
                break
            neighbor_units = phase_units(neighbor_q)
            neighbor_orders = bounded_orders(
                neighbor_units, max_neighbor_phase_permutations,
            )
            if not neighbor_orders:
                continue
            stats["neighbor_bands_generated"] += 1
            stats["neighbor_phase_orders"] += len(neighbor_orders)
            active = tuple(sorted((focus_q, neighbor_q)))
            all_order_pairs = []
            for focus_order in focus_orders:
                for neighbor_order in neighbor_orders:
                    by_crane = {
                        focus_q: focus_order,
                        neighbor_q: neighbor_order,
                    }
                    all_order_pairs.append(tuple(by_crane[q] for q in active))

            for phase_pair in all_order_pairs:
                if expired() or len(proposals) >= max_candidates:
                    break
                if used_for_focus >= focus_state_budget:
                    stats["state_limit"] += 1
                    stats["status"] = "UNKNOWN_STATE_LIMIT"
                    break
                phases = {q: phase_pair[index] for index, q in enumerate(active)}
                first_row = tuple(source_rows[0])
                valid_start = all(
                    phases[q]
                    and int(phases[q][0][0]) == int(first_row[q])
                    for q in active
                )
                if not valid_start:
                    stats["rejected_safety"] += 1
                    continue
                initial_work = dict(source_work)
                initial_remaining = []
                initial_indices = []
                initial_positions = []
                for q in active:
                    first_bay, first_length = phases[q][0]
                    initial_work[(0, q)] = int(first_bay)
                    initial_indices.append(0)
                    initial_remaining.append(int(first_length) - 1)
                    initial_positions.append(int(first_bay))

                # The initial row is fixed by the source and all forced work
                # remains at its original start bay.
                if any(
                    right - left < 2
                    for left, right in zip(first_row, first_row[1:])
                ):
                    stats["rejected_safety"] += 1
                    continue
                initial_working = [
                    value for (time_index, _q), value in initial_work.items()
                    if time_index == 0 and value is not None
                ]
                if len(initial_working) != len(set(initial_working)):
                    stats["rejected_safety"] += 1
                    continue

                failed: set[tuple[Any, ...]] = set()
                memo: dict[tuple[Any, ...], tuple[Any, ...] | None] = {}
                budget_end = min(state_limit, stats["states_expanded"] + focus_state_budget - used_for_focus)

                def visit(
                    time_index: int,
                    phase_indices: tuple[int, ...],
                    remaining: tuple[int, ...],
                    positions: tuple[int, ...],
                    idle_used: int,
                ) -> tuple[Any, ...] | None:
                    nonlocal used_for_focus
                    if expired():
                        return None
                    if time_index >= horizon:
                        if all(
                            phase_indices[index] == len(phases[q]) - 1
                            and remaining[index] == 0
                            for index, q in enumerate(active)
                        ):
                            return ()
                        return None
                    key = (time_index, phase_indices, remaining, positions, idle_used)
                    if key in memo:
                        return memo[key]
                    if key in failed:
                        return None
                    if stats["states_expanded"] >= budget_end:
                        stats["status"] = "UNKNOWN_STATE_LIMIT"
                        stats["state_limit"] += 1
                        return None
                    stats["states_expanded"] += 1
                    used_for_focus += 1
                    pending = 0
                    for index, q in enumerate(active):
                        crane_pending = int(remaining[index]) + sum(
                            int(length)
                            for _bay, length in phases[q][phase_indices[index] + 1:]
                        )
                        pending = max(pending, crane_pending)
                    if pending > horizon - time_index:
                        failed.add(key)
                        return None

                    options_by_crane: list[list[tuple[int, int, int, int | None, int, int]]] = []
                    for index, q in enumerate(active):
                        phase_index = phase_indices[index]
                        slots_left = remaining[index]
                        if slots_left > 0:
                            bay = int(phases[q][phase_index][0])
                            options_by_crane.append([(
                                phase_index, slots_left - 1, bay, bay, 0, 0,
                            )])
                        elif phase_index + 1 < len(phases[q]):
                            next_bay, next_length = phases[q][phase_index + 1]
                            options_by_crane.append([
                                (phase_index + 1, int(next_length) - 1, int(next_bay), int(next_bay), 0, 1),
                                (phase_index, 0, int(positions[index]), None, 1, 0),
                            ])
                        else:
                            options_by_crane.append([(
                                phase_index, 0, int(positions[index]), None, 0, 0,
                            )])

                    choices = list(itertools.product(*options_by_crane))
                    choices.sort(key=lambda combo: (
                        -sum(int(option[5]) for option in combo),
                        sum(int(option[4]) for option in combo),
                        tuple(int(option[2]) for option in combo),
                    ))
                    source_row = source_rows[time_index]
                    for combo in choices:
                        next_idle = idle_used + sum(int(option[4]) for option in combo)
                        if next_idle > max_idle_placements:
                            continue
                        row = list(source_row)
                        work_values = [source_work.get((time_index, q)) for q in range(M)]
                        next_indices = []
                        next_remaining = []
                        next_positions = []
                        starts_now = 0
                        for index, (q, option) in enumerate(zip(active, combo)):
                            phase_index, slots_left, position, work_bay, _wait, started = option
                            row[q] = int(position)
                            work_values[q] = work_bay
                            next_indices.append(int(phase_index))
                            next_remaining.append(int(slots_left))
                            next_positions.append(int(position))
                            starts_now += int(started)
                        if any(
                            right - left < 2
                            for left, right in zip(row, row[1:])
                        ):
                            stats["rejected_safety"] += 1
                            continue
                        active_working = [bay for bay in work_values if bay is not None]
                        if len(active_working) != len(set(active_working)):
                            stats["rejected_safety"] += 1
                            continue
                        if any(int(option[4]) for option in combo):
                            stats["idle_placements_tested"] += 1
                        suffix = visit(
                            time_index + 1,
                            tuple(next_indices),
                            tuple(next_remaining),
                            tuple(next_positions),
                            next_idle,
                        )
                        if suffix is not None:
                            action = (
                                tuple(int(option[2]) for option in combo),
                                tuple(option[3] for option in combo),
                            )
                            result = (action, *suffix)
                            memo[key] = result
                            return result
                        if stats["status"] in {"UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT"}:
                            return None
                    failed.add(key)
                    memo[key] = None
                    return None

                suffix = visit(
                    1,
                    tuple(initial_indices),
                    tuple(initial_remaining),
                    tuple(initial_positions),
                    0,
                )
                if suffix is None:
                    if stats["status"] in {"UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT"}:
                        break
                    continue

                history: list[tuple[int, ...]] = [first_row]
                work_plan = dict(source_work)
                for q in active:
                    work_plan[(0, q)] = int(phases[q][0][0])
                for offset, (active_positions, active_work) in enumerate(suffix, start=1):
                    row = list(source_rows[offset])
                    for index, q in enumerate(active):
                        row[q] = int(active_positions[index])
                        work_plan[(offset, q)] = active_work[index]
                    history.append(tuple(row))
                final_row = list(source_rows[horizon])
                if suffix:
                    for index, q in enumerate(active):
                        final_row[q] = int(suffix[-1][0][index])
                history.append(tuple(final_row))
                signature = (
                    tuple(history),
                    tuple(sorted(work_plan.items())),
                )
                if signature in seen_proposals:
                    continue
                seen_proposals.add(signature)

                moves = sum(
                    history[time_index][q] != history[time_index - 1][q]
                    for time_index in range(1, horizon)
                    for q in range(M)
                )
                details = {
                    "focus": {
                        "crane": int(focus["crane"]),
                        "bay": focus_bay,
                        "prefix": list(focus["prefix"]),
                        "gap": list(focus["gap"]),
                        "residual": list(focus["residual"]),
                        "interrupt_bays": list(focus["interrupt_bays"]),
                        "interrupt_work": int(focus["interrupt_work"]),
                    },
                    "active_cranes": [q + 1 for q in active],
                    "phase_orders": {
                        str(q + 1): [
                            {"bay": int(bay), "length": int(length)}
                            for bay, length in phases[q]
                        ]
                        for q in active
                    },
                    "moves": int(moves),
                    "idle_slots": int(sum(
                        action[1][index] is None
                        for action in suffix
                        for index in range(len(active))
                    )),
                    "operator": "forced_prefix_consolidation",
                    "source_hash": source_hash,
                    "ledger_closed": True,
                }
                proposals.append({
                    "operator": "forced_prefix_consolidation",
                    "history": history,
                    "work_plan": work_plan,
                    "active_cranes": list(active),
                    "regions": [{
                        "crane": q + 1,
                        "start": 0,
                        "end_exclusive": horizon,
                        "length": horizon,
                        "segment": 0,
                    } for q in active],
                    "source_signature": _trajectory_signature(source),
                    "source_hash": source_hash,
                    "details": details,
                })
                stats["ledger_closed"] += 1
                stats["focuses_completed"] = max(stats["focuses_completed"], 1)
                # Once a complete, lower-movement schedule is available, it
                # is a strict formal improvement at fixed H: phase ordering
                # preserves each crane's work ledger and bay ownership.  Do
                # not spend the remaining state budget generating redundant
                # alternatives before the independent verifier can assess it.
                if moves < source.movement_count:
                    stats["status"] = "CANDIDATE_FOUND"
                    break
            if stats["status"] in {
                "UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT", "CANDIDATE_FOUND",
            }:
                break
        if stats["status"] in {
            "UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT", "CANDIDATE_FOUND",
        }:
            break

    proposals.sort(key=lambda item: (
        int(item["details"]["moves"]),
        len(item["details"]["phase_orders"]),
        int(item["details"]["idle_slots"]),
        tuple(
            tuple(phase["bay"] for phase in item["details"]["phase_orders"][str(q)])
            for q in item["details"]["active_cranes"]
        ),
    ))
    return proposals[:max_candidates], stats


def _work_blocks_for_transaction(
    work_plan: dict[tuple[int, int], int | None],
    M: int,
    horizon: int,
) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    for q in range(M):
        t = 0
        while t < horizon:
            bay = work_plan.get((t, q))
            if bay is None:
                t += 1
                continue
            start = t
            t += 1
            while t < horizon and work_plan.get((t, q)) == bay:
                t += 1
            blocks.append({
                "crane": q,
                "bay": int(bay),
                "start": start,
                "end": t,
                "length": t - start,
            })
    return blocks


def _phase_block_resequence_exchange(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    source: _CandidateSchedule,
    *,
    max_active_cranes: int = 4,
    max_block_permutations: int = 256,
    state_limit: int = 20_000,
    max_candidates: int = 64,
    deadline: float | None = None,
    source_hash: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search complete work phases for a long crane/bay revisit.

    The ordinary continuity operators edit one or two short windows.  This
    operator is intentionally phase based: it aggregates each active
    crane/bay's work into an indivisible phase, enumerates a bounded set of
    phase orders, and schedules those phases with a small event-level beam.
    Inactive cranes and their work ledger remain frozen.  The returned
    proposals are decoded by ``_candidate_from_explicit_phase_transaction``.
    """
    stats: dict[str, Any] = {
        "status": "SEARCH_COMPLETE",
        "focus_revisits": 0,
        "multi_slot_revisits": 0,
        "active_bands_generated": 0,
        "phase_permutations_generated": 0,
        "phase_combinations_tested": 0,
        "states_expanded": 0,
        "states_pruned_horizon": 0,
        "states_pruned_safety": 0,
        "states_pruned_split": 0,
        "states_pruned_movement": 0,
        "complete_phase_plans": 0,
        "decoded": 0,
        "verified": 0,
        "accepted": 0,
        "rejected_burden_migration": 0,
        "timeout": 0,
        "state_limit": 0,
        "operators": {"phase_prefix_completion": 0, "phase_order": 0},
    }
    if source.move_time != 0:
        stats.update({"status": "UNSUPPORTED", "reason": "nonzero_move_time"})
        return [], stats
    if source.makespan < 4 or M < 2:
        stats.update({"status": "UNSUPPORTED", "reason": "horizon_or_crane_count"})
        return [], stats
    if state_limit <= 0 or max_candidates <= 0:
        stats.update({"status": "UNKNOWN_STATE_LIMIT", "state_limit": 1})
        return [], stats

    rows, source_work = _transaction_rows_work_map(source, M)
    horizon = source.makespan
    diagnostics = _continuity_diagnostics(source, M)
    focus_revisits = sorted(
        [
            item for item in diagnostics.get("crane_work_revisits", [])
            if int(item.get("residual_length", 0)) > 2
            or int(item.get("gap", 0)) >= 8
        ],
        key=lambda item: (
            -int(item.get("gap", 0)),
            -int(item.get("residual_length", 0)),
            int(item.get("crane", 0)),
            int(item.get("bay", 0)),
        ),
    )
    stats["focus_revisits"] = len(focus_revisits)
    stats["multi_slot_revisits"] = sum(
        int(item.get("residual_length", 0)) > 2 for item in focus_revisits
    )
    if not focus_revisits:
        stats["status"] = "NOT_APPLICABLE"
        return [], stats

    def expired() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    def phase_units(q: int, focus_q: int) -> list[tuple[int, int]]:
        """Build phase units, merging only the focus crane's revisit.

        The active band exists to resolve safety conflicts, not to force every
        adjacent crane to merge its work.  Non-focus cranes therefore retain
        their source block boundaries (including an existing split that the
        current transaction is not trying to repair), while their bounded
        block order may be permuted to make room for the focus phase.  The
        focus crane is the only one whose repeated same-bay blocks are
        aggregated into one indivisible logical phase.
        """
        if q != focus_q:
            blocks: list[tuple[int, int]] = []
            t = 0
            while t < horizon:
                bay = source_work.get((t, q))
                if bay is None:
                    t += 1
                    continue
                start = t
                t += 1
                while t < horizon and source_work.get((t, q)) == bay:
                    t += 1
                blocks.append((int(bay), t - start))
            return blocks
        order: list[int] = []
        totals: dict[int, int] = {}
        for t in range(horizon):
            bay = source_work.get((t, q))
            if bay is None:
                continue
            bay = int(bay)
            if bay not in totals:
                order.append(bay)
                totals[bay] = 0
            totals[bay] += 1
        return [(bay, totals[bay]) for bay in order]

    def phase_orders(q: int, focus_q: int) -> list[tuple[tuple[int, int], ...]]:
        units = phase_units(q, focus_q)
        if not units:
            return []
        first_bay = source_work.get((0, q))
        if first_bay is None:
            return []
        first_index = next(
            (index for index, item in enumerate(units)
             if int(item[0]) == int(first_bay)),
            None,
        )
        if first_index is None:
            return []
        first = units[first_index]
        rest = units[:first_index] + units[first_index + 1:]
        unique: list[tuple[tuple[int, int], ...]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        source_order = (first, *rest)
        for perm in itertools.permutations(rest):
            value = (first, *perm)
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        unique.sort(key=lambda item: (
            0 if item == source_order else 1,
            sum(abs(index - source_order.index(value))
                for index, value in enumerate(item)),
            tuple(bay for bay, _length in item),
        ))
        return unique[:max(1, max_block_permutations)]

    def active_band(focus_crane: int) -> tuple[int, ...]:
        width = max(2, min(max_active_cranes, M))
        left = max(0, min(focus_crane - 1, M - width))
        return tuple(range(left, left + width))

    # The event beam uses compact state fields and keeps only a bounded
    # history per layer.  A state stores full rows because phase transactions
    # must later be auditable, but the beam itself is capped well below the
    # state expansion limit.
    # A narrow beam is intentional here: complete phases have very few
    # meaningful boundary choices, while keeping hundreds of equivalent wait
    # states would exhaust the global expansion budget before another phase
    # permutation is inspected.
    beam_width = max(16, min(48, state_limit // max(1, horizon // 4)))
    proposals: list[dict[str, Any]] = []
    seen_proposals: set[tuple[Any, ...]] = set()

    def pending_work(
        state: tuple[Any, ...],
        orders: dict[int, list[tuple[tuple[int, int], ...]]],
        active_cranes: Sequence[int],
    ) -> int:
        """Count both current and not-yet-started phase work.

        ``remaining`` only stores the current phase's tail.  Treating a
        completed-but-not-advanced phase as zero remaining work makes an idle
        parking state look better than a state that has actually started the
        next phase, so the beam would fill with states that never finish the
        transaction.  Include all future phase lengths in the ranking key.
        """
        phase_index = state[1]
        remaining = state[2]
        total = 0
        for q in active_cranes:
            index = int(phase_index[q])
            if index < 0:
                continue
            total += int(remaining[q])
            total += sum(
                int(length)
                for _bay, length in orders[q][index + 1:]
            )
        return total

    for revisit in focus_revisits:
        if expired():
            stats["status"] = "UNKNOWN_DEADLINE"
            stats["timeout"] += 1
            break
        focus_q = int(revisit["crane"]) - 1
        active = active_band(focus_q)
        if focus_q not in active:
            continue
        stats["active_bands_generated"] += 1
        orders_by_crane: dict[int, list[tuple[tuple[int, int], ...]]] = {}
        invalid = False
        for q in active:
            orders = phase_orders(q, focus_q)
            if not orders:
                invalid = True
                break
            orders_by_crane[q] = orders
            stats["phase_permutations_generated"] += len(orders)
        if invalid:
            continue
        order_lists = [orders_by_crane[q] for q in active]
        total_combinations = math.prod(len(items) for items in order_lists)
        combination_budget = max(
            2_000,
            state_limit // max(1, min(total_combinations, 8)),
        )
        combinations = itertools.product(*order_lists)
        for order_tuple in combinations:
            if expired():
                stats["status"] = "UNKNOWN_DEADLINE"
                stats["timeout"] += 1
                break
            if stats["phase_combinations_tested"] >= max_block_permutations:
                stats["status"] = "UNKNOWN_STATE_LIMIT"
                stats["state_limit"] += 1
                break
            stats["phase_combinations_tested"] += 1
            orders = {q: order_tuple[index] for index, q in enumerate(active)}
            stats["operators"]["phase_order"] += 1
            combination_expanded = 0
            combination_limited = False

            # Each state is a tuple so it can be safely deduplicated.  The
            # variable names are: positions, phase index, remaining slots,
            # last directions, moves, reversals, history, work rows, idle.
            first_row = tuple(rows[0])
            first_work = tuple(source_work.get((0, q)) for q in range(M))
            initial_index = tuple(
                0 if q in active else -1 for q in range(M)
            )
            initial_remaining = tuple(
                (
                    int(orders[q][0][1]) - 1
                    if q in active else 0
                )
                for q in range(M)
            )
            if any(
                q in active
                and int(orders[q][0][0]) != int(first_work[q])
                for q in active
            ):
                stats["states_pruned_safety"] += 1
                continue
            initial = (
                first_row,
                initial_index,
                initial_remaining,
                tuple(0 for _ in range(M)),
                0,
                0,
                (first_row,),
                (first_work,),
                0,
            )
            beam = [initial]
            for t in range(1, horizon):
                if expired():
                    stats["status"] = "UNKNOWN_DEADLINE"
                    stats["timeout"] += 1
                    break
                next_states: list[tuple[Any, ...]] = []
                for state in beam:
                    positions, phase_index, remaining, directions, moves, reversals, history, work_rows, idle = state
                    per_crane: list[list[tuple[int, bool, int, int]]] = []
                    for q in active:
                        idx = int(phase_index[q])
                        rem = int(remaining[q])
                        options: list[tuple[int, bool, int, int]] = []
                        if rem > 0:
                            bay = int(orders[q][idx][0])
                            options = [(bay, True, idx, rem - 1)]
                        else:
                            next_idx = idx + 1
                            if next_idx < len(orders[q]):
                                next_bay = int(orders[q][next_idx][0])
                                options.append((next_bay, True, next_idx,
                                                int(orders[q][next_idx][1]) - 1))
                            # Keep the event beam small.  A completed phase
                            # may wait at its current position; the source
                            # position is the only extra parking choice.  The
                            # next phase itself is the only other move.  This
                            # avoids enumerating every bay as a fake parking
                            # state while retaining the useful Q4/Q5 relay.
                            parking = {
                                int(positions[q]),
                                int(rows[t][q]),
                            }
                            for position in sorted(parking):
                                options.append((position, False, idx, 0))
                        dedup_options: list[tuple[int, bool, int, int]] = []
                        seen_options: set[tuple[int, bool, int, int]] = set()
                        for option in options:
                            if option not in seen_options:
                                seen_options.add(option)
                                dedup_options.append(option)
                        per_crane.append(dedup_options)
                    for choices in itertools.product(*per_crane):
                        if stats["states_expanded"] >= state_limit:
                            stats["status"] = "UNKNOWN_STATE_LIMIT"
                            stats["state_limit"] += 1
                            break
                        if combination_expanded >= combination_budget:
                            combination_limited = True
                            break
                        stats["states_expanded"] += 1
                        combination_expanded += 1
                        proposed_row = list(rows[t])
                        # Non-active cranes are frozen to the source ledger at
                        # this absolute time; carrying the previous row would
                        # silently duplicate or erase their work.
                        proposed_work = [
                            source_work.get((t, q)) for q in range(M)
                        ]
                        next_index = list(phase_index)
                        next_remaining = list(remaining)
                        next_directions = list(directions)
                        next_moves = int(moves)
                        next_reversals = int(reversals)
                        next_idle = int(idle)
                        for q, (position, is_work, idx, rem) in zip(active, choices):
                            proposed_row[q] = int(position)
                            proposed_work[q] = int(orders[q][idx][0]) if is_work else None
                            next_index[q] = int(idx)
                            next_remaining[q] = int(rem)
                            if proposed_row[q] != positions[q]:
                                direction = 1 if proposed_row[q] > positions[q] else -1
                                next_moves += 1
                                if next_directions[q] and next_directions[q] != direction:
                                    next_reversals += 1
                                next_directions[q] = direction
                            if not is_work:
                                next_idle += 1
                        if any(
                            right - left < 2
                            for left, right in zip(proposed_row, proposed_row[1:])
                        ):
                            stats["states_pruned_safety"] += 1
                            continue
                        working = [bay for bay in proposed_work if bay is not None]
                        if len(working) != len(set(working)):
                            stats["states_pruned_safety"] += 1
                            continue
                        # Moves are the second formal priority, but a local
                        # phase may temporarily add moves to bring work
                        # forward.  Keep such states eligible for complete
                        # (C, K) comparison; the explicit state/combination
                        # budgets, not a source-relative move cap, bound this
                        # exploratory neighborhood.
                        next_states.append((
                            tuple(proposed_row), tuple(next_index),
                            tuple(next_remaining), tuple(next_directions),
                            next_moves, next_reversals,
                            (*history, tuple(proposed_row)),
                            (*work_rows, tuple(proposed_work)),
                            next_idle,
                        ))
                    if stats["status"] == "UNKNOWN_STATE_LIMIT":
                        break
                    if combination_limited:
                        break
                if stats["status"] == "UNKNOWN_DEADLINE":
                    break
                if stats["status"] == "UNKNOWN_STATE_LIMIT":
                    break
                if combination_limited:
                    break
                if not next_states:
                    stats["states_pruned_horizon"] += 1
                    beam = []
                    break
                dedup: dict[tuple[Any, ...], tuple[Any, ...]] = {}
                for state in next_states:
                    key = (
                        state[0], state[1], state[2], state[3],
                    )
                    old = dedup.get(key)
                    if old is None or (
                        state[4], state[5], state[8]
                    ) < (old[4], old[5], old[8]):
                        dedup[key] = state
                beam = sorted(
                    dedup.values(),
                    key=lambda state: (
                        pending_work(state, orders, active),
                        state[4], state[5], sum(state[2]), state[8], state[0],
                    ),
                )[:beam_width]
            if stats["status"] in {"UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT"}:
                break
            if combination_limited:
                continue
            for state in beam:
                positions, phase_index, remaining, directions, moves, reversals, history, work_rows, idle = state
                if any(int(value) != 0 for value in remaining):
                    continue
                if any(
                    q in active and int(phase_index[q]) != len(orders[q]) - 1
                    for q in active
                ):
                    continue
                stats["complete_phase_plans"] += 1
                final_history = (*history, tuple(positions))
                plan = {
                    (t, q): work_rows[t][q]
                    for t in range(horizon)
                    for q in range(M)
                }
                signature = (
                    tuple(final_history),
                    tuple(sorted(plan.items())),
                )
                if signature in seen_proposals:
                    continue
                seen_proposals.add(signature)
                details = {
                    "focus": {
                        "crane": int(revisit["crane"]),
                        "bay": int(revisit["bay"]),
                        "primary_block": dict(revisit["primary_block"]),
                        "residual_block": dict(revisit["residual_block"]),
                        "residual_length": int(revisit.get("residual_length", 0)),
                    },
                    "active_cranes": [q + 1 for q in active],
                    "phase_orders": {
                        str(q + 1): [
                            {"bay": int(bay), "length": int(length)}
                            for bay, length in orders[q]
                        ]
                        for q in active
                    },
                    "moves": int(moves),
                    "reversals": int(reversals),
                    "idle": int(idle),
                    "operator": "phase_block_resequence",
                }
                proposals.append({
                    "operator": "phase_block_resequence",
                    "history": [tuple(row) for row in final_history],
                    "work_plan": plan,
                    "active_cranes": list(active),
                    "regions": [{
                        "crane": q + 1,
                        "start": 0,
                        "end_exclusive": horizon,
                        "length": horizon,
                        "segment": 0,
                    } for q in active],
                    "source_signature": _trajectory_signature(source),
                    "source_hash": source_hash,
                    "details": details,
                })
                if len(proposals) >= max_candidates:
                    break
            if len(proposals) >= max_candidates:
                break
        if len(proposals) >= max_candidates:
            break

    proposals.sort(key=lambda item: (
        int(item["details"]["focus"]["residual_length"]),
        int(item["details"]["moves"]),
        int(item["details"]["reversals"]),
        int(item["details"]["idle"]),
    ))
    return proposals[:max_candidates], stats


def _phase_closure_relay_search(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    source: _CandidateSchedule,
    *,
    state_limit: int = 50_000,
    max_candidates: int = 32,
    max_phase_orders: int = 128,
    max_variants_per_crane: int = 12_000,
    deadline: float | None = None,
    source_hash: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search complete work phases with event-time starts and a growing band.

    Every `(crane, bay)` is represented as one indivisible phase, so a source
    revisit can only survive if the crane visits that bay in separate phases
    created by an explicit later operator.  The first phase remains the
    forced t=0 phase.  Internal phase gaps and terminal idle are enumerated as
    event offsets, not as per-slot wait states.  The active band starts at the
    focus crane and expands one adjacent crane at a time; cranes outside the
    band retain their complete source trajectory.

    This operator preserves the source work owner for every bay.  It does not
    transfer work between cranes; such transfers remain the responsibility
    of the explicit work-transfer operators.  Every returned transaction has
    a complete per-slot work ledger and is independently verified by the
    caller.
    """
    stats: dict[str, Any] = {
        "status": "SEARCH_COMPLETE",
        "focus_revisits": 0,
        "activity_bands_tested": 0,
        "activity_chain_expansions": [],
        "phase_orders_generated": 0,
        "event_schedules_generated": 0,
        "states_expanded": 0,
        "complete_phase_plans": 0,
        "ledger_closed": 0,
        "decoded": 0,
        "verified": 0,
        "accepted": 0,
        "rejected_safety": 0,
        "rejected_horizon": 0,
        "rejected_ledger": 0,
        "timeout": 0,
        "state_limit": 0,
        "max_activity_width": 0,
    }
    if source.move_time != 0:
        stats.update({"status": "UNSUPPORTED", "reason": "nonzero_move_time"})
        return [], stats
    if state_limit <= 0 or max_candidates <= 0:
        stats.update({"status": "UNKNOWN_STATE_LIMIT", "state_limit": 1})
        return [], stats

    horizon = source.makespan
    source_rows, source_work = _transaction_rows_work_map(source, M)
    diagnostics = _continuity_diagnostics(source, M, starts)
    focuses = [dict(item) for item in diagnostics.get("crane_work_revisits", [])]
    focuses.sort(key=lambda item: (
        -max(1, int(item.get("total_work_on_crane_bay", 0))
             - int(item.get("residual_length", 0))),
        -int(item.get("residual_length", 0)),
        -int(item.get("gap", 0)),
        int(item.get("crane", 0)),
        int(item.get("bay", 0)),
    ))
    stats["focus_revisits"] = len(focuses)
    if not focuses:
        stats["status"] = "NOT_APPLICABLE"
        return [], stats

    source_positions = [
        tuple(row[q] for row in source_rows) for q in range(M)
    ]
    source_first_time: list[dict[int, int]] = [dict() for _ in range(M)]
    source_bay_counts: list[dict[int, int]] = [dict() for _ in range(M)]
    source_bay_order: list[list[int]] = [[] for _ in range(M)]
    for (t, q), bay in source_work.items():
        if bay is None:
            continue
        bay = int(bay)
        if bay not in source_first_time[q]:
            source_first_time[q][bay] = int(t)
            source_bay_order[q].append(bay)
        source_bay_counts[q][bay] = source_bay_counts[q].get(bay, 0) + 1

    def expired() -> bool:
        if deadline is not None and time.perf_counter() >= deadline:
            stats["status"] = "UNKNOWN_DEADLINE"
            stats["timeout"] += 1
            return True
        return False

    def phase_orders(q: int) -> list[tuple[tuple[int, int], ...]]:
        order = source_bay_order[q]
        if not order:
            return [()]
        first_bay = source_work.get((0, q))
        if first_bay is None or int(first_bay) != order[0]:
            return []
        first = (int(first_bay), source_bay_counts[q][int(first_bay)])
        rest = [
            (bay, source_bay_counts[q][bay])
            for bay in order if bay != int(first_bay)
        ]
        permutations = list(itertools.islice(itertools.permutations(rest), max_phase_orders))
        permutations.sort(key=lambda values: (
            0 if values == tuple(rest) else 1,
            sum(
                abs(index - rest.index(value))
                for index, value in enumerate(values)
            ),
            tuple(bay for bay, _length in values),
        ))
        return [(first, *value) for value in permutations]

    def idle_vectors(slack: int, phase_count: int):
        """Yield distinct internal idle placements by event offsets.

        Terminal idle is implicit in the fixed horizon and cannot change
        phase starts, so it must not be enumerated as a free event.
        """
        bins = max(0, phase_count - 1)
        current = [0] * bins

        def distribute(index: int, remaining: int):
            if index == bins:
                if remaining == 0:
                    yield tuple(current)
                return
            if index == bins - 1:
                current[index] = remaining
                yield tuple(current)
                return
            for value in range(remaining + 1):
                current[index] = value
                yield from distribute(index + 1, remaining - value)

        # Enumerate exact compositions for each total internal idle.  The
        # trailing slack is implicit in the horizon.
        for total in range(max(0, slack) + 1):
            if bins == 0:
                yield ()
            else:
                yield from distribute(0, total)

    variants_by_crane: dict[int, list[dict[str, Any]]] = {}
    for q in range(M):
        phases_list = phase_orders(q)
        stats["phase_orders_generated"] += len(phases_list)
        variants: list[dict[str, Any]] = []
        load = sum(source_bay_counts[q].values())
        slack = max(0, horizon - load)
        for phases in phases_list:
            if not phases:
                continue
            for gaps in idle_vectors(slack, len(phases)):
                phase_starts = []
                time_cursor = 0
                for index, (_bay, length) in enumerate(phases):
                    phase_starts.append(time_cursor)
                    time_cursor += int(length)
                    if index < len(phases) - 1:
                        time_cursor += int(gaps[index])
                if time_cursor > horizon:
                    stats["rejected_horizon"] += 1
                    continue
                movement_count = sum(
                    int(phases[index - 1][0] != phases[index][0])
                    for index in range(1, len(phases))
                    if phase_starts[index] < horizon
                )
                source_deviation = sum(
                    abs(int(phase_starts[index]) - int(source_first_time[q].get(bay, phase_starts[index])))
                    for index, (bay, _length) in enumerate(phases)
                )
                variants.append({
                    "phases": tuple((int(bay), int(length)) for bay, length in phases),
                    "gaps": tuple(int(value) for value in gaps),
                    "starts": tuple(int(value) for value in phase_starts),
                    "moves": movement_count,
                    "source_deviation": source_deviation,
                    "idle": sum(int(value) for value in gaps),
                })
        variants.sort(key=lambda item: (
            item["moves"], item["source_deviation"], item["idle"],
            item["starts"], tuple(bay for bay, _length in item["phases"]),
        ))
        variants_by_crane[q] = variants[:max_variants_per_crane]

    def materialize(q: int, variant: dict[str, Any]):
        positions = [0] * (horizon + 1)
        work = [None] * horizon
        phases = variant["phases"]
        phase_starts = variant["starts"]
        for index, (bay, length) in enumerate(phases):
            start_t = phase_starts[index]
            end_t = start_t + length
            next_start = (
                phase_starts[index + 1]
                if index + 1 < len(phases) else horizon + 1
            )
            if end_t > horizon or next_start > horizon + 1:
                return None
            for t in range(start_t, min(next_start, horizon + 1)):
                positions[t] = bay
            for t in range(start_t, end_t):
                if work[t] is not None:
                    return None
                work[t] = bay
        if any(position <= 0 for position in positions):
            return None
        return tuple(positions), tuple(work)

    focus_orders_cache = {
        q: variants_by_crane[q] for q in range(M)
    }
    proposals: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    first_conflict_by_band: dict[tuple[int, ...], dict[str, Any]] = {}

    for focus in focuses:
        if expired() or len(proposals) >= max_candidates:
            break
        focus_q = int(focus["crane"]) - 1
        focus_bay = int(focus["bay"])
        for width in range(2, M + 1):
            left_min = max(0, focus_q - width + 1)
            left_max = min(focus_q, M - width)
            bands = [tuple(range(left, left + width))
                     for left in range(left_min, left_max + 1)]
            for band in bands:
                if expired() or len(proposals) >= max_candidates:
                    break
                stats["activity_bands_tested"] += 1
                stats["max_activity_width"] = max(
                    stats["max_activity_width"], len(band)
                )
                # The chain grows outward from the focus.  This order resolves
                # its nearest safety conflicts before more distant cranes.
                order = [focus_q]
                for distance in range(1, M):
                    for q in (focus_q - distance, focus_q + distance):
                        if q in band and q not in order:
                            order.append(q)
                order.extend(q for q in band if q not in order)
                expansion = {
                    "focus": {"crane": focus_q + 1, "bay": focus_bay},
                    "active_cranes": [q + 1 for q in band],
                    "ordered_chain": [q + 1 for q in order],
                    "width": len(band),
                    "resolved_conflicts": [],
                }
                resolved: dict[int, tuple[int, ...]] = {
                    q: source_positions[q] for q in range(M) if q not in band
                }
                selected: dict[int, tuple[tuple[int, ...], tuple[int | None, ...], dict[str, Any]]] = {}
                failed = False
                local_nodes = 0

                def extend(depth: int) -> None:
                    nonlocal local_nodes, failed
                    if expired() or len(proposals) >= max_candidates:
                        return
                    if stats["states_expanded"] >= state_limit:
                        stats["status"] = "UNKNOWN_STATE_LIMIT"
                        stats["state_limit"] += 1
                        failed = True
                        return
                    if depth >= len(order):
                        stats["complete_phase_plans"] += 1
                        history = [list(row) for row in source_rows]
                        work_plan = dict(source_work)
                        phase_report = {}
                        for q, (positions, work, variant) in selected.items():
                            for t, bay in enumerate(positions):
                                history[t][q] = int(bay)
                            for t, bay in enumerate(work):
                                work_plan[(t, q)] = None if bay is None else int(bay)
                            phase_report[str(q + 1)] = {
                                "phases": [
                                    {"bay": int(bay), "length": int(length),
                                     "start": int(variant["starts"][index])}
                                    for index, (bay, length) in enumerate(variant["phases"])
                                ],
                                "gaps": list(variant["gaps"]),
                            }
                        if any(
                            sum(value == bay for (time_index, _q), value in work_plan.items()
                                if time_index < horizon) != int(required)
                            for bay, required in enumerate(W, 1)
                        ):
                            stats["rejected_ledger"] += 1
                            return
                        stats["ledger_closed"] += 1
                        signature = (
                            tuple(tuple(row) for row in history),
                            tuple(sorted(work_plan.items())),
                        )
                        if signature in seen:
                            return
                        seen.add(signature)
                        try:
                            candidate = _candidate_from_rows_and_work_plan(
                                W, M, history, work_plan
                            )
                        except RuntimeError:
                            stats["rejected_safety"] += 1
                            return
                        if not _candidate_passes_independent_verifier(
                            W, M, starts, candidate
                        ):
                            stats["rejected_safety"] += 1
                            return
                        stats["decoded"] += 1
                        stats["verified"] += 1
                        candidate_diagnostics = _continuity_diagnostics(
                            candidate, M, starts
                        )
                        focus_remains = any(
                            int(item["crane"]) == focus_q + 1
                            and int(item["bay"]) == focus_bay
                            for item in candidate_diagnostics.get(
                                "crane_work_revisits", []
                            )
                        )
                        if (
                            candidate.objective_key > source.objective_key
                            or focus_remains
                        ):
                            return
                        stats["event_schedules_generated"] += 1
                        if candidate.objective_key < source.objective_key:
                            stats["accepted"] += 1
                        regions = [{
                            "crane": q + 1,
                            "start": 0,
                            "end_exclusive": horizon,
                            "length": horizon,
                            "segment": 0,
                        } for q in band]
                        proposals.append({
                            "operator": "phase_closure_relay",
                            "history": [tuple(row) for row in history],
                            "work_plan": work_plan,
                            "active_cranes": list(band),
                            "regions": regions,
                            "source_signature": _trajectory_signature(source),
                            "source_hash": source_hash,
                            "details": {
                                "focus": {
                                    "crane": focus_q + 1,
                                    "bay": focus_bay,
                                    "blocks": focus.get("blocks", []),
                                    "gap": focus.get("gap", 0),
                                },
                                "active_cranes": [q + 1 for q in band],
                                "activity_chain_expansions": [
                                    *stats["activity_chain_expansions"], expansion,
                                ],
                                "phase_schedule": phase_report,
                                "candidate_objective": list(candidate.objective_key),
                                "candidate_revisit_count": candidate_diagnostics.get(
                                    "work_revisit_count", 0
                                ),
                                "candidate_movement_count": candidate.completion_movement_count,
                                "work_ledger": {
                                    "source_by_bay": [
                                        sum(value == bay for value in source_work.values())
                                        for bay in range(1, len(W) + 1)
                                    ],
                                    "transaction_by_bay": [
                                        sum(value == bay for value in work_plan.values())
                                        for bay in range(1, len(W) + 1)
                                    ],
                                    "closed": True,
                                    "work_transfer": False,
                                },
                                "event_level": True,
                            },
                        })
                        return

                    q = order[depth]
                    for variant in focus_orders_cache[q]:
                        if expired() or len(proposals) >= max_candidates:
                            return
                        if stats["states_expanded"] >= state_limit:
                            stats["status"] = "UNKNOWN_STATE_LIMIT"
                            stats["state_limit"] += 1
                            failed = True
                            return
                        stats["states_expanded"] += 1
                        local_nodes += 1
                        materialized = materialize(q, variant)
                        if materialized is None:
                            stats["rejected_horizon"] += 1
                            continue
                        positions, work = materialized
                        if positions[0] != int(starts[q]):
                            stats["rejected_safety"] += 1
                            continue
                        conflict = None
                        for left_q in range(M - 1):
                            right_q = left_q + 1
                            left_positions = (
                                positions if left_q == q else resolved.get(left_q)
                            )
                            right_positions = (
                                positions if right_q == q else resolved.get(right_q)
                            )
                            if left_positions is None or right_positions is None:
                                continue
                            for t, (left_pos, right_pos) in enumerate(zip(
                                left_positions, right_positions
                            )):
                                if int(right_pos) - int(left_pos) < 2:
                                    conflict = {
                                        "time": t,
                                        "left_crane": left_q + 1,
                                        "right_crane": right_q + 1,
                                        "left_position": int(left_pos),
                                        "right_position": int(right_pos),
                                        "trigger_crane": q + 1,
                                    }
                                    break
                            if conflict:
                                break
                        if conflict is not None:
                            stats["rejected_safety"] += 1
                            if len(expansion["resolved_conflicts"]) < 8:
                                expansion["resolved_conflicts"].append(conflict)
                            continue
                        selected[q] = (positions, work, variant)
                        resolved[q] = positions
                        extend(depth + 1)
                        del resolved[q]
                        del selected[q]
                        if stats["status"] in {"UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT"}:
                            return

                extend(0)
                expansion["states_expanded"] = local_nodes
                stats["activity_chain_expansions"].append(expansion)
                if failed and stats["status"] in {"UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT"}:
                    break
            if stats["status"] in {"UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT"}:
                break
        if stats["status"] in {"UNKNOWN_DEADLINE", "UNKNOWN_STATE_LIMIT"}:
            break

    proposals.sort(key=lambda item: (
        tuple(item["details"]["candidate_objective"]),
        int(item["details"]["candidate_revisit_count"]),
        tuple(item["details"]["active_cranes"]),
    ))
    return proposals[:max_candidates], stats


def _candidate_from_explicit_cross_crane_phase_transaction(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    history: Sequence[tuple[int, ...]],
    source: _CandidateSchedule,
    work_plan: dict[tuple[int, int], int | None],
    active_cranes: Sequence[int],
    *,
    source_hash: str | None = None,
    transaction_source_hash: str | None = None,
) -> _CandidateSchedule:
    """Decode one atomic phase transaction that may change work owners.

    The older phase decoder already freezes every inactive crane and checks the
    complete explicit ledger.  Keeping this small named entry point separate
    makes the new operator's contract visible to callers and prevents it from
    silently falling back to the short-window work-transfer decoder.
    """
    return _candidate_from_explicit_phase_transaction(
        W, M, starts, history, source, work_plan, active_cranes,
        source_hash=source_hash,
        transaction_source_hash=transaction_source_hash,
    )


def _cross_crane_phase_relay_search(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    source: _CandidateSchedule,
    *,
    state_limit: int = 50_000,
    max_candidates: int = 32,
    max_assignment_variants: int = 256,
    deadline: float | None = None,
    source_hash: str | None = None,
    include_idle_capacity: bool = False,
    idle_capacity_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search one atomic cross-crane phase/ownership transaction.

    ``_phase_closure_relay_search`` keeps each phase bound to its source
    crane.  This operator starts from the same complete source work blocks but
    treats their owner as a decision: a focus block may be assigned to an
    adjacent (or two-hop) crane, while all work outside the active band stays
    frozen.  A small dynamic program then chooses legal safe configurations for
    every event row, so donor displacement and receiver movement are solved in
    the same transaction rather than in two independent local edits.

    The operator is deliberately bounded.  It is a neighborhood expander, not
    a claim of global optimality: a timeout or state limit is reported as
    ``UNKNOWN_*`` and never as infeasibility.
    """
    stats: dict[str, Any] = {
        "status": "SEARCH_COMPLETE",
        "focus_revisits": 0,
        "idle_capacity_focuses": 0,
        "idle_capacity_chain_focuses": 0,
        "partial_transfer_focuses": 0,
        "focuses_without_revisits": 0,
        "activity_bands_tested": 0,
        "assignment_variants_generated": 0,
        "retimed_plans_generated": 0,
        "retimed_plans_solved": 0,
        "owner_change_branches": 0,
        "two_hop_relay_branches": 0,
        "complete_phase_plans": 0,
        "ledger_closed": 0,
        "states_expanded": 0,
        "decoded": 0,
        "verified": 0,
        "accepted": 0,
        "rejected_overlap": 0,
        "rejected_eligibility": 0,
        "rejected_safety": 0,
        "rejected_ledger": 0,
        "rejected_horizon": 0,
        "timeout": 0,
        "state_limit": 0,
        "max_activity_width": 0,
        "activity_chain_expansions": [],
    }
    if source.move_time != 0:
        stats.update({"status": "UNSUPPORTED", "reason": "nonzero_move_time"})
        return [], stats
    if state_limit <= 0 or max_candidates <= 0:
        stats.update({"status": "UNKNOWN_STATE_LIMIT", "state_limit": 1})
        return [], stats

    horizon = int(source.makespan)
    source_rows, source_work = _transaction_rows_work_map(source, M)
    diagnostics = _continuity_diagnostics(source, M, starts)
    focuses = (
        [] if idle_capacity_only else
        [dict(item) for item in diagnostics.get("crane_work_revisits", [])]
    )
    revisit_focus_count = len(focuses)
    N = len(W)
    eligibility = _bay_eligibility(N, M, [])
    legal_configurations = _legal_configurations(N, M)
    if not legal_configurations:
        stats.update({"status": "NOT_APPLICABLE", "reason": "no_safe_configuration"})
        return [], stats

    # Convert the source work map into maximal same-crane/same-bay phases.
    source_blocks: list[dict[str, int]] = []
    blocks_by_crane: dict[int, list[dict[str, int]]] = {
        q: [] for q in range(M)
    }
    for q in range(M):
        t = 0
        while t < horizon:
            bay = source_work.get((t, q))
            if bay is None:
                t += 1
                continue
            start_t = t
            bay = int(bay)
            while t < horizon and source_work.get((t, q)) == bay:
                t += 1
            block = {
                "id": len(source_blocks),
                "source_owner": q,
                "bay": bay,
                "start": start_t,
                "end": t,
                "length": t - start_t,
            }
            source_blocks.append(block)
            blocks_by_crane[q].append(block)

    # The old relay neighborhood had no entry point when a crane finished
    # early without revisiting a bay.  Add generic donor/receiver foci from
    # actual loads and eligible source blocks; quantities are integer units,
    # and a long source phase is split at the selected transfer boundary.
    if include_idle_capacity:
        source_loads = [
            sum(value is not None for (_t, owner), value in source_work.items()
                if owner == q)
            for q in range(M)
        ]
        remaining_capacity = [
            max(0, horizon - load) for load in source_loads
        ]
        idle_focuses: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for block in source_blocks:
            donor = int(block["source_owner"])
            mandatory_prefix = int(
                int(block["start"]) == 0 and int(block["bay"]) in set(starts)
            )
            transferable_capacity = max(
                0, int(block["length"]) - mandatory_prefix
            )
            for receiver in sorted(eligibility[int(block["bay"]) - 1]):
                # Idle-capacity balancing is a local diffusion operator.  A
                # direct hand-off to a distant crane forces the active band to
                # span every intervening crane, consumes the bounded search on
                # mostly infeasible full-width variants, and is exactly the
                # kind of global rewrite that ``local_only`` must avoid.
                # Longer transfers are therefore composed from several
                # independently verified adjacent-crane transactions.
                if (
                    abs(receiver - donor) != 1
                    or source_loads[donor] <= source_loads[receiver]
                ):
                    continue
                capacity = remaining_capacity[receiver]
                ideal = min(
                    transferable_capacity, capacity,
                    max(1, (source_loads[donor] - source_loads[receiver]) // 2),
                )
                if ideal <= 0:
                    continue
                quantities = {ideal}
                for delta in (-1, 1):
                    neighbor = ideal + delta
                    if 1 <= neighbor <= min(transferable_capacity, capacity):
                        quantities.add(neighbor)
                if ideal < int(block["length"]) and ideal < capacity:
                    quantities.add(min(transferable_capacity, capacity))
                for transfer in sorted(quantities):
                    projected = list(source_loads)
                    projected[donor] -= transfer
                    projected[receiver] += transfer
                    balance_deviation = sum(
                        (M * load - sum(source_loads)) ** 2
                        for load in projected
                    )
                    max_idle = horizon - min(projected, default=0)
                    rank = (
                        max_idle, balance_deviation,
                        abs(source_loads[donor] - source_loads[receiver]),
                        -int(transfer), int(block["start"]),
                        donor, receiver, int(block["bay"]),
                    )
                    sides = (
                        ("suffix",) if transfer == int(block["length"])
                        else ("suffix",) if mandatory_prefix
                        else ("prefix", "suffix")
                    )
                    for side in sides:
                        focus = {
                            "focus_type": "idle_capacity",
                            "crane": donor + 1,
                            "bay": int(block["bay"]),
                            "focus_source_block_id": int(block["id"]),
                            "transfer_length": int(transfer),
                            "transfer_side": side,
                            "target_owner": int(receiver),
                            "projected_max_nonwork_capacity": int(max_idle),
                            "projected_load_balance_deviation": int(balance_deviation),
                            "blocks": [{
                                "start": int(block["start"]),
                                "end_exclusive": int(block["end"]),
                            }],
                            "gap": max(0, horizon - source_loads[donor]),
                        }
                        idle_focuses.append((rank + (side,), focus))
                        # When the receiver cannot enter the donor's bay
                        # because the upstream crane leaves no parking gap,
                        # a two-crane hand-off is structurally impossible even
                        # though the receiver has ample idle capacity.  Add a
                        # three-crane atomic relay: the upstream crane gives
                        # the same quantity to the donor while the donor gives
                        # it to the receiver.  Net load of the middle crane is
                        # unchanged and no infeasible intermediate schedule is
                        # ever published.
                        direction = receiver - donor
                        upstream = donor - direction
                        if 0 <= upstream < M:
                            relay_options = []
                            for relay_block in blocks_by_crane[upstream]:
                                relay_mandatory = int(
                                    int(relay_block["start"]) == 0
                                    and int(relay_block["bay"]) in set(starts)
                                )
                                relay_capacity = max(
                                    0,
                                    int(relay_block["length"])
                                    - relay_mandatory,
                                )
                                if (
                                    donor in eligibility[
                                        int(relay_block["bay"]) - 1
                                    ]
                                    and relay_capacity >= transfer
                                ):
                                    relay_options.append(relay_block)
                            if relay_options:
                                relay_block = min(
                                    relay_options,
                                    key=lambda item: (
                                        int(any(
                                            int(existing["bay"])
                                            == int(item["bay"])
                                            for existing in blocks_by_crane[
                                                donor
                                            ]
                                        )),
                                        -direction * int(item["bay"]),
                                        -int(item["start"]),
                                        -int(item["length"]),
                                    ),
                                )
                                chain_projected = list(source_loads)
                                chain_projected[upstream] -= transfer
                                chain_projected[receiver] += transfer
                                chain_deviation = sum(
                                    (M * load - sum(source_loads)) ** 2
                                    for load in chain_projected
                                )
                                chain_max_idle = horizon - min(
                                    chain_projected, default=0
                                )
                                chain_focus = {
                                    **focus,
                                    "relay_source_block_id": int(
                                        relay_block["id"]
                                    ),
                                    "relay_target_owner": donor,
                                    "relay_transfer_length": int(transfer),
                                    "relay_transfer_side": "suffix",
                                    "projected_max_nonwork_capacity": int(
                                        chain_max_idle
                                    ),
                                    "projected_load_balance_deviation": int(
                                        chain_deviation
                                    ),
                                }
                                chain_rank = (
                                    chain_max_idle,
                                    chain_deviation,
                                    -int(transfer),
                                    int(block["start"]),
                                    donor,
                                    receiver,
                                    int(block["bay"]),
                                    "chain",
                                    side,
                                )
                                idle_focuses.append(
                                    (chain_rank, chain_focus)
                                )
        idle_focuses.sort(key=lambda item: item[0])
        focuses.extend(focus for _rank, focus in idle_focuses[:64])
        stats["idle_capacity_focuses"] = min(64, len(idle_focuses))
        stats["idle_capacity_chain_focuses"] = sum(
            "relay_source_block_id" in focus
            for _rank, focus in idle_focuses[:64]
        )
        stats["partial_transfer_focuses"] = sum(
            int(focus["transfer_length"])
            < next(
                int(block["length"]) for block in source_blocks
                if int(block["id"]) == int(focus["focus_source_block_id"])
            )
            for _rank, focus in idle_focuses[:64]
        )
        stats["focuses_without_revisits"] = int(
            idle_capacity_only and not diagnostics.get("crane_work_revisits")
        )
    focuses.sort(key=lambda item: (
        0 if item.get("focus_type") == "idle_capacity" else 1,
        int(item.get("projected_max_nonwork_capacity", 0)),
        int(item.get("projected_load_balance_deviation", 0)),
        -int(item.get("transfer_length", 0)),
        -int(item.get("residual_length", item.get("work_count", 0))),
        -int(item.get("gap", 0)),
        int(item.get("crane", 0)),
        int(item.get("bay", 0)),
    ))
    stats["focus_revisits"] = revisit_focus_count
    if not focuses:
        stats["status"] = "NOT_APPLICABLE"
        return [], stats

    def expired() -> bool:
        if deadline is not None and time.perf_counter() >= deadline:
            stats["status"] = "UNKNOWN_DEADLINE"
            stats["timeout"] += 1
            return True
        return False

    def block_is_focus(block: dict[str, int], focus: dict[str, Any]) -> bool:
        focus_task_ids = {
            int(value) for value in focus.get("focus_task_ids", [])
        }
        if focus_task_ids and int(block["id"]) in focus_task_ids:
            return True
        source_focus_ids = {
            int(value) for value in (
                focus.get("focus_source_block_id"),
                focus.get("relay_source_block_id"),
            )
            if value is not None
        }
        if source_focus_ids and int(block["id"]) in source_focus_ids:
            return True
        if block["source_owner"] != int(focus.get("crane", 0)) - 1:
            return False
        if block["bay"] != int(focus.get("bay", 0)):
            return False
        if "focus_task_id" in focus:
            return block["id"] == int(focus["focus_task_id"])
        primary = focus.get("primary_block") or {}
        residual = focus.get("residual_block") or {}
        return (
            block["start"] == int(residual.get("start", -1))
            and block["end"] == int(residual.get("end_exclusive", -1))
        ) or (
            block["start"] == int(primary.get("start", -1))
            and block["end"] == int(primary.get("end_exclusive", -1))
        )

    proposals: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    tested_bands: set[tuple[Any, ...]] = set()

    def owner_choices(block: dict[str, int], band: tuple[int, ...]) -> list[int]:
        source_owner = block["source_owner"]
        bay = block["bay"]
        allowed = set(band) & eligibility[bay - 1]
        # Keep the neighborhood local, but include a two-hop choice when the
        # focus chain needs one relay through an intermediate crane.
        if block_is_focus(block, current_focus[0]):
            requested_owner = (
                current_focus[0].get("focus_task_targets", {}).get(
                    int(block["id"])
                )
            )
            if requested_owner is None:
                requested_owner = current_focus[0].get("target_owner")
            if requested_owner is not None:
                allowed &= {int(requested_owner)}
            elif current_focus[0].get("focus_type") != "idle_capacity":
                allowed &= {q for q in band if abs(q - source_owner) <= 2}
        elif current_focus[0].get("focus_type") == "idle_capacity":
            # Load-balance transactions change only the explicitly split
            # relay slices.  Other ownership changes belong to a later local
            # transaction and would unnecessarily enlarge this neighborhood.
            return [source_owner]
        elif current_focus[0].get("focus_type") != "idle_capacity":
            allowed &= {q for q in band if abs(q - source_owner) <= 2}
        max_hops = M if current_focus[0].get("focus_type") == "idle_capacity" else 2
        choices = [source_owner]
        choices.extend(
            q for q in sorted(allowed)
            if q != source_owner and abs(q - source_owner) <= max_hops
        )
        return list(dict.fromkeys(choices))

    current_focus: list[dict[str, Any]] = [{}]

    def assignment_variants(
        band: tuple[int, ...], focus: dict[str, Any],
        tasks: list[dict[str, int]],
    ) -> list[tuple[int, ...]]:
        current_focus[0] = focus
        if not tasks:
            return []
        by_id = {block["id"]: block for block in tasks}
        focus_ids = [block["id"] for block in tasks if block_is_focus(block, focus)]
        if not focus_ids:
            return []
        base = tuple(block["source_owner"] for block in tasks)
        local_choices = {
            block["id"]: owner_choices(block, band)
            for block in tasks
        }
        local_seen: set[tuple[int, ...]] = set()
        variants: list[tuple[int, ...]] = [base]
        forced_targets = {
            int(task_id): int(target)
            for task_id, target in focus.get(
                "focus_task_targets", {}
            ).items()
        }
        # First branch only moves the focus phase.  Additional branches move a
        # nearby phase too, which is the minimum useful representation of a
        # two-hop hand-off around a safety conflict.
        related = sorted(
            (block for block in tasks if block["id"] not in focus_ids),
            key=lambda block: (
                min(
                    abs(block["start"] - by_id[fid]["start"])
                    for fid in focus_ids
                ),
                block["start"],
                block["source_owner"],
            ),
        )[:8]

        def add_variant(values: dict[int, int]) -> None:
            if len(variants) >= max_assignment_variants:
                return
            result = tuple(values.get(block["id"], block["source_owner"]) for block in tasks)
            if result == base or result in local_seen:
                return
            # Do not manufacture a third owner for the same bay in this
            # bounded phase neighborhood; a later transaction can introduce a
            # second band if needed.
            owners_by_bay: dict[int, set[int]] = {}
            for block, owner in zip(tasks, result):
                owners_by_bay.setdefault(block["bay"], set()).add(owner)
            if any(len(owners) > 2 for owners in owners_by_bay.values()):
                return
            local_seen.add(result)
            variants.append(result)

        if len(forced_targets) > 1:
            add_variant(forced_targets)
            return variants

        for focus_id in focus_ids:
            for target in local_choices[focus_id]:
                if target == by_id[focus_id]["source_owner"]:
                    continue
                stats["owner_change_branches"] += 1
                if abs(target - by_id[focus_id]["source_owner"]) == 2:
                    stats["two_hop_relay_branches"] += 1
                add_variant({focus_id: target})
                for other in related:
                    if other["id"] == focus_id:
                        continue
                    for other_target in local_choices[other["id"]]:
                        if other_target == other["source_owner"]:
                            continue
                        add_variant({
                            focus_id: target,
                            other["id"]: other_target,
                        })
        return variants

    def tasks_for_focus(
        band: tuple[int, ...], focus: dict[str, Any]
    ) -> list[dict[str, int]]:
        tasks = [
            dict(block) for block in source_blocks
            if block["source_owner"] in band
        ]
        if focus.get("focus_type") != "idle_capacity":
            return tasks
        focus_task_ids: list[int] = []
        focus_task_targets: dict[int, int] = {}

        def split_transfer(
            source_id: int,
            requested_transfer: int,
            side: str,
            target_owner: int,
        ) -> bool:
            source_block = next(
                (block for block in tasks if block["id"] == source_id),
                None,
            )
            if source_block is None:
                return False
            transfer = min(
                int(source_block["length"]), int(requested_transfer)
            )
            if transfer <= 0:
                return False
            mandatory_prefix = int(
                int(source_block["start"]) == 0
                and int(source_block["bay"]) in set(starts)
            )
            if transfer == int(source_block["length"]):
                transferable_id = int(source_block["id"])
            else:
                next_id = max(
                    (int(task["id"]) for task in tasks), default=-1
                ) + 1
                original_start = int(source_block["start"])
                original_end = int(source_block["end"])
                if side == "prefix":
                    transfer_start = original_start + mandatory_prefix
                    transfer_end = transfer_start + transfer
                else:
                    transfer_start = original_end - transfer
                    transfer_end = original_end
                segments: list[dict[str, int]] = []
                segment_id = next_id
                for left, right in (
                    (original_start, transfer_start),
                    (transfer_end, original_end),
                ):
                    if right <= left:
                        continue
                    segments.append({
                        **source_block,
                        "id": segment_id,
                        "start": left,
                        "end": right,
                        "length": right - left,
                    })
                    segment_id += 1
                transferable = {
                    **source_block,
                    "id": segment_id,
                    "start": transfer_start,
                    "end": transfer_end,
                    "length": transfer,
                }
                position = next(
                    i for i, task in enumerate(tasks)
                    if task["id"] == source_id
                )
                tasks[position:position + 1] = [
                    *segments, transferable
                ]
                transferable_id = int(transferable["id"])
            focus_task_ids.append(transferable_id)
            focus_task_targets[transferable_id] = int(target_owner)
            return True

        if not split_transfer(
            int(focus["focus_source_block_id"]),
            int(focus["transfer_length"]),
            str(focus.get("transfer_side", "suffix")),
            int(focus["target_owner"]),
        ):
            return []
        focus["focus_task_id"] = focus_task_ids[0]
        relay_source_id = focus.get("relay_source_block_id")
        if relay_source_id is not None and not split_transfer(
            int(relay_source_id),
            int(focus.get("relay_transfer_length", focus["transfer_length"])),
            str(focus.get("relay_transfer_side", "suffix")),
            int(focus["relay_target_owner"]),
        ):
            return []
        focus["focus_task_ids"] = list(focus_task_ids)
        focus["focus_task_targets"] = dict(focus_task_targets)
        return tasks

    def solve_positions(
        band: tuple[int, ...],
        assigned_owners: tuple[int, ...],
        tasks: list[dict[str, int]],
        retimed_work_plan: dict[tuple[int, int], int | None] | None = None,
    ) -> tuple[list[tuple[int, ...]], dict[tuple[int, int], int | None]] | None:
        work_plan = (
            dict(retimed_work_plan)
            if retimed_work_plan is not None
            else dict(source_work)
        )
        occupied: dict[tuple[int, int], int] = {}
        # Rewrite the complete active-band ledger atomically.  Overlap is
        # rejected before any position search, so no partial transfer leaks.
        if retimed_work_plan is None:
            for task in tasks:
                for t in range(task["start"], task["end"]):
                    work_plan[(t, task["source_owner"])] = None
            for task, new_owner in zip(tasks, assigned_owners):
                bay = task["bay"]
                for t in range(task["start"], task["end"]):
                    key = (t, new_owner)
                    old_task = occupied.get(key)
                    if old_task is not None and old_task != task["id"]:
                        stats["rejected_overlap"] += 1
                        return None
                    occupied[key] = task["id"]
                    work_plan[key] = bay

        # The active-band ledger must be exactly the source ledger by bay.
        for bay, required in enumerate(W, 1):
            if sum(value == bay for value in work_plan.values()) != int(required):
                stats["rejected_ledger"] += 1
                return None
        for (t, q), bay in work_plan.items():
            if bay is not None and q not in eligibility[int(bay) - 1]:
                stats["rejected_eligibility"] += 1
                return None

        # Build fixed work positions.  Idle cells are deliberately left free;
        # the row DP can move a donor away and a receiver into the transferred
        # phase in the same atomic transaction.
        fixed_work: list[dict[int, int]] = [dict() for _ in range(horizon)]
        for (t, q), bay in work_plan.items():
            if bay is not None:
                fixed_work[t][q] = int(bay)

        options_cache: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}

        def row_options(t: int) -> list[tuple[int, ...]]:
            key = (
                tuple(sorted(fixed_work[t].items())),
                tuple((q, source_rows[t][q]) for q in range(M) if q not in band),
                t == 0,
            )
            cached = options_cache.get(key)
            if cached is not None:
                return cached
            result: list[tuple[int, ...]] = []
            for config in legal_configurations:
                if t == 0 and config != source_rows[0]:
                    continue
                if any(config[q] != position for q, position in key[1]):
                    continue
                if any(config[q] != bay for q, bay in fixed_work[t].items()):
                    continue
                result.append(config)
            options_cache[key] = result
            return result

        first_options = row_options(0)
        if not first_options:
            stats["rejected_safety"] += 1
            return None
        # ``parents[t][row] = previous_row``.  Movement into the terminal
        # boundary row H is not counted in K, matching the candidate metric.
        parents: list[dict[tuple[int, ...], tuple[int, ...] | None]] = [
            {source_rows[0]: None}
        ]
        costs: dict[tuple[int, ...], tuple[int, int]] = {
            source_rows[0]: (0, 0)
        }
        for t in range(1, horizon + 1):
            if expired():
                return None
            options = row_options(t if t < horizon else horizon - 1)
            # The final row has no work cell; it only needs to remain safe and
            # keep inactive cranes frozen.  Use the same work constraints as
            # the preceding boundary when the last slot is active.
            if t == horizon:
                options = [
                    config for config in legal_configurations
                    if all(
                        config[q] == source_rows[t][q]
                        for q in range(M) if q not in band
                    )
                    and all(
                        config[q] == bay
                        for q, bay in fixed_work[horizon - 1].items()
                    )
                ]
            current: dict[tuple[int, ...], tuple[int, int]] = {}
            current_parents: dict[tuple[int, ...], tuple[int, ...] | None] = {}
            for row in options:
                best_item: tuple[tuple[int, int], tuple[int, ...] | None] | None = None
                for previous, (move_cost, source_distance) in costs.items():
                    transition = sum(
                        before != after
                        for before, after in zip(previous, row)
                    )
                    # The edge into row H is a terminal boundary and is not a
                    # completed move when the last work ends at H.
                    counted = transition if t < horizon else 0
                    item = (
                        move_cost + counted,
                        source_distance + sum(
                            abs(int(row[q]) - int(source_rows[t][q]))
                            for q in band
                        ),
                    )
                    if best_item is None or item < best_item[0]:
                        best_item = (item, previous)
                if best_item is None:
                    continue
                current[row] = best_item[0]
                current_parents[row] = best_item[1]
            stats["states_expanded"] += len(current)
            if stats["states_expanded"] >= state_limit:
                stats["status"] = "UNKNOWN_STATE_LIMIT"
                stats["state_limit"] += 1
                return None
            if not current:
                stats["rejected_safety"] += 1
                return None
            costs = current
            parents.append(current_parents)

        final_row = min(costs, key=lambda row: costs[row])
        history: list[tuple[int, ...]] = [final_row]
        for t in range(horizon, 0, -1):
            previous = parents[t].get(history[-1])
            if previous is None:
                break
            history.append(previous)
        history.reverse()
        if len(history) != horizon + 1:
            stats["rejected_safety"] += 1
            return None
        return history, work_plan

    def retimed_work_plans(
        band: tuple[int, ...],
        tasks: list[dict[str, int]],
        assigned_owners: tuple[int, ...],
        focus: dict[str, Any],
        max_variants: int = 24,
    ) -> list[dict[tuple[int, int], int | None]]:
        """Create a few event-time phase orders for one owner assignment."""
        by_crane: dict[int, list[dict[str, int]]] = {q: [] for q in band}
        for task, owner in zip(tasks, assigned_owners):
            by_crane.setdefault(owner, []).append(task)
        focus_ids = {
            block["id"] for block in tasks
            if block_is_focus(block, focus)
        }
        order_options: dict[int, list[list[dict[str, int]]]] = {}
        for q in band:
            crane_tasks = list(by_crane.get(q, []))
            crane_tasks.sort(key=lambda item: (item["start"], item["source_owner"], item["id"]))
            if not crane_tasks:
                order_options[q] = [[]]
                continue
            forced_first = [
                task for task in crane_tasks
                if task["source_owner"] == q and task["start"] == 0
            ]
            options: list[list[dict[str, int]]] = [list(crane_tasks)]
            for focus_task in crane_tasks:
                if focus_task["id"] not in focus_ids:
                    continue
                for index in range(len(crane_tasks)):
                    candidate_order = [
                        task for task in crane_tasks if task["id"] != focus_task["id"]
                    ]
                    candidate_order.insert(index, focus_task)
                    if forced_first and candidate_order[0]["id"] != forced_first[0]["id"]:
                        continue
                    if not any(
                        [task["id"] for task in candidate_order]
                        == [task["id"] for task in old]
                        for old in options
                    ):
                        options.append(candidate_order)
                    if len(options) >= 8:
                        break
                if len(options) >= 8:
                    break
            order_options[q] = options

        combinations: list[dict[int, list[dict[str, int]]]] = []

        def combine(index: int, selected: dict[int, list[dict[str, int]]]) -> None:
            if len(combinations) >= max_variants:
                return
            if index >= len(band):
                combinations.append({q: list(value) for q, value in selected.items()})
                return
            q = band[index]
            for option in order_options.get(q, [[]]):
                selected[q] = option
                combine(index + 1, selected)
                if len(combinations) >= max_variants:
                    return
            selected.pop(q, None)

        combine(0, {})
        plans: list[dict[tuple[int, int], int | None]] = []
        seen_plans: set[tuple[tuple[tuple[int, int], int | None], ...]] = set()
        for combination in combinations:
            placements: list[tuple[dict[str, int], int, int]] = []
            placed_work: dict[tuple[int, int], int] = {}
            valid = True
            for q in band:
                cursor = 0
                crane_tasks = combination.get(q, [])
                # Keep an original t=0 phase at t=0; this preserves the
                # mandatory opening work while allowing later relay phases to
                # move earlier or later as one event-time block.
                for task in crane_tasks:
                    start_t = cursor
                    if task["source_owner"] == q and task["start"] == 0:
                        if cursor != 0:
                            valid = False
                            break
                        start_t = 0
                    else:
                        # Keep the transferred slice inside its source-side
                        # local time window, but let the receiver's existing
                        # phases slide left into the newly available idle
                        # capacity.  Freezing every phase at its old start
                        # left an artificial gap between the transferred
                        # suffix and an existing same-bay receiver phase (for
                        # example Q4@bay13), turning one useful hand-off into
                        # two work blocks and an avoidable return move.
                        is_idle_balance = (
                            focus.get("focus_type") == "idle_capacity"
                        )
                        release = (
                            max(cursor, min(task["start"], horizon))
                            if not is_idle_balance or task["id"] in focus_ids
                            else cursor
                        )
                        start_t = release
                        # Compact to the earliest locally safe interval.  A
                        # simple left shift can collide with a frozen neighbor
                        # (H208 Q2 remains at bay 9 while Q3 would like to move
                        # to bay 10).  Scan only this phase's bounded window and
                        # insert exactly the wait forced by neighboring work.
                        while (
                            is_idle_balance
                            and start_t + int(task["length"]) <= horizon
                        ):
                            interval_safe = True
                            for t in range(
                                start_t, start_t + int(task["length"])
                            ):
                                for other in range(M):
                                    if other == q:
                                        continue
                                    other_bay = (
                                        placed_work.get((t, other))
                                        if other in band else source_rows[t][other]
                                    )
                                    if other_bay is None:
                                        continue
                                    if (
                                        other < q
                                        and int(task["bay"]) - int(other_bay) < 2
                                    ) or (
                                        other > q
                                        and int(other_bay) - int(task["bay"]) < 2
                                    ):
                                        interval_safe = False
                                        break
                                if not interval_safe:
                                    break
                            if interval_safe:
                                break
                            start_t += 1
                    end_t = start_t + task["length"]
                    if end_t > horizon:
                        valid = False
                        break
                    placements.append((task, start_t, end_t))
                    for t in range(start_t, end_t):
                        placed_work[(t, q)] = int(task["bay"])
                    cursor = end_t
                if not valid:
                    break
            if not valid:
                continue
            plan = dict(source_work)
            for task in tasks:
                for t in range(task["start"], task["end"]):
                    plan[(t, task["source_owner"])] = None
            occupied: set[tuple[int, int]] = set()
            for task, start_t, end_t in placements:
                owner = next(
                    q for q, values in combination.items()
                    if any(item["id"] == task["id"] for item in values)
                )
                for t in range(start_t, end_t):
                    key = (t, owner)
                    if key in occupied:
                        valid = False
                        break
                    occupied.add(key)
                    plan[key] = task["bay"]
                if not valid:
                    break
            if not valid:
                continue
            signature = tuple(sorted(plan.items()))
            if signature in seen_plans:
                continue
            seen_plans.add(signature)
            plans.append(plan)
            if len(plans) >= max_variants:
                break
        return plans

    for focus in focuses:
        if expired() or len(proposals) >= max_candidates:
            break
        focus_q = int(focus.get("crane", 0)) - 1
        # Load diffusion is intentionally a two-crane neighborhood.  Once the
        # donor and its adjacent receiver are in the band, adding more cranes
        # turns a local repair into a much larger reschedule and multiplies
        # infeasible assignment combinations.  If another pair also needs
        # work, the outer polish loop performs a separate verified relay.
        widths = (
            (3,) if (
                focus.get("focus_type") == "idle_capacity"
                and "relay_source_block_id" in focus
            ) else (2,) if focus.get("focus_type") == "idle_capacity"
            else range(2, M + 1)
        )
        for width in widths:
            left_min = max(0, focus_q - width + 1)
            left_max = min(focus_q, M - width)
            for left in range(left_min, left_max + 1):
                band = tuple(range(left, left + width))
                focus_signature = (
                    focus.get("focus_type", "revisit"), focus_q,
                    int(focus.get("bay", 0)),
                    int(focus.get("transfer_length", 0)),
                    int(focus.get("target_owner", -1)),
                    focus.get("transfer_side", ""),
                    int(focus.get("focus_source_block_id", -1)),
                    int(focus.get("relay_source_block_id", -1)),
                )
                band_key = (focus_signature, band)
                if band_key in tested_bands:
                    continue
                tested_bands.add(band_key)
                stats["activity_bands_tested"] += 1
                stats["max_activity_width"] = max(
                    stats["max_activity_width"], len(band)
                )
                tasks = tasks_for_focus(band, focus)
                if focus.get("focus_type") == "idle_capacity":
                    target_owner = int(focus["target_owner"])
                    if target_owner not in band:
                        continue
                variants = assignment_variants(band, focus, tasks)
                expansion = {
                    "focus": {"crane": focus_q + 1, "bay": int(focus.get("bay", 0))},
                    "active_cranes": [q + 1 for q in band],
                    "task_count": len(tasks),
                    "assignment_variants": len(variants),
                    "verified_candidates": 0,
                }
                for assigned_owners in variants:
                    if expired() or len(proposals) >= max_candidates:
                        break
                    if stats["assignment_variants_generated"] >= max_assignment_variants:
                        stats["status"] = "UNKNOWN_STATE_LIMIT"
                        stats["state_limit"] += 1
                        break
                    stats["assignment_variants_generated"] += 1
                    if all(
                        owner == task["source_owner"]
                        for owner, task in zip(assigned_owners, tasks)
                    ):
                        continue
                    retimed_plans = retimed_work_plans(
                        band, tasks, assigned_owners, focus
                    )
                    stats["retimed_plans_generated"] += len(retimed_plans)
                    # For load balancing, try the locally compacted event
                    # plans before the unchanged timestamps.  The unchanged
                    # plan is legal but often leaves a gap between a handed-
                    # off suffix and the receiver's existing same-bay phase,
                    # consuming the candidate cap with fragmented variants.
                    plans_to_try: list[
                        dict[tuple[int, int], int | None] | None
                    ] = (
                        [*retimed_plans, None]
                        if focus.get("focus_type") == "idle_capacity"
                        else [None, *retimed_plans]
                    )
                    for retimed_plan in plans_to_try:
                        if expired() or len(proposals) >= max_candidates:
                            break
                        solved = solve_positions(
                            band, assigned_owners, tasks, retimed_plan
                        )
                        if solved is None:
                            continue
                        if retimed_plan is not None:
                            stats["retimed_plans_solved"] += 1
                        history, work_plan = solved
                        stats["complete_phase_plans"] += 1
                        stats["ledger_closed"] += 1
                        signature = (
                            tuple(history), tuple(sorted(work_plan.items()))
                        )
                        if signature in seen:
                            continue
                        seen.add(signature)
                        try:
                            candidate = _candidate_from_rows_and_work_plan(
                                W, M, history, work_plan
                            )
                        except RuntimeError:
                            stats["rejected_safety"] += 1
                            continue
                        stats["decoded"] += 1
                        if not _candidate_passes_independent_verifier(W, M, starts, candidate):
                            stats["rejected_safety"] += 1
                            continue
                        stats["verified"] += 1
                        expansion["verified_candidates"] += 1
                        owner_changes = [
                            {
                                "bay": task["bay"],
                                "start": task["start"],
                                "end_exclusive": task["end"],
                                "length": task["length"],
                                "from_crane": task["source_owner"] + 1,
                                "to_crane": owner + 1,
                            }
                            for task, owner in zip(tasks, assigned_owners)
                            if owner != task["source_owner"]
                        ]
                        candidate_balance = _balanced_schedule_metrics(candidate, M)
                        operator_name = (
                            "idle_capacity_rebalance"
                            if focus.get("focus_type") == "idle_capacity"
                            else "cross_crane_phase_relay"
                        )
                        proposals.append({
                        "operator": operator_name,
                        "history": [tuple(row) for row in history],
                        "work_plan": work_plan,
                        "active_cranes": list(band),
                        "regions": [
                            {
                                "crane": q + 1,
                                "start": 0,
                                "end_exclusive": horizon,
                                "length": horizon,
                                "segment": 0,
                            }
                            for q in band
                        ],
                        "source_signature": _trajectory_signature(source),
                        "source_hash": source_hash,
                        "details": {
                            "focus": {
                                "type": focus.get("focus_type", "revisit"),
                                "crane": focus_q + 1,
                                "bay": int(focus.get("bay", 0)),
                                "blocks": focus.get("blocks", []),
                                "gap": int(focus.get("gap", 0)),
                                "target_crane": (
                                    int(focus["target_owner"]) + 1
                                    if focus.get("target_owner") is not None else None
                                ),
                                "transfer_length": focus.get("transfer_length"),
                                "transfer_side": focus.get("transfer_side"),
                                "atomic_chain": bool(
                                    focus.get("relay_source_block_id")
                                    is not None
                                ),
                            },
                            "active_cranes": [q + 1 for q in band],
                            "owner_changes": owner_changes,
                            "owner_change_count": len(owner_changes),
                            "two_hop_relay": any(
                                abs(item["from_crane"] - item["to_crane"]) == 2
                                for item in owner_changes
                            ),
                            "candidate_objective": list(candidate.objective_key),
                            "candidate_movement_count": candidate.completion_movement_count,
                            "candidate_balance": {
                                **candidate_balance,
                                "key": list(candidate_balance["key"]),
                            },
                            "work_ledger": {
                                "source_by_bay": [
                                    sum(value == bay for value in source_work.values())
                                    for bay in range(1, N + 1)
                                ],
                                "transaction_by_bay": [
                                    sum(value == bay for value in work_plan.values())
                                    for bay in range(1, N + 1)
                                ],
                                "closed": True,
                                "work_transfer": True,
                            },
                            "event_level": True,
                        },
                        })
                        if candidate.objective_key <= source.objective_key:
                            stats["accepted"] += 1
                stats["activity_chain_expansions"].append(expansion)
                if stats["status"] == "UNKNOWN_STATE_LIMIT":
                    break
            if stats["status"] == "UNKNOWN_STATE_LIMIT":
                break
        if stats["status"] == "UNKNOWN_STATE_LIMIT":
            break

    proposals.sort(key=lambda item: (
        (
            tuple(item["details"]["candidate_balance"]["key"])
            if item["operator"] == "idle_capacity_rebalance"
            else tuple(item["details"]["candidate_objective"])
        ),
        int(item["details"].get("owner_change_count", 0)),
        tuple(item["active_cranes"]),
    ))
    if proposals and stats["status"] == "UNKNOWN_STATE_LIMIT":
        # The bounded assignment cap stopped enumeration after complete,
        # independently verified candidates were already produced.  Expose
        # that useful result as a candidate-found status; only an empty
        # capped search remains UNKNOWN.
        stats["status"] = "CANDIDATE_FOUND"
    return proposals[:max_candidates], stats


def _build_relay_transaction(
    rows: Sequence[tuple[int, ...]],
    work_plan: dict[tuple[int, int], int | None],
    chain: Sequence[int],
    start: int,
    end: int,
    *,
    operator: str,
    source_signature: tuple[Any, ...],
    source_hash: str | None = None,
) -> dict[str, Any] | None:
    """Construct a one-segment adjacent relay from an explicit work map.

    ``chain`` is either ``[donor, receiver]`` or a three-crane chain with the
    final crane idle.  Work moves one position to the right or left while the
    outer donor leaves the work position.  The surrounding context slot is
    included in the declared region so no outside transition is silently
    changed.
    """
    horizon = len(rows) - 1
    if not 0 <= start < end <= horizon:
        return None
    region_start = start if start == 0 else start - 1
    region_end = end
    if region_end - region_start > 8:
        return None
    if len(chain) not in (2, 3):
        return None
    direction = 1 if chain[-1] > chain[0] else -1
    if list(chain) != list(range(chain[0], chain[-1] + direction, direction)):
        return None
    chain = tuple(chain)
    if any(q < 0 or q >= len(rows[0]) for q in chain):
        return None

    physical = chain if direction > 0 else tuple(reversed(chain))
    bays: list[int] = []
    if direction > 0:
        for q in physical[:-1]:
            values = {work_plan.get((t, q)) for t in range(start, end)}
            if len(values) != 1 or None in values:
                return None
            bays.append(int(next(iter(values))))
        if any(work_plan.get((t, physical[-1])) is not None for t in range(start, end)):
            return None
    else:
        for q in physical[1:]:
            values = {work_plan.get((t, q)) for t in range(start, end)}
            if len(values) != 1 or None in values:
                return None
            bays.append(int(next(iter(values))))
        if any(work_plan.get((t, physical[0])) is not None for t in range(start, end)):
            return None

    def safe_rows(outer_position: int) -> list[tuple[int, ...]] | None:
        changed = [list(row) for row in rows]
        for t in range(start, end):
            if direction > 0:
                changed[t][physical[0]] = outer_position
                for index, bay in enumerate(bays, 1):
                    changed[t][physical[index]] = bay
            else:
                for index, bay in enumerate(bays):
                    changed[t][physical[index]] = bay
                changed[t][physical[-1]] = outer_position
            if any(
                right - left < 2
                for left, right in zip(changed[t], changed[t][1:])
            ):
                return None
        return [tuple(row) for row in changed]

    # Prefer a legal outward position, then the closest legal in-rail position.
    N = max(max(row) for row in rows if row)
    outer_q = physical[0] if direction > 0 else physical[-1]
    target_edge = bays[0] if direction > 0 else bays[-1]
    choices = [-1] + list(range(1, N + 1)) + [N + 2]
    choices.sort(key=lambda value: (
        0 if value in (-1, N + 2) else 1,
        abs(value - target_edge),
    ))
    proposed_rows: list[tuple[int, ...]] | None = None
    chosen_outer = None
    for outer in choices:
        candidate_rows = safe_rows(outer)
        if candidate_rows is not None:
            proposed_rows = candidate_rows
            chosen_outer = outer
            break
    if proposed_rows is None or chosen_outer is None:
        return None

    proposed_work = dict(work_plan)
    if direction > 0:
        for t in range(start, end):
            for q in physical[:-1]:
                proposed_work[(t, q)] = None
            for index, bay in enumerate(bays, 1):
                proposed_work[(t, physical[index])] = bay
    else:
        for t in range(start, end):
            for q in physical[1:]:
                proposed_work[(t, q)] = None
            for index, bay in enumerate(bays):
                proposed_work[(t, physical[index])] = bay

    segment = {
        "crane": chain[0] + 1,
        "start": region_start,
        "end_exclusive": region_end,
        "length": region_end - region_start,
        "segment": 0,
    }
    regions = [dict(segment, crane=q + 1) for q in physical]
    return {
        "operator": operator,
        "history": proposed_rows,
        "work_plan": proposed_work,
        "regions": regions,
        "source_signature": source_signature,
        "source_hash": source_hash,
        "details": {
            "chain": [q + 1 for q in physical],
            "direction": "right" if direction > 0 else "left",
            "source_interval": [start, end],
            "declared_interval": [region_start, region_end],
            "outer_crane": outer_q + 1,
            "outer_position": chosen_outer,
            "transferred_bays": bays,
            "work_ledger": {
                "before": [
                    {"crane": q + 1, "bay": work_plan.get((start, q))}
                    for q in physical
                ],
                "after": [
                    {"crane": q + 1, "bay": proposed_work.get((start, q))}
                    for q in physical
                ],
            },
        },
    }


def _work_transfer_transactions(
    W: Sequence[int],
    M: int,
    source: _CandidateSchedule,
    *,
    max_candidates: int = 128,
    source_hash: str | None = None,
    enable_multi_relay: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate real adjacent work-transfer transactions from one source.

    The generator only proposes blocks whose receiver is idle for the whole
    interval.  It therefore cannot claim a benefit merely because a receiver
    has a large trailing idle suffix.  A candidate must still pass the strict
    decoder and the independent verifier at the caller.
    """
    stats: dict[str, Any] = {
        "generated": 0,
        "unique": 0,
        "capacity_rejected": 0,
        "safety_rejected": 0,
        "boundary_rejected": 0,
        "verified": 0,
        "accepted": 0,
        "timeout": 0,
        "operators": {
            "paired_residual_exchange": 0,
            "tail_relay": 0,
            "idle_fill": 0,
        },
    }
    if source.move_time != 0:
        stats["unsupported"] = "nonzero_move_time"
        return [], stats
    rows, work_plan = _transaction_rows_work_map(source, M)
    horizon = source.makespan
    blocks = _work_blocks_for_transaction(work_plan, M, horizon)
    proposals: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def interval_choices(left: int, right: int) -> list[tuple[int, int]]:
        if left >= right:
            return []
        result: set[tuple[int, int]] = set()
        max_length = min(8, right - left)
        for length in (1, 2, 3, 4, 6, 7, 8):
            if length > max_length:
                continue
            for start in (left, right - length, left + (right - left - length) // 2):
                if left <= start and start + length <= right:
                    result.add((start, start + length))
        return sorted(result, key=lambda value: (-(value[1] - value[0]), value[0]))

    def add(operator: str, chain: Sequence[int], left: int, right: int) -> None:
        if len(proposals) >= max_candidates:
            return
        for start, end in interval_choices(left, right):
            if len(proposals) >= max_candidates:
                return
            item = _build_relay_transaction(
                rows,
                work_plan,
                chain,
                start,
                end,
                operator=operator,
                source_signature=_trajectory_signature(source),
                source_hash=source_hash,
            )
            stats["generated"] += 1
            stats["operators"][operator] += 1
            if item is None:
                stats["safety_rejected"] += 1
                continue
            signature = (
                tuple(item["history"]),
                tuple(sorted(item["work_plan"].items())),
            )
            if signature in seen:
                stats["unique"] += 0
                continue
            seen.add(signature)
            item["details"]["source_signature"] = list(
                item["source_signature"]
            )
            proposals.append(item)
            stats["unique"] += 1

    def add_temporal_residual_exchange(
        q: int,
        bay: int,
        left_end: int,
        right_start: int,
        right_end: int,
    ) -> None:
        """Move a late residual into an idle gap at the same position.

        This is the smallest genuine continuity transaction: it changes the
        work ledger at two short, explicitly declared segments while keeping
        the physical trajectory fixed.  It is useful when a crane waits at a
        bay and then returns to the same bay later; no global decoder can
        discover this without inventing a new work assignment.
        """
        gap_start, gap_end = left_end, right_start
        gap_length = gap_end - gap_start
        if not 0 < gap_length <= 8 or right_end - right_start < gap_length:
            return
        if any(rows[t][q] != bay for t in range(gap_start, gap_end)):
            return
        proposed_work = dict(work_plan)
        for t in range(gap_start, gap_end):
            proposed_work[(t, q)] = bay
        # Shift the late block left by the gap length.  Removing its *tail*
        # (rather than its first cells) keeps the old right block connected to
        # the newly filled gap and therefore actually closes the visit.
        late_tail_start = right_end - gap_length
        for t in range(late_tail_start, right_end):
            proposed_work[(t, q)] = None
        regions = [
            {
                "crane": q + 1,
                "start": gap_start,
                "end_exclusive": gap_end,
                "length": gap_length,
                "segment": 0,
            },
            {
                "crane": q + 1,
                "start": late_tail_start,
                "end_exclusive": right_end,
                "length": gap_length,
                "segment": 1,
            },
        ]
        signature = (
            tuple(rows),
            tuple(sorted(proposed_work.items())),
        )
        if signature in seen or len(proposals) >= max_candidates:
            return
        seen.add(signature)
        proposals.append({
            "operator": "paired_residual_exchange",
            "history": [tuple(row) for row in rows],
            "work_plan": proposed_work,
            "regions": regions,
            "source_signature": _trajectory_signature(source),
            "source_hash": source_hash,
            "details": {
                "crane": q + 1,
                "bay": bay,
                "source_gap": [gap_start, gap_end],
                "source_late_residual": [late_tail_start, right_end],
                "declared_segments": [
                    [gap_start, gap_end],
                    [late_tail_start, right_end],
                ],
                "work_ledger": {
                    "moved_count": gap_length,
                    "from": [late_tail_start, right_end],
                    "to": [gap_start, gap_end],
                },
            },
        })
        stats["generated"] += 1
        stats["unique"] += 1
        stats["operators"]["paired_residual_exchange"] += 1

    # First enumerate same-bay gap closures.  The blocks are built per crane,
    # so this cannot accidentally merge work from a different crane or bay.
    blocks_by_crane: dict[int, list[dict[str, int]]] = {
        q: sorted((item for item in blocks if item["crane"] == q),
                  key=lambda item: item["start"])
        for q in range(M)
    }
    for q, crane_blocks in blocks_by_crane.items():
        for left, right in zip(crane_blocks, crane_blocks[1:]):
            if left["bay"] != right["bay"]:
                continue
            add_temporal_residual_exchange(
                q, left["bay"], left["end"], right["start"], right["end"]
            )

    # A pair is a real transfer whenever a work block overlaps an adjacent
    # crane's idle cells.  Classify short early/late pieces separately so the
    # report can distinguish residual exchanges from generic idle filling.
    for block in blocks:
        q = block["crane"]
        for direction in (-1, 1):
            receiver = q + direction
            if not 0 <= receiver < M:
                continue
            left, right = block["start"], block["end"]
            idle_times = [
                t for t in range(left, right)
                if work_plan.get((t, receiver)) is None
            ]
            if not idle_times:
                continue
            start = min(idle_times)
            end = start
            while end < right and work_plan.get((end, receiver)) is None:
                end += 1
            operator = (
                "tail_relay"
                if end == horizon or start >= max(0, horizon - 12)
                else "paired_residual_exchange"
                if block["length"] <= 3
                else "idle_fill"
            )
            # The donor is first in the directed chain.  A leftward transfer
            # is therefore deliberately passed in descending crane order.
            chain = [q, receiver]
            add(operator, chain, start, end)

    # A depth-two adjacent relay moves q0's work to q1 and q1's work to an
    # idle q2 (or the mirror image).  This is the smallest transaction that
    # can explain a genuine Q1 -> Q2 -> Q3 hand-off.
    for left_crane in range(M - 2) if enable_multi_relay else ():
        chain = [left_crane, left_crane + 1, left_crane + 2]
        overlap = [
            t for t in range(horizon)
            if work_plan.get((t, chain[0])) is not None
            and work_plan.get((t, chain[1])) is not None
            and work_plan.get((t, chain[2])) is None
        ]
        if overlap:
            start = min(overlap)
            end = start
            while end < horizon and (
                work_plan.get((end, chain[0])) is not None
                and work_plan.get((end, chain[1])) is not None
                and work_plan.get((end, chain[2])) is None
            ):
                end += 1
            add("tail_relay", chain, start, end)
        right_chain = list(reversed(chain))
        overlap = [
            t for t in range(horizon)
            if work_plan.get((t, right_chain[0])) is None
            and work_plan.get((t, right_chain[1])) is not None
            and work_plan.get((t, right_chain[2])) is not None
        ]
        if overlap:
            start = min(overlap)
            end = start
            while end < horizon and (
                work_plan.get((end, right_chain[0])) is None
                and work_plan.get((end, right_chain[1])) is not None
                and work_plan.get((end, right_chain[2])) is not None
            ):
                end += 1
            add("tail_relay", right_chain, start, end)
    return proposals, stats


def _paired_window_cyclic_work_exchange(
    W: Sequence[int],
    M: int,
    source: _CandidateSchedule,
    *,
    max_candidates: int = 64,
    state_limit: int = 256,
    source_hash: str | None = None,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Propose bounded early/late relay cycles for short work residuals.

    This first implementation targets a one-slot residual after a long gap.
    It moves the residual into the next slot after its primary block, relays
    the displaced work through two or three adjacent cranes, and closes the
    work ledger in a separate tail window.  A second tail variant lets the
    residual crane take over its neighbor's continuous tail block, which can
    remove the unnecessary return move while keeping every edited slot
    explicit.  Both windows are at most eight slots and at most three adjacent
    cranes participate.

    The generator is intentionally finite and conservative.  An empty result
    or a state limit is UNKNOWN for this proposal family, never proof that a
    longer-horizon schedule is impossible.
    """
    stats: dict[str, Any] = {
        "status": "SEARCH_COMPLETE",
        "generated": 0,
        "unique": 0,
        "expanded_states": 0,
        "early_window_states": 0,
        "late_window_states": 0,
        "capacity_rejected": 0,
        "safety_rejected": 0,
        "boundary_rejected": 0,
        "ledger_rejected": 0,
        "verified": 0,
        "accepted": 0,
        "timeout": 0,
        "state_limit": 0,
        "operators": {
            "residual_absorb": 0,
            "cyclic_exchange": 0,
            "block_boundary_shift": 0,
            "pre_tail_compensation": 0,
        },
    }
    if source.move_time != 0:
        stats.update({"status": "UNSUPPORTED", "reason": "nonzero_move_time"})
        return [], stats
    if source.makespan < 4:
        stats.update({"status": "UNSUPPORTED", "reason": "horizon_too_short"})
        return [], stats

    rows, source_work = _transaction_rows_work_map(source, M)
    horizon = source.makespan
    diagnostics = _continuity_diagnostics(source, M)
    proposals: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def expired() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    revisits = sorted(
        diagnostics["crane_work_revisits"],
        key=lambda item: (
            -int(item.get("gap", 0)),
            int(item.get("residual_length", 0)),
            int(item["crane"]),
            int(item["bay"]),
        ),
    )
    for revisit in revisits:
        if expired() or len(proposals) >= max_candidates:
            stats["timeout"] += int(expired())
            if expired() or len(proposals) >= max_candidates:
                stats["status"] = "UNKNOWN"
                stats["limit_reason"] = "deadline" if expired() else "proposal_limit"
            break
        q = int(revisit["crane"]) - 1
        target_bay = int(revisit["bay"])
        primary = revisit["primary_block"]
        residual = revisit["residual_block"]
        early_time = int(primary["end_exclusive"])
        residual_time = int(residual["start"])
        residual_length = int(
            revisit.get("residual_length", residual["length"])
        )
        # Moving multi-slot residuals needs a wider early window and is left
        # to other operators; the intended failure is an isolated one-slot
        # tail completion.
        if residual_length != 1 or early_time >= horizon - 8:
            stats["capacity_rejected"] += 1
            continue
        tail_start = residual_time
        tail_end = horizon
        if not 0 <= early_time < early_time + 1 <= horizon:
            stats["boundary_rejected"] += 1
            continue
        if not 0 <= tail_start < tail_end <= horizon or tail_end - tail_start > 8:
            stats["boundary_rejected"] += 1
            continue
        if early_time + 1 > tail_start:
            stats["boundary_rejected"] += 1
            continue

        # Try relay chains extending to either side of the residual crane.
        chains: list[tuple[int, ...]] = []
        for direction in (-1, 1):
            for length in (2, 3):
                chain = tuple(q + direction * step for step in range(length))
                if any(not 0 <= crane < M for crane in chain):
                    continue
                if sorted(chain) != list(range(min(chain), max(chain) + 1)):
                    continue
                if chain not in chains:
                    chains.append(chain)

        for chain in chains:
            if expired() or len(proposals) >= max_candidates:
                stats["timeout"] += int(expired())
                if expired() or len(proposals) >= max_candidates:
                    stats["status"] = "UNKNOWN"
                    stats["limit_reason"] = "deadline" if expired() else "proposal_limit"
                break
            if stats["expanded_states"] >= max(1, state_limit):
                stats["state_limit"] += 1
                stats["status"] = "UNKNOWN"
                break
            stats["expanded_states"] += 1
            stats["early_window_states"] += 1
            stats["late_window_states"] += 1
            old_bays = [source_work.get((early_time, crane)) for crane in chain]
            if any(bay is None for bay in old_bays):
                stats["capacity_rejected"] += 1
                continue
            if chain[0] != q or old_bays[0] == target_bay:
                # A cyclic relay must start at the crane that owns the
                # residual; reversed chains are only useful when they do.
                stats["boundary_rejected"] += 1
                continue
            old_bays = [int(bay) for bay in old_bays]
            new_bays = [target_bay, *old_bays[:-1]]
            if len(set(new_bays)) != len(new_bays):
                stats["capacity_rejected"] += 1
                continue

            # Tail takeover is possible when the next crane continuously
            # works the displaced bay through H and the residual crane was
            # already positioned at that bay immediately before the tail.
            tail_receiver = chain[1] if len(chain) >= 2 else None
            receiver_bay = (
                source_work.get((tail_start, tail_receiver))
                if tail_receiver is not None else None
            )
            tail_takeover = bool(
                tail_receiver is not None
                and receiver_bay is not None
                and rows[tail_start - 1][q] == receiver_bay
                and all(
                    source_work.get((t, tail_receiver)) == receiver_bay
                    for t in range(tail_start, tail_end)
                )
            )
            # The terminal crane's old bay becomes the only uncompensated
            # early delta.  Find a same-position idle cell in the late window
            # to close it without relying on a greedy decoder.  If the
            # terminal crane is also the tail receiver, the takeover itself
            # creates that idle capacity at the first late slot.
            final_crane = chain[-1]
            final_bay = old_bays[-1]
            compensation_options = [
                (t, (t, t + 1), "tail")
                for t in range(tail_start, tail_end)
                if rows[t][final_crane] == final_bay
                and source_work.get((t, final_crane)) is None
                and all(
                    source_work.get((t, other)) != final_bay
                    for other in range(M) if other != final_crane
                )
            ]
            # A compensation slot immediately before the terminal window can
            # absorb a displaced bay while the crane is already waiting at
            # that bay.  This is the important Q2@bay9 case: it removes the
            # late one-slot return instead of creating a third work block.
            pre_tail_options = [
                (t, (t, t + 1), "pre_tail")
                for t in range(early_time + 1, tail_start)
                if rows[t][final_crane] == final_bay
                and source_work.get((t, final_crane)) is None
                and all(
                    source_work.get((t, other)) != final_bay
                    for other in range(M) if other != final_crane
                )
            ]
            # Prefer the first contiguous waiting slot.  Later slots leave a
            # needless idle hole inside the same-position residual block and
            # would recreate a third crane/bay work segment.
            if pre_tail_options:
                pre_tail_options = [
                    min(pre_tail_options, key=lambda item: (
                        item[0], -(item[1][1] - item[1][0])
                    ))
                ]
            # Use the earliest pre-tail slot as the primary closure.  Keeping
            # one option per relay variant also makes the bounded transaction
            # accounting deterministic; the tail option remains the fallback
            # when no compatible waiting slot exists.
            compensation_options = (
                pre_tail_options
                if pre_tail_options
                else compensation_options[:1]
            )
            if (
                tail_takeover
                and final_crane == tail_receiver
                and final_bay != receiver_bay
            ):
                compensation_options.insert(0, (tail_start, (tail_start, tail_start + 1), "tail"))
            if not compensation_options:
                stats["capacity_rejected"] += 1
                continue

            variants = [False, True] if tail_takeover else [False]
            for use_tail_takeover in variants:
                for compensation_time, compensation_interval, compensation_kind in compensation_options:
                    if expired() or len(proposals) >= max_candidates:
                        stats["timeout"] += int(expired())
                        break
                    stats["generated"] += 1
                    stats["operators"]["residual_absorb"] += 1
                    stats["operators"]["cyclic_exchange"] += 1
                    stats["operators"]["pre_tail_compensation"] += int(
                        compensation_kind == "pre_tail"
                    )
                    history = [list(row) for row in rows]
                    work_plan = dict(source_work)

                    # At the early slot, each crane takes the preceding
                    # crane's original bay.  This makes the relay a closed
                    # local chain instead of losing displaced work.
                    for index, crane in enumerate(chain):
                        history[early_time][crane] = new_bays[index]
                        work_plan[(early_time, crane)] = new_bays[index]

                    # The original late residual is removed.  With a tail
                    # takeover, its crane remains on the neighboring work
                    # bay and the neighbor is parked at a safe pre-tail
                    # position.
                    work_plan[(residual_time, q)] = None
                    hold_position = None
                    if use_tail_takeover:
                        assert tail_receiver is not None and receiver_bay is not None
                        hold_choices = list(
                            range(1, max(max(row) for row in rows) + 1)
                        )
                        # Keep the receiver at its pre-tail position when it
                        # is the terminal crane.  Returning to the
                        # compensated bay after bay10 is pure positional
                        # noise (the regression had 9->7->9->10->9).
                        preferred_hold = int(rows[tail_start - 1][tail_receiver])
                        hold_choices.sort(key=lambda position: (
                            position != preferred_hold,
                            abs(position - preferred_hold),
                            position,
                        ))
                        for proposed_hold in hold_choices:
                            safe_hold = True
                            for t in range(tail_start, tail_end):
                                check_row = list(history[t])
                                check_row[q] = int(receiver_bay)
                                check_row[tail_receiver] = proposed_hold
                                if any(
                                    right - left < 2
                                    for left, right in zip(check_row, check_row[1:])
                                ):
                                    safe_hold = False
                                    break
                            if safe_hold:
                                hold_position = proposed_hold
                                break
                        if hold_position is None:
                            stats["safety_rejected"] += 1
                            continue
                        for t in range(tail_start, tail_end):
                            history[t][q] = int(receiver_bay)
                            history[t][tail_receiver] = hold_position
                            work_plan[(t, q)] = int(receiver_bay)
                            work_plan[(t, tail_receiver)] = None

                    # If the tail donor is itself the terminal crane, its
                    # compensation slot replaces that donor's removed tail
                    # work.  The slot may be in the pre-tail waiting window.
                    work_plan[(compensation_time, final_crane)] = final_bay

                # A final ledger precheck catches impossible template matches
                # before the more expensive strict transaction decoder.
                def interval_delta(start: int, end: int) -> dict[int, int]:
                    delta = {bay: 0 for bay in range(1, len(W) + 1)}
                    for t in range(start, end):
                        for crane in chain:
                            old_bay = source_work.get((t, crane))
                            new_bay = work_plan.get((t, crane))
                            if old_bay is not None:
                                delta[int(old_bay)] -= 1
                            if new_bay is not None:
                                delta[int(new_bay)] += 1
                    return {bay: change for bay, change in delta.items() if change}

                intervals = [
                    (early_time, early_time + 1),
                    *(
                        [compensation_interval]
                        if compensation_kind == "pre_tail" else []
                    ),
                    (tail_start, tail_end),
                ]
                early_delta = interval_delta(early_time, early_time + 1)
                late_delta = {bay: 0 for bay in range(1, len(W) + 1)}
                for interval_start, interval_end in intervals[1:]:
                    delta = interval_delta(interval_start, interval_end)
                    for bay, change in delta.items():
                        late_delta[bay] += change
                late_delta = {
                    bay: change for bay, change in late_delta.items() if change
                }
                closed_delta = {
                    bay: early_delta.get(bay, 0) + late_delta.get(bay, 0)
                    for bay in set(early_delta) | set(late_delta)
                }
                closed_delta = {
                    bay: change for bay, change in closed_delta.items() if change
                }
                if closed_delta:
                    stats["ledger_rejected"] += 1
                    continue
                totals = [0] * len(W)
                for bay in work_plan.values():
                    if bay is not None:
                        totals[int(bay) - 1] += 1
                if totals != [int(value) for value in W]:
                    stats["ledger_rejected"] += 1
                    continue

                safe = all(
                    right - left >= 2
                    for row in history
                    for left, right in zip(row, row[1:])
                )
                if not safe:
                    stats["safety_rejected"] += 1
                    continue

                regions = [
                    {
                        "crane": crane + 1,
                        "start": start,
                        "end_exclusive": end,
                        "length": end - start,
                        "segment": segment,
                    }
                    for segment, (start, end) in enumerate(intervals)
                    for crane in sorted(chain)
                ]
                signature = (
                    tuple(tuple(row) for row in history),
                    tuple(sorted(work_plan.items())),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                proposals.append({
                    "operator": "paired_window_cyclic_exchange",
                    "history": [tuple(row) for row in history],
                    "work_plan": work_plan,
                    "regions": regions,
                    "source_signature": _trajectory_signature(source),
                    "source_hash": source_hash,
                    "details": {
                        "crane": q + 1,
                        "bay": target_bay,
                        "primary_block": dict(primary),
                        "residual_block": dict(residual),
                        "gap": int(revisit["gap"]),
                        "chain": [crane + 1 for crane in chain],
                        "early_window": list(intervals[0]),
                        "late_window": [list(item) for item in intervals[1:]],
                        "relay_from_bays": old_bays,
                        "relay_to_bays": new_bays,
                        "compensation_bay": final_bay,
                        "compensation_crane": final_crane + 1,
                        "compensation_time": compensation_time,
                        "compensation_kind": compensation_kind,
                        "tail_takeover": use_tail_takeover,
                        "tail_hold_position": hold_position,
                        "tail_receiver": (
                            tail_receiver + 1
                            if use_tail_takeover and tail_receiver is not None
                            else None
                        ),
                        "ledger_delta": {
                            "early": {
                                str(bay): change
                                for bay, change in sorted(early_delta.items())
                            },
                            "late": {
                                str(bay): change
                                for bay, change in sorted(late_delta.items())
                            },
                            "closed": closed_delta,
                        },
                    },
                })
                stats["unique"] += 1
                stats["operators"]["block_boundary_shift"] += int(
                    use_tail_takeover
                )
    return proposals, stats


def _continuity_block_neighbors(
    W: Sequence[int],
    M: int,
    candidate: _CandidateSchedule,
    max_candidates: int = 128,
    return_details: bool = False,
) -> list[tuple[str, list[tuple[int, ...]]] | tuple[
    str, list[tuple[int, ...]], list[dict[str, Any]]
]]:
    """Generate fixed-H work-block proposals with fair operator quotas.

    Every proposal edits one bounded block or an explicitly paired set of
    bounded blocks.  The decoder is responsible for assigning work/idle at
    the resulting positions; this function never edits or deletes a slot.
    """
    if candidate.move_time != 0 or candidate.makespan < 2:
        return []
    rows = [tuple(row) for row in _candidate_position_rows(candidate, M)]
    horizon = candidate.makespan
    work_at = {
        (slot.time, slot.crane - 1): int(slot.work_bay)
        for slot in candidate.slots
        if slot.state == "work" and slot.work_bay is not None
        and 0 <= slot.time < horizon
    }
    proposals: list[tuple[str, list[tuple[int, ...]], list[dict[str, Any]]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = {tuple(rows)}
    quotas = {
        "revisit_work_exchange": 48,
        "early_residual_completion": 32,
        "idle_completion_hold": 24,
        "adjacent_relay_batch": 32,
    }
    generated_by_operator = {name: 0 for name in quotas}

    def add(
        name: str,
        changed: list[list[int]],
        declared_regions: list[dict[str, Any]] | None = None,
    ) -> None:
        if name not in quotas or generated_by_operator[name] >= quotas[name]:
            return
        history = tuple(tuple(row) for row in changed)
        if history in seen:
            return
        if len(proposals) >= max_candidates:
            return
        seen.add(history)
        generated_by_operator[name] += 1
        proposals.append((
            name,
            [tuple(row) for row in changed],
            declared_regions or _history_diff_regions(rows, history),
        ))

    def candidate_lengths(values: Sequence[int]) -> list[int]:
        usable = sorted({int(value) for value in values if int(value) > 0})
        return [value for value in (1, 2, 3, 4, 7, 8) if value in usable]

    def position_blocks(q: int) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        start = 0
        while start < horizon:
            position = rows[start][q]
            end = start + 1
            while end < horizon and rows[end][q] == position:
                end += 1
            blocks.append({
                "start": start,
                "end": end,
                "position": position,
                "length": end - start,
                "work_count": sum((t, q) in work_at for t in range(start, end)),
            })
            start = end
        return blocks

    def safe_replacements(
        q: int,
        times: Sequence[int],
        excluded: int,
    ) -> list[int]:
        """Find one common on-rail replacement safe in every edited row."""
        if not times:
            return []
        preferred = [
            rows[times[0] - 1][q] if times[0] > 0 else excluded,
            1 + 2 * q,
            len(W) - 2 * (M - q - 1),
        ]
        candidates = list(dict.fromkeys([
            *preferred,
            *range(1, len(W) + 1),
        ]))
        result: list[int] = []
        for target in candidates:
            if target == excluded or not 1 <= target <= len(W):
                continue
            if all(
                all(
                    other_q == q or abs(target - rows[t][other_q]) >= 2
                    for other_q in range(M)
                )
                for t in times
            ):
                result.append(target)
        return result

    # A repeated work position can often exchange complete units between an
    # early block and a later block.  The two changed regions stay bounded:
    # the early block is returned to its predecessor and the later block is
    # extended backwards into the intervening block.
    for q in range(M):
        blocks = position_blocks(q)
        for left_index, left in enumerate(blocks):
            if not left["work_count"] or left["start"] <= 0:
                continue
            for right in blocks[left_index + 1:]:
                if right["position"] != left["position"] or not right["work_count"]:
                    continue
                gap = right["start"] - left["end"]
                predecessor = rows[left["start"] - 1][q]
                if gap <= 0 or predecessor == left["position"]:
                    continue
                lengths = candidate_lengths((left["length"], gap, right["length"]))
                for length in lengths:
                    changed = [list(row) for row in rows]
                    for t in range(left["start"], left["start"] + length):
                        changed[t][q] = predecessor
                    for t in range(right["start"] - length, right["start"]):
                        changed[t][q] = left["position"]
                    add(
                        "revisit_work_exchange", changed, [
                            {
                                "crane": q + 1,
                                "start": left["start"],
                                "end_exclusive": left["start"] + length,
                                "length": length,
                            },
                            {
                                "crane": q + 1,
                                "start": right["start"] - length,
                                "end_exclusive": right["start"],
                                "length": length,
                            },
                        ]
                    )

    # When the first block leaves a small amount of work for a distant return,
    # move the same number of capacity slots to the beginning and give the
    # late block back to the position it came from.  This is the generalized
    # form of the Q1/bay-2 pattern; no bay or crane identity is hard-coded.
    for q in range(M):
        blocks = position_blocks(q)
        for left_index, left in enumerate(blocks):
            if left["start"] != 0 or not left["work_count"]:
                continue
            for right in blocks[left_index + 1:]:
                if right["position"] != left["position"] or not right["work_count"]:
                    continue
                if left_index + 1 >= len(blocks):
                    continue
                next_block = blocks[left_index + 1]
                available_early = next_block["length"]
                available_late = right["length"]
                for length in candidate_lengths((available_early, available_late)):
                    late_times = list(range(right["end"] - length, right["end"]))
                    for replacement in safe_replacements(
                        q, late_times, left["position"]
                    ):
                        changed = [list(row) for row in rows]
                        for t in range(left["end"], left["end"] + length):
                            changed[t][q] = left["position"]
                        for t in late_times:
                            changed[t][q] = replacement
                        add(
                            "early_residual_completion", changed, [
                                {
                                    "crane": q + 1,
                                    "start": left["end"],
                                    "end_exclusive": left["end"] + length,
                                    "length": length,
                                },
                                {
                                    "crane": q + 1,
                                    "start": right["end"] - length,
                                    "end_exclusive": right["end"],
                                    "length": length,
                                },
                            ]
                        )

    # A completed crane may have a short, work-free terminal visit.  Holding
    # its preceding position is safe only as a candidate proposal; the full
    # verifier still checks every row and the work decoder checks capacity.
    for q in range(M):
        blocks = position_blocks(q)
        if len(blocks) < 2:
            continue
        final = blocks[-1]
        previous = blocks[-2]
        if final["work_count"] or final["length"] > 8:
            continue
        if final["position"] == previous["position"]:
            continue
        changed = [list(row) for row in rows]
        for t in range(final["start"], final["end"]):
            changed[t][q] = previous["position"]
        add("idle_completion_hold", changed, [{
            "crane": q + 1,
            "start": final["start"],
            "end_exclusive": final["end"],
            "length": final["length"],
        }])

    # Apply the same bounded boundary shift to adjacent cranes.  This is a
    # relay proposal, not an unconstrained global re-layout.
    for t in range(1, horizon):
        for q in range(M - 1):
            if rows[t - 1][q] == rows[t][q] or rows[t - 1][q + 1] == rows[t][q + 1]:
                continue
            changed = [list(row) for row in rows]
            changed[t][q] = rows[t - 1][q]
            changed[t][q + 1] = rows[t - 1][q + 1]
            add("adjacent_relay_batch", changed, [
                {"crane": q + 1, "start": t, "end_exclusive": t + 1, "length": 1},
                {"crane": q + 2, "start": t, "end_exclusive": t + 1, "length": 1},
            ])

    # Rotate operators rather than returning a fixed prefix.  This preserves
    # the quota guarantee even if one family generates many duplicates.
    by_name: dict[
        str, list[tuple[str, list[tuple[int, ...]], list[dict[str, Any]]]]
    ] = {}
    for item in proposals:
        by_name.setdefault(item[0], []).append(item)
    ordered: list[tuple[str, list[tuple[int, ...]], list[dict[str, Any]]]] = []
    names = list(quotas)
    index = 0
    while any(by_name.get(name) for name in names):
        name = names[index % len(names)]
        if by_name.get(name):
            ordered.append(by_name[name].pop(0))
        index += 1
    if return_details:
        return ordered[:max_candidates]
    return [(name, history) for name, history, _regions in ordered[:max_candidates]]


def _continuity_rank(
    candidate: _CandidateSchedule,
    M: int,
) -> tuple[int, ...]:
    """Rank a candidate only after the formal acceptance constraints pass."""
    report = _continuity_diagnostics(candidate, M)
    return (
        candidate.completion_time,
        candidate.completion_movement_count,
        *(int(value) for value in report["continuity_key"]),
    )


def _operational_rank(
    candidate: _CandidateSchedule,
    M: int,
) -> tuple[int, ...]:
    """Rank schedules by the user objective before operational diagnostics.

    This is deliberately separate from ``objective_key``.  It is used to
    publish a useful operational alternative while the formal result keeps
    the project's original lexicographic priorities.
    """
    continuity = _continuity_diagnostics(candidate, M)
    idle = _idle_diagnostics(candidate, M)
    return (
        int(candidate.completion_time),
        int(candidate.completion_movement_count),
        int(candidate.reversal_count),
        int(continuity["work_revisit_count"]),
        int(continuity["extra_work_blocks_total"]),
        int(continuity["bay_fragmentation"]),
        int(idle["max_internal_idle"]),
        int(idle["total_internal_idle"]),
        int(candidate.split_bay_count),
        int(candidate.load_deviation),
    )


def _execution_rank(
    candidate: _CandidateSchedule,
    M: int,
) -> tuple[int, ...]:
    """Rank schedules by actual completion and moves, then diagnostics.

    Continuity diagnostics only break ties after both user priorities.
    """
    continuity = _continuity_diagnostics(candidate, M)
    idle = _idle_diagnostics(candidate, M)
    return (
        int(candidate.completion_time),
        int(candidate.completion_movement_count),
        int(continuity["long_revisit_count"]),
        int(continuity["work_revisit_count"]),
        int(continuity["extra_work_blocks_total"]),
        int(continuity["bay_fragmentation"]),
        int(continuity["short_excursion_count"]),
        int(continuity["terminal_return_count"]),
        int(continuity["max_crane_movement_count"]),
        int(continuity["max_work_revisit_gap"]),
        int(candidate.reversal_count),
        int(idle["max_internal_idle"]),
        int(idle["total_internal_idle"]),
        int(candidate.load_deviation),
    )


def _trajectory_signature(candidate: _CandidateSchedule) -> tuple[Any, ...]:
    """Joint position/work signature used to deduplicate candidate pools."""
    return tuple(
        (
            slot.time,
            slot.crane,
            slot.state,
            slot.start_bay,
            slot.end_bay,
            slot.work_bay,
        )
        for slot in candidate.slots
    )


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
    accept_smooth_ties: bool = False,
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
                    and (
                        candidate.objective_key < incumbent.objective_key
                        or (
                            accept_smooth_ties
                            and candidate.objective_key == incumbent.objective_key
                            and _trajectory_smoothness(candidate, M)
                            < _trajectory_smoothness(incumbent, M)
                        )
                    )
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
    preserve_horizon: bool = False,
) -> _CandidateSchedule:
    """Decode a configuration path using the configured relocation duration.

    ``move_time == 0`` means a relocation occurs at the boundary between two
    work periods.  It therefore creates no ``move`` slot.  Positive durations
    are expanded into that many unit slots; positions during a simultaneous
    move are linearly interpolated, which preserves crane order and the
    two-bay separation whenever both endpoint configurations are safe.

    With ``preserve_horizon=True`` a zero-time history keeps every absolute
    time row, including rows where every crane is idle.  The older decoder
    intentionally removed such rows while searching for a shorter makespan;
    that behavior is not safe for fixed-H continuity proposals.
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
            if not any(work_here) and not preserve_horizon:
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


def _refine_same_horizon_trajectory(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    candidate: _CandidateSchedule,
    deadline: float,
    seed: int,
    *,
    move_time: int = 0,
    attempt_trace: list[dict[str, Any]] | None = None,
    continuity: bool = False,
    result_box: dict[str, Any] | None = None,
    enable_work_transfer: bool = False,
    enable_fragmentation_repair: bool = False,
    enable_cyclic_exchange: bool = True,
    enable_phase_resequence: bool = False,
    enable_phase_closure: bool = True,
    enable_cross_crane_phase_relay: bool = False,
    enable_idle_capacity_rebalance: bool = False,
    enable_forced_prefix_consolidation: bool = True,
    local_state_limit: int = 64,
    protect_source_continuity: bool = False,
    strict_local_transactions: bool = False,
    source_hash: str | None = None,
    enable_multi_relay: bool = True,
) -> tuple[_CandidateSchedule, int]:
    """Polish a complete shortened trajectory without changing its horizon.

    The first feasible H-1 trajectory is valuable evidence, but it is not
    necessarily operationally smooth.  Reuse the bounded local trajectory
    operator on the remaining budget, prioritizing windows containing real
    work revisits.  ``continuity=True`` additionally enables paired fixed-H
    block exchanges and keeps formal and continuity candidates separately.
    The formal objective is never silently replaced by the continuity rank.
    """
    if move_time != 0 or candidate.move_time != 0:
        return candidate, 0
    if not _candidate_passes_independent_verifier(W, M, starts, candidate):
        raise ValueError("trajectory 平滑阶段收到非法的首个可行方案。")
    best = candidate
    formal_best = candidate
    continuity_best = candidate
    operational_best = candidate
    execution_best = candidate
    balanced_best = candidate
    baseline = candidate
    baseline_continuity = _continuity_diagnostics(candidate, M, starts)
    evaluated_total = 0
    polish_started_at = time.perf_counter()
    cycle = 0
    stale_cycles = 0
    pool: dict[tuple[Any, ...], _CandidateSchedule] = {
        _trajectory_signature(candidate): candidate
    }
    paired_transactions: list[dict[str, Any]] = []
    phase_transactions: list[dict[str, Any]] = []
    phase_closure_transactions: list[dict[str, Any]] = []
    cross_crane_phase_transactions: list[dict[str, Any]] = []
    idle_capacity_transactions: list[dict[str, Any]] = []
    forced_prefix_transactions: list[dict[str, Any]] = []
    operator_stats: dict[str, dict[str, int]] = {}
    continuity_stats = {
        "generated": 0,
        "deduplicated": 0,
        "capacity_rejected": 0,
        "safety_rejected": 0,
        "decoded": 0,
        "verified": 0,
        "accepted": 0,
        "unique_complete": 0,
        "formal_improvements": 0,
        "operational_improvements": 0,
        "execution_improvements": 0,
        "balanced_improvements": 0,
        "pure_sync_delay_rejected": 0,
        "time_seconds": 0.0,
        "operator": operator_stats,
        "work_transfer": {
            "rounds": 0,
            "generated": 0,
            "unique": 0,
            "capacity_rejected": 0,
            "safety_rejected": 0,
            "boundary_rejected": 0,
            "verified": 0,
            "accepted": 0,
            "operators": {},
        },
        "paired_window_cyclic": {
            "status": "NOT_RUN",
            "rounds": 0,
            "generated": 0,
            "unique": 0,
            "expanded_states": 0,
            "early_window_states": 0,
            "late_window_states": 0,
            "capacity_rejected": 0,
            "safety_rejected": 0,
            "boundary_rejected": 0,
            "ledger_rejected": 0,
            "verified": 0,
            "accepted": 0,
            "timeout": 0,
            "state_limit": 0,
            "operators": {},
        },
        "phase_block_resequence": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_revisits": 0,
            "multi_slot_revisits": 0,
            "active_bands_generated": 0,
            "phase_permutations_generated": 0,
            "phase_combinations_tested": 0,
            "states_expanded": 0,
            "states_pruned_horizon": 0,
            "states_pruned_safety": 0,
            "states_pruned_split": 0,
            "states_pruned_movement": 0,
            "complete_phase_plans": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_burden_migration": 0,
            "timeout": 0,
            "state_limit": 0,
            "operators": {},
        },
        "phase_closure_relay": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_revisits": 0,
            "activity_bands_tested": 0,
            "phase_orders_generated": 0,
            "event_schedules_generated": 0,
            "states_expanded": 0,
            "complete_phase_plans": 0,
            "ledger_closed": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_safety": 0,
            "rejected_horizon": 0,
            "rejected_ledger": 0,
            "timeout": 0,
            "state_limit": 0,
            "max_activity_width": 0,
            "activity_chain_expansions": [],
        },
        "cross_crane_phase_relay": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_revisits": 0,
            "activity_bands_tested": 0,
            "assignment_variants_generated": 0,
            "retimed_plans_generated": 0,
            "retimed_plans_solved": 0,
            "owner_change_branches": 0,
            "two_hop_relay_branches": 0,
            "complete_phase_plans": 0,
            "ledger_closed": 0,
            "states_expanded": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_overlap": 0,
            "rejected_eligibility": 0,
            "rejected_safety": 0,
            "rejected_ledger": 0,
            "rejected_horizon": 0,
            "timeout": 0,
            "state_limit": 0,
            "max_activity_width": 0,
            "activity_chain_expansions": [],
        },
        "idle_capacity_rebalance": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_revisits": 0,
            "idle_capacity_focuses": 0,
            "idle_capacity_chain_focuses": 0,
            "partial_transfer_focuses": 0,
            "focuses_without_revisits": 0,
            "activity_bands_tested": 0,
            "assignment_variants_generated": 0,
            "retimed_plans_generated": 0,
            "retimed_plans_solved": 0,
            "owner_change_branches": 0,
            "two_hop_relay_branches": 0,
            "complete_phase_plans": 0,
            "ledger_closed": 0,
            "states_expanded": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_overlap": 0,
            "rejected_eligibility": 0,
            "rejected_safety": 0,
            "rejected_ledger": 0,
            "rejected_horizon": 0,
            "timeout": 0,
            "state_limit": 0,
            "max_activity_width": 0,
            "activity_chain_expansions": [],
            "status_counts": {},
        },
        "forced_prefix_consolidation": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_interruptions": 0,
            "focuses_started": 0,
            "focuses_completed": 0,
            "neighbor_bands_generated": 0,
            "focus_phase_orders": 0,
            "neighbor_phase_orders": 0,
            "idle_placements_tested": 0,
            "states_expanded": 0,
            "ledger_closed": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_safety": 0,
            "rejected_ledger": 0,
            "rejected_burden_migration": 0,
            "timeout": 0,
            "state_limit": 0,
        },
        "continuity_rejected": 0,
        "continuity_component_rejected": 0,
    }
    stop_reason = "deadline"

    def operator_counter(name: str) -> dict[str, int]:
        return operator_stats.setdefault(name, {
            "generated": 0,
            "deduplicated": 0,
            "capacity_rejected": 0,
            "safety_rejected": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "continuity_rejected": 0,
        })

    def work_ledger(candidate: _CandidateSchedule) -> list[int]:
        ledger = [0] * len(W)
        for slot in candidate.slots:
            if slot.state == "work" and slot.work_bay is not None:
                ledger[int(slot.work_bay) - 1] += 1
        return ledger

    def continuity_allowed(proposed: _CandidateSchedule) -> bool:
        return (
            proposed.makespan == baseline.makespan
            # Keep a small exploration allowance.  A relay may temporarily
            # use one extra split or two extra moves before a later block
            # exchange removes more movement and waiting.  These are search
            # bounds only; the published operational candidate still shows
            # the complete trade-off against formal_best.
            and proposed.split_bay_count <= baseline.split_bay_count + 1
            and proposed.movement_count <= baseline.movement_count + 2
        )

    def continuity_component_guard(
        proposed: _CandidateSchedule,
        operator: str,
        source_candidate: _CandidateSchedule | None = None,
        focus: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """Reject a local repair that exports a new fragmentation burden.

        The paired relay is allowed one replacement micro excursion while it
        removes at least one complete work block, but it may not create the
        two extra Q2@bay9 blocks seen in the old H=208 result.  This keeps the
        search exploratory without accepting a pure burden migration.
        """
        if not continuity or operator not in {
            "paired_window_cyclic_exchange",
            "phase_block_resequence",
            "phase_closure_relay",
            "cross_crane_phase_relay",
            "idle_capacity_rebalance",
            "forced_prefix_consolidation",
        }:
            return True, None
        # A cross-crane relay may create a receiver-side block because that is
        # its intended ownership-change mechanism.  Idle-capacity balancing is
        # stricter: load equality is never allowed to reintroduce the same
        # crane/same-bay revisits and terminal returns that Step 8 removed.
        if operator == "cross_crane_phase_relay":
            return True, None
        before = _continuity_diagnostics(
            source_candidate if source_candidate is not None else baseline,
            M,
            starts,
        )
        after = _continuity_diagnostics(proposed, M, starts)
        guard_source = source_candidate if source_candidate is not None else baseline
        if operator == "idle_capacity_rebalance":
            atomic_chain = bool((focus or {}).get("atomic_chain"))
            if after["work_revisit_count"] > before["work_revisit_count"]:
                return False, "new_work_revisit"
            if after["extra_work_blocks_total"] > before["extra_work_blocks_total"]:
                return False, "new_extra_work_block"
            if after["terminal_return_count"] > before["terminal_return_count"]:
                return False, "new_terminal_return"
            movement_allowance = 3 if atomic_chain else 1
            reversal_allowance = 3 if atomic_chain else 0
            if (
                proposed.movement_count
                > guard_source.movement_count + movement_allowance
            ):
                return False, "movement_regression"
            if (
                proposed.reversal_count
                > guard_source.reversal_count + reversal_allowance
            ):
                return False, "new_reversal"
            return True, None
        if proposed.objective_key < guard_source.objective_key:
            return True, None
        before_counts = before.get("work_block_count_by_crane_bay", {})
        after_counts = after.get("work_block_count_by_crane_bay", {})
        if operator in {
            "phase_block_resequence", "phase_closure_relay",
            "forced_prefix_consolidation",
        } and focus:
            focus_crane = str(int(focus.get("crane", 0)))
            focus_bay = str(int(focus.get("bay", 0)))
            before_focus_count = int(
                before_counts.get(focus_crane, {}).get(focus_bay, 0)
            )
            after_focus_count = int(
                after_counts.get(focus_crane, {}).get(focus_bay, 0)
            )
            if after_focus_count >= before_focus_count:
                return False, "focus_work_block_not_reduced"
        for crane, by_bay in after_counts.items():
            for bay, count in by_bay.items():
                old_count = int(before_counts.get(crane, {}).get(bay, 0))
                allowance = 1 if operator == "paired_window_cyclic_exchange" else 0
                if int(count) > old_count + allowance:
                    return False, f"new_work_block:{crane}:{bay}"
        if operator == "forced_prefix_consolidation":
            if after["forced_prefix_interruption_count"] >= before["forced_prefix_interruption_count"]:
                return False, "forced_prefix_not_consolidated"
            if after["extra_work_blocks_total"] > before["extra_work_blocks_total"]:
                return False, "new_extra_work_block"
            baseline_candidate = source_candidate if source_candidate is not None else baseline
            if proposed.split_bay_count > baseline_candidate.split_bay_count:
                return False, "new_split_bay"
            if proposed.movement_count > baseline_candidate.movement_count:
                return False, "movement_regression"
            if after["terminal_return_count"] > before["terminal_return_count"]:
                return False, "new_terminal_return"
            if after["short_excursion_count"] > before["short_excursion_count"]:
                return False, "new_short_excursion"
        if operator in {"phase_block_resequence", "phase_closure_relay"}:
            if after["extra_work_blocks_total"] > before["extra_work_blocks_total"]:
                return False, "new_extra_work_block"
            baseline_candidate = source_candidate if source_candidate is not None else baseline
            if proposed.split_bay_count > baseline_candidate.split_bay_count:
                return False, "new_split_bay"
            if proposed.movement_count > baseline_candidate.movement_count:
                return False, "movement_regression"
            # A phase transaction may remove the focus revisit only by
            # reordering the adjacent band.  It must not turn an existing
            # non-focus revisit into a longer gap or create a new terminal
            # return (for example, moving Q2@bay9 to the end of the horizon
            # while fixing Q3@bay13).
            before_gap_by_pair = {
                (int(item["crane"]), int(item["bay"])):
                int(item.get("gap", 0))
                for item in before.get("crane_work_revisits", [])
            }
            for item in after.get("crane_work_revisits", []):
                pair = (int(item["crane"]), int(item["bay"]))
                if int(item.get("gap", 0)) > before_gap_by_pair.get(pair, -1):
                    return False, f"burden_migration_gap:{pair[0]}:{pair[1]}"
            before_terminal_pairs = {
                (int(item["crane"]), int(item["bay"]))
                for item in before.get("terminal_returns", [])
            }
            after_terminal_pairs = {
                (int(item["crane"]), int(item["bay"]))
                for item in after.get("terminal_returns", [])
            }
            if after_terminal_pairs - before_terminal_pairs:
                crane, bay = sorted(after_terminal_pairs - before_terminal_pairs)[0]
                return False, f"burden_migration_terminal:{crane}:{bay}"
        if after["terminal_return_count"] > before["terminal_return_count"]:
            return False, "new_terminal_return"
        excursion_allowance = 0 if operator in {
            "phase_block_resequence", "phase_closure_relay",
        } else 1
        if after["short_excursion_count"] > before["short_excursion_count"] + excursion_allowance:
            return False, "new_short_excursion"
        if operator in {"phase_block_resequence", "phase_closure_relay"} and any(
            int(after["movement_count_by_crane"].get(str(crane), 0))
            > int(before["movement_count_by_crane"].get(str(crane), 0)) + 1
            for crane in range(1, M + 1)
        ):
            return False, "crane_movement_regression"
        if after["max_crane_movement_count"] > before["max_crane_movement_count"]:
            return False, "crane_movement_regression"
        return True, None

    def record_candidate(
        proposed: _CandidateSchedule | None,
        operator: str,
        before_key: tuple[int, int],
        before_smoothness: tuple[int, int, int],
        cycle_number: int,
        segment_index: int,
        before_history: Sequence[tuple[int, ...]],
        decode_failed: bool = False,
        declared_regions: list[dict[str, Any]] | None = None,
        transaction_metadata: dict[str, Any] | None = None,
        snapshot_source_candidate: _CandidateSchedule | None = None,
    ) -> tuple[bool, bool]:
        """Validate and consider one proposal; return (accepted, legal)."""
        nonlocal best, formal_best, continuity_best, operational_best, execution_best, balanced_best, evaluated_total
        source_candidate = snapshot_source_candidate or best
        counter = operator_counter(operator)
        counter["generated"] += 1
        continuity_stats["generated"] += 1
        candidate_found = proposed is not None
        continuity_guard_rejected = False
        continuity_component_rejected = False
        continuity_guard_reason = None
        candidate_legal = proposed is not None and (
            proposed.makespan == best.makespan
            and _candidate_passes_independent_verifier(W, M, starts, proposed)
        )
        if candidate_legal and proposed is not None:
            focus = None
            if transaction_metadata is not None:
                focus_value = transaction_metadata.get("focus")
                if isinstance(focus_value, dict):
                    focus = focus_value
            component_allowed, component_reason = continuity_component_guard(
                proposed, operator, source_candidate, focus
            )
            if not component_allowed:
                continuity_component_rejected = True
                continuity_guard_reason = component_reason
                counter["continuity_rejected"] += 1
                continuity_stats["continuity_rejected"] += 1
                continuity_stats["continuity_component_rejected"] += 1
                proposed = None
                candidate_legal = False
        core_improvement = (
            proposed is not None
            and proposed.objective_key < source_candidate.objective_key
        )
        if (
            candidate_legal and protect_source_continuity
            and proposed is not None and not core_improvement
        ):
            proposed_continuity = _continuity_diagnostics(proposed, M)
            proposed_burden = (
                proposed_continuity["max_work_revisit_gap"],
                proposed_continuity["work_revisit_count"],
                proposed_continuity["bay_fragmentation"],
            )
            baseline_burden = (
                baseline_continuity["max_work_revisit_gap"],
                baseline_continuity["work_revisit_count"],
                baseline_continuity["bay_fragmentation"],
            )
            if proposed_burden > baseline_burden:
                continuity_guard_rejected = True
                counter["continuity_rejected"] += 1
                continuity_stats["continuity_rejected"] += 1
                proposed = None
                candidate_legal = False
        if decode_failed:
            counter["capacity_rejected"] += 1
            continuity_stats["capacity_rejected"] += 1
        if proposed is not None:
            counter["decoded"] += 1
            continuity_stats["decoded"] += 1
        if not candidate_legal:
            if proposed is not None:
                counter["safety_rejected"] += 1
                continuity_stats["safety_rejected"] += 1
            proposed = None
        else:
            counter["verified"] += 1
            continuity_stats["verified"] += 1
        accepted = False
        if proposed is not None:
            proposed_balance = _balanced_schedule_metrics(proposed, M)
            source_balance = _balanced_schedule_metrics(source_candidate, M)
            balance_improvement = proposed_balance["key"] < _balanced_schedule_metrics(
                balanced_best, M
            )["key"]
            pure_sync_delay = bool(
                proposed_balance["completion_time"]
                == source_balance["completion_time"]
                and proposed_balance["loads"] == source_balance["loads"]
                and proposed_balance["finish_gap"] < source_balance["finish_gap"]
                and proposed_balance["leading_plus_internal_idle"]
                > source_balance["leading_plus_internal_idle"]
            )
            if pure_sync_delay:
                continuity_stats["pure_sync_delay_rejected"] += 1
            elif balance_improvement:
                balanced_best = proposed
                continuity_stats["balanced_improvements"] += 1
            signature = _trajectory_signature(proposed)
            if signature in pool:
                counter["deduplicated"] += 1
                continuity_stats["deduplicated"] += 1
            else:
                if len(pool) < 16:
                    pool[signature] = proposed
                else:
                    # Do not let the first 16 legal trajectories starve
                    # later schedules.  Replace the worst complete member
                    # when the new joint position/work signature is better.
                    worst_signature, worst_candidate = max(
                        pool.items(),
                        key=lambda item: (
                            _execution_rank(item[1], M),
                            item[1].objective_key,
                        ),
                    )
                    if (
                        _execution_rank(proposed, M), proposed.objective_key
                    ) < (
                        _execution_rank(worst_candidate, M),
                        worst_candidate.objective_key,
                    ):
                        del pool[worst_signature]
                        pool[signature] = proposed
                continuity_stats["unique_complete"] += 1
                formal_improvement = proposed.objective_key < formal_best.objective_key
                formal_tie_smoother = (
                    proposed.objective_key == before_key
                    and _trajectory_smoothness(proposed, M) < before_smoothness
                )
                continuity_improvement = (
                    continuity
                    and continuity_allowed(proposed)
                    and _continuity_rank(proposed, M) < _continuity_rank(continuity_best, M)
                )
                operational_improvement = (
                    continuity
                    and _operational_rank(proposed, M)
                    < _operational_rank(operational_best, M)
                )
                execution_improvement = (
                    continuity
                    and _execution_rank(proposed, M)
                    < _execution_rank(execution_best, M)
                )
                accepted = (
                    formal_improvement
                    or formal_tie_smoother
                    or continuity_improvement
                    or operational_improvement
                    or execution_improvement
                    or (
                        operator == "idle_capacity_rebalance"
                        and balance_improvement
                    )
                )
                if formal_improvement:
                    formal_best = proposed
                    continuity_stats["formal_improvements"] += 1
                    if balanced_best.completion_time > formal_best.completion_time:
                        balanced_best = proposed
                if continuity and continuity_allowed(proposed) and (
                    _continuity_rank(proposed, M) < _continuity_rank(continuity_best, M)
                ):
                    continuity_best = proposed
                if continuity and operational_improvement:
                    operational_best = proposed
                    continuity_stats["operational_improvements"] += 1
                if execution_improvement:
                    execution_best = proposed
                    continuity_stats["execution_improvements"] += 1
                if accepted:
                    best = proposed
                    counter["accepted"] += 1
                    continuity_stats["accepted"] += 1
                    if operator == "paired_window_cyclic_exchange":
                        paired_transactions.append({
                            "before": source_candidate,
                            "after": proposed,
                            "details": dict(transaction_metadata or {}),
                            "regions": list(declared_regions or []),
                        })
                    if operator == "phase_block_resequence":
                        before_diagnostics = _continuity_diagnostics(
                            source_candidate, M
                        )
                        after_diagnostics = _continuity_diagnostics(
                            proposed, M
                        )
                        before_ledger = work_ledger(source_candidate)
                        after_ledger = work_ledger(proposed)
                        phase_transactions.append({
                            "before": source_candidate,
                            "after": proposed,
                            "details": {
                                **dict(transaction_metadata or {}),
                                "verifier": "independent",
                                "verifier_passed": True,
                                "ledger_delta": {
                                    str(bay + 1): after_ledger[bay] - before_ledger[bay]
                                    for bay in range(len(W))
                                },
                                "ledger_delta_all_zero": (
                                    before_ledger == after_ledger
                                ),
                                "before_blocks": before_diagnostics.get(
                                    "work_blocks_by_crane_bay", {}
                                ),
                                "after_blocks": after_diagnostics.get(
                                    "work_blocks_by_crane_bay", {}
                                ),
                                "before_movement_arcs": before_diagnostics.get(
                                    "movement_arcs_by_crane", {}
                                ),
                                "after_movement_arcs": after_diagnostics.get(
                                    "movement_arcs_by_crane", {}
                                ),
                            },
                            "regions": list(declared_regions or []),
                        })
                    if operator == "phase_closure_relay":
                        before_diagnostics = _continuity_diagnostics(
                            source_candidate, M
                        )
                        after_diagnostics = _continuity_diagnostics(
                            proposed, M
                        )
                        before_ledger = work_ledger(source_candidate)
                        after_ledger = work_ledger(proposed)
                        phase_closure_transactions.append({
                            "before": source_candidate,
                            "after": proposed,
                            "details": {
                                **dict(transaction_metadata or {}),
                                "verifier": "independent",
                                "verifier_passed": True,
                                "ledger_delta": {
                                    str(bay + 1): after_ledger[bay] - before_ledger[bay]
                                    for bay in range(len(W))
                                },
                                "ledger_delta_all_zero": (
                                    before_ledger == after_ledger
                                ),
                                "before_blocks": before_diagnostics.get(
                                    "work_blocks_by_crane_bay", {}
                                ),
                                "after_blocks": after_diagnostics.get(
                                    "work_blocks_by_crane_bay", {}
                                ),
                                "before_movement_arcs": before_diagnostics.get(
                                    "movement_arcs_by_crane", {}
                                ),
                                "after_movement_arcs": after_diagnostics.get(
                                    "movement_arcs_by_crane", {}
                                ),
                            },
                            "regions": list(declared_regions or []),
                        })
                    if operator == "cross_crane_phase_relay":
                        before_diagnostics = _continuity_diagnostics(
                            source_candidate, M
                        )
                        after_diagnostics = _continuity_diagnostics(
                            proposed, M
                        )
                        before_ledger = work_ledger(source_candidate)
                        after_ledger = work_ledger(proposed)
                        cross_crane_phase_transactions.append({
                            "before": source_candidate,
                            "after": proposed,
                            "details": {
                                **dict(transaction_metadata or {}),
                                "verifier": "independent",
                                "verifier_passed": True,
                                "ledger_delta": {
                                    str(bay + 1): after_ledger[bay] - before_ledger[bay]
                                    for bay in range(len(W))
                                },
                                "ledger_delta_all_zero": (
                                    before_ledger == after_ledger
                                ),
                                "before_blocks": before_diagnostics.get(
                                    "work_blocks_by_crane_bay", {}
                                ),
                                "after_blocks": after_diagnostics.get(
                                    "work_blocks_by_crane_bay", {}
                                ),
                                "before_movement_arcs": before_diagnostics.get(
                                    "movement_arcs_by_crane", {}
                                ),
                                "after_movement_arcs": after_diagnostics.get(
                                    "movement_arcs_by_crane", {}
                                ),
                            },
                            "regions": list(declared_regions or []),
                        })
        if attempt_trace is not None:
            attempt_trace.append({
                "cycle": cycle_number,
                "segment_index": segment_index,
                "operator": operator,
                "chain": None,
                "window": None,
                "phase": "polish",
                "evaluated": 1,
                "candidate_found": candidate_found,
                "candidate_legal": candidate_legal,
                "accepted": accepted,
                "continuity_guard_rejected": continuity_guard_rejected,
                "continuity_component_rejected": continuity_component_rejected,
                "continuity_guard_reason": continuity_guard_reason,
                "transaction": transaction_metadata,
                "before_objective": list(before_key),
                "before_smoothness": list(before_smoothness),
                "after_objective": (
                    list(proposed.objective_key) if accepted else None
                ),
                "after_continuity": (
                    list(_continuity_rank(proposed, M)) if accepted else None
                ),
                "after_operational": (
                    list(_operational_rank(proposed, M))
                    if accepted and proposed is not None else None
                ),
                "changed_regions": (
                    declared_regions if declared_regions is not None else
                    _history_diff_regions(
                        before_history,
                        _candidate_position_rows(proposed, M),
                    ) if accepted and proposed is not None else []
                ),
            })
        return accepted, candidate_legal

    while time.perf_counter() < deadline and stale_cycles < 2:
        # Deterministic block operators get first refusal.  They are cheap,
        # explainable proposals and are the primary way to remove short
        # hand-offs without relying on isolated random cell edits.
        progress = False
        if continuity and enable_forced_prefix_consolidation:
            proposal_source = best
            prefix_transactions, prefix_output = _forced_prefix_consolidation_exchange(
                W, M, starts, proposal_source,
                max_neighbor_phase_permutations=64,
                max_focus_phase_permutations=16,
                max_idle_placements=16,
                state_limit=4_096,
                max_candidates=32,
                source_hash=source_hash,
                deadline=deadline,
            )
            prefix_stats = continuity_stats["forced_prefix_consolidation"]
            prefix_output_status = prefix_output.get("status", "UNKNOWN")
            # A successful earlier round is commonly followed by a no-op
            # round because the forced prefix has already been consolidated.
            # Keep that earlier search outcome instead of replacing it with
            # NOT_APPLICABLE in the aggregate record.
            if (
                prefix_output_status != "NOT_APPLICABLE"
                or prefix_stats.get("status") in {"NOT_RUN", "UNKNOWN"}
            ):
                prefix_stats["status"] = prefix_output_status
            prefix_stats["rounds"] += 1
            for key in (
                "focus_interruptions", "focuses_started", "focuses_completed",
                "neighbor_bands_generated", "focus_phase_orders",
                "neighbor_phase_orders", "idle_placements_tested",
                "states_expanded", "ledger_closed", "rejected_safety",
                "rejected_ledger", "timeout", "state_limit",
            ):
                prefix_stats[key] += int(prefix_output.get(key, 0))
            for transaction_index, transaction in enumerate(prefix_transactions):
                if time.perf_counter() >= deadline:
                    break
                if transaction.get("source_signature") != _trajectory_signature(proposal_source):
                    prefix_stats["rejected_burden_migration"] += 1
                    continue
                before_key = proposal_source.objective_key
                before_smoothness = _trajectory_smoothness(proposal_source, M)
                before_history = [
                    tuple(row)
                    for row in _candidate_position_rows(proposal_source, M)
                ]
                metadata = dict(transaction.get("details") or {})
                metadata.update({
                    "source_hash": source_hash,
                    "transaction_index": transaction_index,
                    "state_limit": 4_096,
                })
                try:
                    proposed = _candidate_from_explicit_phase_transaction(
                        W, M, starts, transaction["history"], proposal_source,
                        transaction["work_plan"], transaction["active_cranes"],
                        source_hash=source_hash,
                        transaction_source_hash=transaction.get("source_hash"),
                    )
                    prefix_stats["decoded"] += 1
                    prefix_stats["verified"] += 1
                except RuntimeError:
                    proposed = None
                    prefix_stats["rejected_safety"] += 1
                accepted, legal = record_candidate(
                    proposed, "forced_prefix_consolidation",
                    before_key, before_smoothness, cycle, transaction_index,
                    before_history, decode_failed=proposed is None,
                    declared_regions=transaction["regions"],
                    transaction_metadata=metadata,
                    snapshot_source_candidate=proposal_source,
                )
                prefix_stats["accepted"] += int(accepted)
                if not legal and proposed is not None:
                    prefix_stats["rejected_burden_migration"] += 1
                if accepted and proposed is not None:
                    before_diagnostics = _continuity_diagnostics(
                        proposal_source, M, starts
                    )
                    after_diagnostics = _continuity_diagnostics(
                        proposed, M, starts
                    )
                    before_ledger = work_ledger(proposal_source)
                    after_ledger = work_ledger(proposed)
                    forced_prefix_transactions.append({
                        "before": proposal_source,
                        "after": proposed,
                        "details": {
                            **metadata,
                            "verifier": "independent",
                            "verifier_passed": True,
                            "ledger_delta": {
                                str(bay + 1): after_ledger[bay] - before_ledger[bay]
                                for bay in range(len(W))
                            },
                            "ledger_delta_all_zero": before_ledger == after_ledger,
                            "before_blocks": before_diagnostics["work_blocks_by_crane_bay"],
                            "after_blocks": after_diagnostics["work_blocks_by_crane_bay"],
                            "before_movement_arcs": before_diagnostics["movement_arcs_by_crane"],
                            "after_movement_arcs": after_diagnostics["movement_arcs_by_crane"],
                        },
                        "regions": list(transaction["regions"]),
                    })
                if accepted:
                    progress = True
            if progress:
                stale_cycles = 0
                cycle += 1
                continue
        if enable_work_transfer:
            transactions, transaction_stats = _work_transfer_transactions(
                W, M, best, max_candidates=128, source_hash=source_hash,
                enable_multi_relay=enable_multi_relay,
            )
            work_stats = continuity_stats["work_transfer"]
            work_stats["rounds"] += 1
            for key in (
                "generated", "unique", "capacity_rejected",
                "safety_rejected", "boundary_rejected", "verified",
                "accepted",
            ):
                work_stats[key] += int(transaction_stats.get(key, 0))
            for name, count in (transaction_stats.get("operators") or {}).items():
                work_stats["operators"][name] = (
                    work_stats["operators"].get(name, 0) + int(count)
                )
            for transaction_index, transaction in enumerate(transactions):
                if time.perf_counter() >= deadline:
                    break
                if transaction.get("source_signature") != _trajectory_signature(best):
                    continuity_stats["work_transfer"]["boundary_rejected"] += 1
                    continue
                before_key = best.objective_key
                before_smoothness = _trajectory_smoothness(best, M)
                before_history = [
                    tuple(row) for row in _candidate_position_rows(best, M)
                ]
                operator = str(transaction["operator"])
                metadata = dict(transaction.get("details") or {})
                metadata.update({
                    "source_signature": list(transaction["source_signature"]),
                    "source_hash": source_hash,
                    "transaction_index": transaction_index,
                })
                try:
                    proposed = _candidate_from_explicit_work_transaction(
                        W, M, transaction["history"], best,
                        transaction["work_plan"], transaction["regions"],
                        source_hash=source_hash,
                        transaction_source_hash=transaction.get("source_hash"),
                    )
                    continuity_stats["work_transfer"]["verified"] += 1
                except RuntimeError:
                    proposed = None
                    continuity_stats["work_transfer"]["safety_rejected"] += 1
                accepted, legal = record_candidate(
                    proposed, operator, before_key, before_smoothness,
                    cycle, transaction_index, before_history,
                    decode_failed=proposed is None,
                    declared_regions=transaction["regions"],
                    transaction_metadata=metadata,
                )
                if legal:
                    continuity_stats["work_transfer"]["accepted"] += int(accepted)
                if accepted:
                    progress = True
            if progress:
                stale_cycles = 0
                cycle += 1
                continue
        if continuity and enable_cross_crane_phase_relay:
            # The unified operator must see the current source after every
            # accepted transaction.  It is deliberately separate from the
            # short-window work-transfer generator: one proposal contains the
            # complete active-band ledger and the phase/safety path together.
            proposal_source = best
            relay_state_limit = max(50_000, int(local_state_limit) * 256)
            transactions, transaction_stats = _cross_crane_phase_relay_search(
                W, M, starts, proposal_source,
                state_limit=relay_state_limit,
                max_candidates=32,
                max_assignment_variants=256,
                deadline=deadline,
                source_hash=source_hash,
            )
            relay_stats = continuity_stats["cross_crane_phase_relay"]
            relay_status = transaction_stats.get("status", "UNKNOWN")
            if (
                relay_status != "NOT_APPLICABLE"
                or relay_stats.get("status") in {"NOT_RUN", "UNKNOWN"}
            ):
                relay_stats["status"] = relay_status
            relay_stats["rounds"] += 1
            for key in (
                "focus_revisits", "activity_bands_tested",
                "assignment_variants_generated", "retimed_plans_generated",
                "retimed_plans_solved", "owner_change_branches",
                "two_hop_relay_branches", "complete_phase_plans",
                "ledger_closed", "states_expanded", "decoded", "verified",
                "accepted", "rejected_overlap", "rejected_eligibility",
                "rejected_safety", "rejected_ledger", "rejected_horizon",
                "timeout", "state_limit",
            ):
                relay_stats[key] += int(transaction_stats.get(key, 0))
            relay_stats["max_activity_width"] = max(
                relay_stats["max_activity_width"],
                int(transaction_stats.get("max_activity_width", 0)),
            )
            relay_stats["activity_chain_expansions"].extend(
                transaction_stats.get("activity_chain_expansions", [])
            )
            for transaction_index, transaction in enumerate(transactions):
                if time.perf_counter() >= deadline:
                    break
                if transaction.get("source_signature") != _trajectory_signature(proposal_source):
                    relay_stats["rejected_ledger"] += 1
                    continue
                before_key = proposal_source.objective_key
                before_smoothness = _trajectory_smoothness(proposal_source, M)
                before_history = [
                    tuple(row)
                    for row in _candidate_position_rows(proposal_source, M)
                ]
                metadata = dict(transaction.get("details") or {})
                metadata.update({
                    "source_hash": source_hash,
                    "transaction_index": transaction_index,
                    "state_limit": relay_state_limit,
                })
                try:
                    proposed = _candidate_from_explicit_cross_crane_phase_transaction(
                        W, M, starts, transaction["history"], proposal_source,
                        transaction["work_plan"], transaction["active_cranes"],
                        source_hash=source_hash,
                        transaction_source_hash=transaction.get("source_hash"),
                    )
                    relay_stats["decoded"] += 1
                    relay_stats["verified"] += 1
                except RuntimeError:
                    proposed = None
                    relay_stats["rejected_safety"] += 1
                accepted, legal = record_candidate(
                    proposed, "cross_crane_phase_relay",
                    before_key, before_smoothness, cycle, transaction_index,
                    before_history, decode_failed=proposed is None,
                    declared_regions=transaction["regions"],
                    transaction_metadata=metadata,
                    snapshot_source_candidate=proposal_source,
                )
                relay_stats["accepted"] += int(accepted)
                if not legal and proposed is not None:
                    relay_stats["rejected_safety"] += 1
                if accepted:
                    progress = True
            if progress:
                stale_cycles = 0
                cycle += 1
                continue
        if continuity and enable_idle_capacity_rebalance:
            proposal_source = (
                balanced_best
                if balanced_best.makespan == best.makespan
                else best
            )
            # A first adjacent hand-off changes the useful donor/receiver pair
            # for the next round.  Keep enough bounded states and assignment
            # variants for that second local relay instead of letting many
            # infeasible variants from the first focus starve all later foci.
            balance_state_limit = max(150_000, int(local_state_limit) * 512)
            transactions, transaction_stats = _cross_crane_phase_relay_search(
                W, M, starts, proposal_source,
                state_limit=balance_state_limit,
                max_candidates=8,
                max_assignment_variants=1_024,
                deadline=deadline,
                source_hash=source_hash,
                include_idle_capacity=True,
                idle_capacity_only=True,
            )
            balance_stats = continuity_stats["idle_capacity_rebalance"]
            balance_status = transaction_stats.get("status", "UNKNOWN")
            if balance_status != "NOT_APPLICABLE" or balance_stats["status"] == "NOT_RUN":
                balance_stats["status"] = balance_status
            balance_stats["rounds"] += 1
            for key in (
                "focus_revisits", "idle_capacity_focuses",
                "idle_capacity_chain_focuses", "partial_transfer_focuses",
                "focuses_without_revisits", "activity_bands_tested",
                "assignment_variants_generated", "retimed_plans_generated",
                "retimed_plans_solved",
                "owner_change_branches", "two_hop_relay_branches",
                "complete_phase_plans", "ledger_closed", "states_expanded",
                "decoded", "verified", "accepted", "rejected_overlap",
                "rejected_eligibility", "rejected_safety", "rejected_ledger",
                "rejected_horizon", "timeout", "state_limit",
            ):
                balance_stats[key] += int(transaction_stats.get(key, 0))
            balance_stats["max_activity_width"] = max(
                balance_stats["max_activity_width"],
                int(transaction_stats.get("max_activity_width", 0)),
            )
            balance_stats["activity_chain_expansions"].extend(
                transaction_stats.get("activity_chain_expansions", [])
            )
            for transaction_index, transaction in enumerate(transactions):
                if time.perf_counter() >= deadline:
                    break
                if transaction.get("source_signature") != _trajectory_signature(proposal_source):
                    balance_stats["rejected_ledger"] += 1
                    continue
                before_key = proposal_source.objective_key
                before_smoothness = _trajectory_smoothness(proposal_source, M)
                before_history = [
                    tuple(row)
                    for row in _candidate_position_rows(proposal_source, M)
                ]
                metadata = dict(transaction.get("details") or {})
                metadata.update({
                    "source_hash": source_hash,
                    "transaction_index": transaction_index,
                    "state_limit": balance_state_limit,
                })
                try:
                    proposed = _candidate_from_explicit_cross_crane_phase_transaction(
                        W, M, starts, transaction["history"], proposal_source,
                        transaction["work_plan"], transaction["active_cranes"],
                        source_hash=source_hash,
                        transaction_source_hash=transaction.get("source_hash"),
                    )
                    balance_stats["decoded"] += 1
                    balance_stats["verified"] += 1
                except RuntimeError:
                    proposed = None
                    balance_stats["rejected_safety"] += 1
                accepted, legal = record_candidate(
                    proposed, "idle_capacity_rebalance",
                    before_key, before_smoothness, cycle, transaction_index,
                    before_history, decode_failed=proposed is None,
                    declared_regions=transaction["regions"],
                    transaction_metadata=metadata,
                    snapshot_source_candidate=proposal_source,
                )
                balance_stats["accepted"] += int(accepted)
                if legal and proposed is not None:
                    idle_capacity_transactions.append({
                        "before": proposal_source,
                        "after": proposed,
                        "details": metadata,
                        "regions": list(transaction.get("regions", [])),
                    })
                elif not legal and proposed is not None:
                    balance_stats["rejected_safety"] += 1
                if accepted:
                    progress = True
            if progress:
                stale_cycles = 0
                cycle += 1
                continue
        if continuity and enable_phase_closure:
            proposal_source = best
            closure_state_limit = max(50_000, int(local_state_limit) * 256)
            transactions, transaction_stats = _phase_closure_relay_search(
                W, M, starts, proposal_source,
                state_limit=closure_state_limit,
                max_candidates=32,
                max_phase_orders=128,
                max_variants_per_crane=12_000,
                deadline=deadline,
                source_hash=source_hash,
            )
            closure_stats = continuity_stats["phase_closure_relay"]
            closure_stats["status"] = transaction_stats.get("status", "UNKNOWN")
            closure_stats["rounds"] += 1
            for key in (
                "focus_revisits", "activity_bands_tested",
                "phase_orders_generated", "event_schedules_generated",
                "states_expanded", "complete_phase_plans", "ledger_closed",
                "decoded", "verified", "accepted", "rejected_safety",
                "rejected_horizon", "rejected_ledger", "timeout",
                "state_limit", "max_activity_width",
            ):
                closure_stats[key] += int(transaction_stats.get(key, 0))
            closure_stats["activity_chain_expansions"].extend(
                transaction_stats.get("activity_chain_expansions", [])
            )
            for transaction_index, transaction in enumerate(transactions):
                if time.perf_counter() >= deadline:
                    break
                if transaction.get("source_signature") != _trajectory_signature(proposal_source):
                    closure_stats["rejected_ledger"] += 1
                    continue
                before_key = proposal_source.objective_key
                before_smoothness = _trajectory_smoothness(proposal_source, M)
                before_history = [
                    tuple(row)
                    for row in _candidate_position_rows(proposal_source, M)
                ]
                metadata = dict(transaction.get("details") or {})
                metadata.update({
                    "source_hash": source_hash,
                    "transaction_index": transaction_index,
                    "state_limit": closure_state_limit,
                })
                try:
                    proposed = _candidate_from_explicit_phase_transaction(
                        W, M, starts, transaction["history"],
                        proposal_source, transaction["work_plan"],
                        transaction["active_cranes"],
                        source_hash=source_hash,
                        transaction_source_hash=transaction.get("source_hash"),
                    )
                    closure_stats["decoded"] += 1
                    closure_stats["verified"] += 1
                except RuntimeError:
                    proposed = None
                    closure_stats["rejected_safety"] += 1
                accepted, legal = record_candidate(
                    proposed, "phase_closure_relay",
                    before_key, before_smoothness, cycle, transaction_index,
                    before_history, decode_failed=proposed is None,
                    declared_regions=transaction["regions"],
                    transaction_metadata=metadata,
                    snapshot_source_candidate=proposal_source,
                )
                closure_stats["accepted"] += int(accepted)
                if not legal and proposed is not None:
                    closure_stats["rejected_safety"] += 1
                if accepted:
                    progress = True
            if progress:
                stale_cycles = 0
                cycle += 1
                continue
        if continuity and enable_phase_resequence:
            proposal_source = best
            # The phase beam ranks complete-but-not-yet-started work as
            # pending, so the H=208 Q3/Q2 conflict chain reaches a legal
            # full-plan permutation in roughly 50k expansions.  Keep a
            # deterministic floor above that threshold; tying the phase
            # budget to the short-window limit (64/256) would otherwise stop
            # before the first complete transaction and report a misleading
            # ``UNKNOWN_STATE_LIMIT``.
            phase_state_limit = max(150_000, int(local_state_limit) * 256)
            transactions, transaction_stats = _phase_block_resequence_exchange(
                W, M, starts, proposal_source,
                max_active_cranes=4,
                max_block_permutations=256,
                state_limit=phase_state_limit,
                max_candidates=32,
                source_hash=source_hash,
                deadline=deadline,
            )
            phase_stats = continuity_stats["phase_block_resequence"]
            phase_stats["status"] = transaction_stats.get("status", "UNKNOWN")
            phase_stats["rounds"] += 1
            for key in (
                "focus_revisits", "multi_slot_revisits",
                "active_bands_generated", "phase_permutations_generated",
                "phase_combinations_tested", "states_expanded",
                "states_pruned_horizon", "states_pruned_safety",
                "states_pruned_split", "states_pruned_movement",
                "complete_phase_plans", "decoded", "verified", "accepted",
                "rejected_burden_migration", "timeout", "state_limit",
            ):
                phase_stats[key] += int(transaction_stats.get(key, 0))
            for name, count in (transaction_stats.get("operators") or {}).items():
                phase_stats["operators"][name] = (
                    phase_stats["operators"].get(name, 0) + int(count)
                )
            for transaction_index, transaction in enumerate(transactions):
                if time.perf_counter() >= deadline:
                    break
                if transaction.get("source_signature") != _trajectory_signature(proposal_source):
                    phase_stats["rejected_burden_migration"] += 1
                    continue
                before_key = proposal_source.objective_key
                before_smoothness = _trajectory_smoothness(proposal_source, M)
                before_history = [
                    tuple(row)
                    for row in _candidate_position_rows(proposal_source, M)
                ]
                metadata = dict(transaction.get("details") or {})
                metadata.update({
                    "source_hash": source_hash,
                    "transaction_index": transaction_index,
                    "state_limit": phase_state_limit,
                })
                try:
                    proposed = _candidate_from_explicit_phase_transaction(
                        W, M, starts, transaction["history"],
                        proposal_source, transaction["work_plan"],
                        transaction["active_cranes"],
                        source_hash=source_hash,
                        transaction_source_hash=transaction.get("source_hash"),
                    )
                    phase_stats["decoded"] += 1
                    phase_stats["verified"] += 1
                except RuntimeError:
                    proposed = None
                    phase_stats["states_pruned_safety"] += 1
                accepted, legal = record_candidate(
                    proposed, "phase_block_resequence",
                    before_key, before_smoothness, cycle, transaction_index,
                    before_history, decode_failed=proposed is None,
                    declared_regions=transaction["regions"],
                    transaction_metadata=metadata,
                    snapshot_source_candidate=proposal_source,
                )
                phase_stats["accepted"] += int(accepted)
                if not legal and proposed is not None:
                    phase_stats["rejected_burden_migration"] += 1
                if accepted:
                    progress = True
            if progress:
                stale_cycles = 0
                cycle += 1
                continue
        if enable_fragmentation_repair and enable_cyclic_exchange:
            proposal_source = best
            transactions, transaction_stats = _paired_window_cyclic_work_exchange(
                W, M, proposal_source,
                max_candidates=max(1, local_state_limit),
                state_limit=max(1, local_state_limit),
                source_hash=source_hash, deadline=deadline,
            )
            pair_stats = continuity_stats["paired_window_cyclic"]
            pair_stats["status"] = transaction_stats.get("status", "UNKNOWN")
            pair_stats["rounds"] += 1
            for key in (
                "generated", "unique", "expanded_states",
                "early_window_states", "late_window_states",
                "capacity_rejected", "safety_rejected", "boundary_rejected",
                "ledger_rejected", "verified", "accepted", "timeout",
                "state_limit",
            ):
                pair_stats[key] += int(transaction_stats.get(key, 0))
            for name, count in (transaction_stats.get("operators") or {}).items():
                pair_stats["operators"][name] = (
                    pair_stats["operators"].get(name, 0) + int(count)
                )
            for transaction_index, transaction in enumerate(transactions):
                if time.perf_counter() >= deadline:
                    break
                if transaction.get("source_signature") != _trajectory_signature(proposal_source):
                    pair_stats["boundary_rejected"] += 1
                    continue
                before_key = proposal_source.objective_key
                before_smoothness = _trajectory_smoothness(proposal_source, M)
                before_history = [
                    tuple(row) for row in _candidate_position_rows(proposal_source, M)
                ]
                metadata = dict(transaction.get("details") or {})
                metadata.update({
                    "source_hash": source_hash,
                    "transaction_index": transaction_index,
                    "state_limit": local_state_limit,
                })
                try:
                    proposed = _candidate_from_explicit_work_transaction(
                        W, M, transaction["history"], proposal_source,
                        transaction["work_plan"], transaction["regions"],
                        source_hash=source_hash,
                        transaction_source_hash=transaction.get("source_hash"),
                        max_segments=4,
                        max_region_length=8,
                    )
                    pair_stats["verified"] += 1
                except RuntimeError:
                    proposed = None
                    pair_stats["safety_rejected"] += 1
                accepted, legal = record_candidate(
                    proposed, "paired_window_cyclic_exchange",
                    before_key, before_smoothness, cycle, transaction_index,
                    before_history, decode_failed=proposed is None,
                    declared_regions=transaction["regions"],
                    transaction_metadata=metadata,
                    snapshot_source_candidate=proposal_source,
                )
                if legal:
                    pair_stats["accepted"] += int(accepted)
                if accepted:
                    progress = True
            if progress:
                stale_cycles = 0
                cycle += 1
                continue
        if strict_local_transactions:
            # In strict mode the explicit transaction engine is the complete
            # fixed-H neighborhood.  Do not fall through to the historical
            # position-only decoder after a transaction is rejected.
            stale_cycles += 1
            cycle += 1
            continue
        segment_proposals = _trajectory_segment_neighbors(W, M, starts, best)
        declared_by_proposal: dict[tuple[str, tuple[tuple[int, ...], ...]], list[dict[str, Any]]] = {}
        if continuity:
            details = _continuity_block_neighbors(
                W, M, best, return_details=True
            )
            for item in details:
                if len(item) == 3:
                    operator, history, regions = item
                else:
                    operator, history = item
                    regions = None
                segment_proposals.append((operator, history))
                if regions is not None:
                    declared_by_proposal[(operator, tuple(history))] = regions
        for segment_index, (operator, history) in enumerate(segment_proposals):
            if time.perf_counter() >= deadline:
                break
            before_key = best.objective_key
            before_smoothness = _trajectory_smoothness(best, M)
            before_history = [
                tuple(row) for row in _candidate_position_rows(best, M)
            ]
            evaluated = 1
            evaluated_total += evaluated
            declared_regions = declared_by_proposal.get(
                (operator, tuple(history))
            )
            try:
                if declared_regions:
                    proposed = _candidate_from_history_with_frozen_work(
                        W, M, history, best, declared_regions
                    )
                elif strict_local_transactions:
                    proposed = None
                else:
                    proposed = _candidate_from_history(
                        W, M, history, move_time, preserve_horizon=True
                    )
            except RuntimeError:
                proposed = None
            accepted, _ = record_candidate(
                proposed, operator, before_key, before_smoothness,
                cycle, segment_index, before_history,
                decode_failed=proposed is None,
                declared_regions=declared_regions,
            )
            if accepted:
                progress = True
        if progress:
            stale_cycles = 0
            cycle += 1
            continue

        windows = _critical_repair_windows(best, M)
        if not windows:
            break
        excursions = _short_excursion_details(best, M)
        ranked: list[tuple[int, int, tuple[int, ...], tuple[int, int]]] = []
        for index, (chain, window) in enumerate(windows):
            relevance = sum(
                1
                for q, start, end, *_ in excursions
                if q in chain and start < window[1] and end >= window[0]
            )
            ranked.append((-relevance, index, chain, window))
        ranked.sort(key=lambda item: (item[0], item[3][0], item[1]))
        progress = False
        for rank, (_, index, chain, window) in enumerate(ranked):
            if time.perf_counter() >= deadline:
                break
            remaining_calls = max(1, len(ranked) - rank)
            slice_deadline = min(
                deadline,
                time.perf_counter() + max(
                    0.05,
                    (deadline - time.perf_counter()) / remaining_calls,
                ),
            )
            before_key = best.objective_key
            before_smoothness = _trajectory_smoothness(best, M)
            before_history = [
                tuple(row) for row in _candidate_position_rows(best, M)
            ]
            proposed, evaluated = _trajectory_repair(
                W, M, starts, best, slice_deadline,
                seed + cycle * 1009 + index,
                active_cranes=chain,
                window=window,
                move_time=move_time,
                preserve_horizon=True,
                accept_smooth_ties=True,
            )
            evaluated_total += evaluated
            candidate_found = proposed is not None
            candidate_legal = proposed is not None and (
                proposed.makespan == best.makespan
                and _candidate_passes_independent_verifier(
                    W, M, starts, proposed
                )
            )
            if not candidate_legal:
                proposed = None
            accepted, _ = record_candidate(
                proposed, f"random_window_{index}", before_key,
                before_smoothness, cycle, index, before_history,
            )
            if accepted:
                progress = True
                break
        if progress:
            stale_cycles = 0
        else:
            stale_cycles += 1
        cycle += 1
    if time.perf_counter() >= deadline:
        stop_reason = "deadline"
    elif stale_cycles >= 2:
        stop_reason = "stalled_after_operator_rounds"
    elif not _critical_repair_windows(best, M):
        stop_reason = "no_windows"
    result = best
    if continuity:
        # A continuity candidate with the same formal key is safe to return;
        # otherwise the official formal candidate remains the result.
        if (
            continuity_best.objective_key == formal_best.objective_key
            and _continuity_rank(continuity_best, M)
            <= _continuity_rank(formal_best, M)
        ):
            result = continuity_best
        else:
            result = formal_best
    continuity_stats["time_seconds"] = round(
        max(0.0, time.perf_counter() - polish_started_at),
        6,
    )
    if result_box is not None:
        result_box.update({
            "formal_best": formal_best,
            "continuity_best": continuity_best,
            "operational_best": operational_best,
            "execution_best": execution_best,
            "balanced_best": balanced_best,
            "balanced_best_metrics": _balanced_schedule_metrics(balanced_best, M),
            "paired_transactions": paired_transactions,
            "phase_transactions": phase_transactions,
            "phase_closure_transactions": phase_closure_transactions,
            "cross_crane_phase_transactions": cross_crane_phase_transactions,
            "idle_capacity_transactions": idle_capacity_transactions,
            "forced_prefix_transactions": forced_prefix_transactions,
            "pool_size": len(pool),
            "stats": continuity_stats,
            "stop_reason": stop_reason,
        })
    return result, evaluated_total


def _cumulative_local_trajectory_repair(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    incumbent: _CandidateSchedule,
    deadline: float,
    seed: int,
    move_time: int = 1,
    attempt_trace: list[dict[str, Any]] | None = None,
    continuity_output: dict[str, Any] | None = None,
    *,
    polish_after_first: bool = True,
    enable_phase_resequence: bool = True,
    enable_phase_closure: bool = True,
    enable_cross_crane_phase_relay: bool = False,
    enable_idle_capacity_rebalance: bool = False,
    enable_forced_prefix_consolidation: bool = True,
) -> tuple[
    _CandidateSchedule | None, int, _CandidateSchedule, _CandidateSchedule | None
]:
    """Prepare and shorten through cumulative bounded local windows.

    Every mutation remains inside one ordinary critical window.  Accepted
    same-horizon preparations become the source of the next window, allowing
    two distant local repairs to cooperate without introducing a global
    window.  Once the first H-1 schedule is found, the remaining deadline is
    used for fixed-horizon smoothing.  The returned values are the polished
    candidate (if any), evaluated iterations, prepared H schedule, and the
    first feasible H-1 candidate.
    """
    current = incumbent
    evaluated_total = 0
    first_feasible: _CandidateSchedule | None = None
    started = time.perf_counter()
    if move_time != 0:
        return None, evaluated_total, current, first_feasible

    def polish_and_return(
        shortened: _CandidateSchedule,
        prepared: _CandidateSchedule,
    ) -> tuple[_CandidateSchedule, _CandidateSchedule, _CandidateSchedule]:
        nonlocal evaluated_total, first_feasible
        if first_feasible is None:
            first_feasible = shortened
            if attempt_trace is not None:
                attempt_trace.append({
                    "phase": "first_feasible",
                    "time_to_first_feasible": round(
                        time.perf_counter() - started, 6
                    ),
                    "objective": list(shortened.objective_key),
                    "smoothness": list(_trajectory_smoothness(shortened, M)),
                })
        if not polish_after_first:
            return shortened, prepared, first_feasible
        polish_started = time.perf_counter()
        trace_start = len(attempt_trace) if attempt_trace is not None else 0
        polish_result: dict[str, Any] = {}
        polished, polish_evaluated = _refine_same_horizon_trajectory(
            W, M, starts, first_feasible, deadline,
            seed + 700_001, move_time=move_time,
            attempt_trace=attempt_trace,
            continuity=True,
            result_box=polish_result,
            enable_phase_resequence=enable_phase_resequence,
            enable_phase_closure=enable_phase_closure,
            enable_cross_crane_phase_relay=enable_cross_crane_phase_relay,
            enable_idle_capacity_rebalance=enable_idle_capacity_rebalance,
            enable_forced_prefix_consolidation=enable_forced_prefix_consolidation,
        )
        if continuity_output is not None:
            continuity_output.update(polish_result)
        evaluated_total += polish_evaluated
        polish_trace = (
            attempt_trace[trace_start:]
            if attempt_trace is not None else []
        )
        polish_attempts = sum(
            item.get("phase") == "polish" for item in polish_trace
        )
        polish_complete_candidates = sum(
            item.get("phase") == "polish" and item.get("candidate_legal", False)
            for item in polish_trace
        )
        polish_legal_improvements = sum(
            item.get("phase") == "polish" and item.get("accepted", False)
            for item in polish_trace
        )
        if attempt_trace is not None:
            attempt_trace.append({
                "phase": "polish_complete",
                "polish_evaluated": polish_evaluated,
                "polish_seconds": round(
                    time.perf_counter() - polish_started, 6
                ),
                "polish_attempts": polish_attempts,
                "polish_complete_candidates": polish_complete_candidates,
                "polish_legal_improvements": polish_legal_improvements,
                "objective": list(polished.objective_key),
                "smoothness": list(_trajectory_smoothness(polished, M)),
                "formal_best_objective": list(
                    polish_result["formal_best"].objective_key
                ) if polish_result.get("formal_best") is not None else None,
                "continuity_best_objective": list(
                    polish_result["continuity_best"].objective_key
                ) if polish_result.get("continuity_best") is not None else None,
                "continuity_best_rank": list(
                    _continuity_rank(polish_result["continuity_best"], M)
                ) if polish_result.get("continuity_best") is not None else None,
                "continuity_operator_stats": polish_result.get("stats"),
                "continuity_stop_reason": polish_result.get("stop_reason"),
            })
        return polished, prepared, first_feasible

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
                    polished, prepared, first = polish_and_return(targeted, current)
                    return polished, evaluated_total, prepared, first
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
                    polished, prepared, first = polish_and_return(shortened, current)
                    return polished, evaluated_total, prepared, first
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
                    polished, prepared, first = polish_and_return(shortened, current)
                    return polished, evaluated_total, prepared, first
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
                    polished, prepared, first = polish_and_return(shortened, current)
                    return polished, evaluated_total, prepared, first
        if progress:
            stale_cycles = 0
        else:
            stale_cycles += 1
        cycle += 1
    return None, evaluated_total, current, first_feasible


def _congestion_workload_lower_bound(W: Sequence[int], M: int) -> int:
    """Return the path-congestion lower bound for zero-time crane movement."""
    bound = max(max(W, default=0), math.ceil(sum(W) / max(1, M)))
    for left in range(len(W)):
        work = 0
        for right in range(left, len(W)):
            work += int(W[right])
            capacity = min(M, (right - left + 2) // 2)
            if capacity > 0:
                bound = max(bound, math.ceil(work / capacity))
    return bound


def _synchronize_by_compressed_row_groups(
    W: Sequence[int], M: int, starts: Sequence[int],
    source: _CandidateSchedule, deadline: float, *,
    beam_width: int = 12_000,
) -> tuple[_CandidateSchedule | None, dict[str, Any]]:
    """Permute complete row groups to remove trailing idle without fragmentation.

    Every distinct safe configuration plus explicit per-crane work row becomes
    one indivisible block.  The beam remembers closed ``(crane, bay)`` tasks;
    reopening one is the primary cost, starting another internal idle episode
    is secondary, and crane moves are tertiary.  Work ownership and amounts
    never change, so this is a safe synchronization post-process for an
    already strong low-move schedule.
    """
    stats: dict[str, Any] = {
        "status": "NOT_RUN", "groups": 0, "states": 0,
        "decoded": 0, "verified": 0, "reopens": None,
    }
    if source.move_time != 0 or source.makespan <= 1:
        stats["status"] = "NOT_APPLICABLE"
        return None, stats
    rows = _candidate_position_rows(source, M)
    work_by_cell: dict[tuple[int, int], int | None] = {
        (slot.time, slot.crane - 1): (
            int(slot.work_bay) if slot.state == "work" else None
        )
        for slot in source.slots
    }

    def record_at(time_index: int) -> tuple[Any, ...]:
        config = tuple(int(value) for value in rows[time_index])
        work = tuple(work_by_cell[(time_index, q)] for q in range(M))
        active = frozenset(
            (q, int(bay)) for q, bay in enumerate(work) if bay is not None
        )
        return config, work, active

    opening = record_at(0)
    grouped: dict[tuple[Any, ...], int] = {}
    for time_index in range(1, source.makespan):
        record = record_at(time_index)
        grouped[record] = grouped.get(record, 0) + 1
    groups = [
        {"record": record, "count": count}
        for record, count in grouped.items()
    ]
    stats["groups"] = len(groups)
    if not groups or len(groups) > 24 or beam_width <= 0:
        stats["status"] = "NOT_APPLICABLE"
        return None, stats

    full_mask = (1 << len(groups)) - 1
    # reopens, reversals, max_crane_moves, moves, idle_starts, mask, last,
    # closed, last_directions, move_counts, path
    states: list[tuple[Any, ...]] = [
        (0, 0, 0, 0, 0, 0, -1, frozenset(), (0,) * M, (0,) * M, tuple())
    ]
    for depth in range(len(groups)):
        if time.perf_counter() >= deadline:
            stats["status"] = "UNKNOWN_DEADLINE"
            return None, stats
        next_states: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        for (
            reopens, reversals, max_crane_moves, moves, idle_starts, mask,
            last, closed, last_directions, move_counts, path,
        ) in states:
            previous = opening if last < 0 else groups[last]["record"]
            for index, group in enumerate(groups):
                bit = 1 << index
                if mask & bit:
                    continue
                config, work, active = group["record"]
                if depth == len(groups) - 1 and any(
                    bay is None for bay in work
                ):
                    continue
                new_closed = closed | (previous[2] - active)
                directions = tuple(
                    0 if before == after else (1 if after > before else -1)
                    for before, after in zip(previous[0], config)
                )
                next_directions = tuple(
                    direction or last_direction
                    for direction, last_direction
                    in zip(directions, last_directions)
                )
                next_move_counts = tuple(
                    count + int(before != after)
                    for count, before, after
                    in zip(move_counts, previous[0], config)
                )
                item = (
                    reopens + len(active & closed),
                    reversals + sum(
                        direction != 0
                        and last_direction != 0
                        and direction != last_direction
                        for direction, last_direction
                        in zip(directions, last_directions)
                    ),
                    max(next_move_counts, default=0),
                    moves + sum(
                        before != after
                        for before, after in zip(previous[0], config)
                    ),
                    idle_starts + sum(
                        before is not None and after is None
                        for before, after in zip(previous[1], work)
                    ),
                    mask | bit,
                    index,
                    new_closed,
                    next_directions,
                    next_move_counts,
                    path + (index,),
                )
                key = (
                    item[5], index, new_closed, next_directions,
                    next_move_counts,
                )
                old = next_states.get(key)
                if old is None or item[:5] < old[:5]:
                    next_states[key] = item
        stats["states"] += len(next_states)
        states = sorted(
            next_states.values(),
            key=lambda item: (*item[:5], item[10]),
        )[:beam_width]
        if not states:
            stats["status"] = "NO_CANDIDATE"
            return None, stats
    complete = [state for state in states if state[5] == full_mask]
    if not complete:
        stats["status"] = "NO_CANDIDATE"
        return None, stats
    best_state = min(complete, key=lambda item: item[:5])
    ordered = [opening]
    for group_index in best_state[10]:
        ordered.extend(
            [groups[group_index]["record"]] * groups[group_index]["count"]
        )
    history = [record[0] for record in ordered]
    history.append(history[-1])
    work_plan = {
        (time_index, q): record[1][q]
        for time_index, record in enumerate(ordered)
        for q in range(M)
    }
    try:
        candidate = _candidate_from_rows_and_work_plan(
            W, M, history, work_plan
        )
    except (RuntimeError, ValueError):
        stats["status"] = "DECODE_FAILED"
        return None, stats
    stats["decoded"] = 1
    candidate = _normalize_completed_candidate(candidate)
    if not _candidate_passes_independent_verifier(W, M, starts, candidate):
        stats["status"] = "VERIFY_FAILED"
        return None, stats
    stats.update({
        "status": "FOUND", "verified": 1,
        "reopens": int(best_state[0]),
        "reversals": int(best_state[1]),
        "max_crane_moves": int(best_state[2]),
        "idle_starts": int(best_state[4]),
        "objective": list(candidate.objective_key),
    })
    return candidate, stats


def _contiguous_phase_schedule_search(
    W: Sequence[int], M: int, starts: Sequence[int],
    source: _CandidateSchedule, deadline: float, *, beam_width: int = 8_000,
    task_counts: Sequence[dict[int, int]] | None = None,
    profile_vectors: Sequence[Sequence[str]] | None = None,
) -> tuple[_CandidateSchedule | None, dict[str, Any]]:
    """Rebuild a schedule with every owned crane/bay task indivisible.

    This operator addresses a limitation of row-group permutation: correlated
    rows can force a crane to leave a bay and return later.  Here each owned
    ``(crane, bay)`` workload is a single non-preemptive phase.  The beam may
    wait or reposition only between phases and every decoded result is passed
    through the independent verifier.
    """
    stats: dict[str, Any] = {
        "status": "NOT_RUN", "profiles": 0, "states": 0,
        "decoded": 0, "verified": 0,
    }
    if source.move_time != 0 or source.completion_time <= 1:
        stats["status"] = "NOT_APPLICABLE"
        return None, stats
    horizon = source.completion_time
    counts: list[dict[int, int]] = [dict() for _ in range(M)]
    first_seen: list[list[int]] = [[] for _ in range(M)]
    for slot in sorted(source.slots, key=lambda item: (item.time, item.crane)):
        if slot.state != "work" or slot.work_bay is None:
            continue
        q, bay = slot.crane - 1, int(slot.work_bay)
        counts[q][bay] = counts[q].get(bay, 0) + 1
        if bay not in first_seen[q]:
            first_seen[q].append(bay)
    if task_counts is not None:
        counts = [
            {int(bay): int(amount) for bay, amount in crane.items() if amount > 0}
            for crane in task_counts
        ]
        for q in range(M):
            first_seen[q] = [
                bay for bay in first_seen[q] if bay in counts[q]
            ] + [
                bay for bay in sorted(counts[q]) if bay not in first_seen[q]
            ]
    if any(int(starts[q]) not in counts[q] for q in range(M)):
        stats["status"] = "NOT_APPLICABLE"
        return None, stats

    def profile(kinds: Sequence[str]) -> tuple[tuple[tuple[int, int], ...], ...]:
        result = []
        for q in range(M):
            kind = kinds[q]
            start = int(starts[q])
            rest = [bay for bay in counts[q] if bay != start]
            if kind == "source":
                order = [bay for bay in first_seen[q] if bay != start]
            elif kind == "ascending":
                order = sorted(rest)
            else:
                order = sorted(rest, reverse=True)
            result.append(tuple(
                (bay, int(counts[q][bay])) for bay in (start, *order)
            ))
        return tuple(result)

    legal = _legal_configurations(len(W), M)
    best: _CandidateSchedule | None = None
    # Descending paths move edge cranes inward without returning; source and
    # ascending profiles remain bounded fallbacks for asymmetric instances.
    profiles = list(profile_vectors or (
        ("descending",) * M,
        ("source",) * M,
        ("ascending",) * M,
    ))
    for profile_kinds in profiles:
        if time.perf_counter() >= deadline:
            stats["status"] = "UNKNOWN_DEADLINE"
            break
        phases = profile(profile_kinds)
        profile_name = "/".join(profile_kinds)
        stats["profiles"] += 1
        match_cache: dict[tuple[int | None, ...], list[tuple[int, ...]]] = {}
        near_cache: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}

        def near_configs(
            work: tuple[int | None, ...], positions: tuple[int, ...],
        ) -> list[tuple[int, ...]]:
            if work not in match_cache:
                match_cache[work] = [
                    config for config in legal
                    if all(
                        bay is None or config[q] == bay
                        for q, bay in enumerate(work)
                    )
                ]
            key = (work, positions)
            if key not in near_cache:
                near_cache[key] = sorted(
                    match_cache[work],
                    key=lambda config: (
                        sum(a != b for a, b in zip(positions, config)),
                        sum(abs(a - b) for a, b in zip(positions, config)),
                        config,
                    ),
                )[:4]
            return near_cache[key]

        # indices, remaining-in-phase, positions, directions, was-idle,
        # moves, reversals, idle-starts, row path, work path
        initial_remaining = tuple(phases[q][0][1] - 1 for q in range(M))
        opening = tuple(int(value) for value in starts)
        states: list[tuple[Any, ...]] = [(
            (0,) * M, initial_remaining, opening, (0,) * M,
            (False,) * M, 0, 0, 0, (opening,), (opening,),
        )]
        completed = True
        for time_index in range(1, horizon):
            if time.perf_counter() >= deadline:
                stats["status"] = "UNKNOWN_DEADLINE"
                completed = False
                break
            rows_left = horizon - time_index
            next_states: dict[tuple[Any, ...], tuple[Any, ...]] = {}
            for state in states:
                (indices, remaining, positions, directions, was_idle,
                 moves, reversals, idle_starts, row_path, work_path) = state
                choices: list[tuple[tuple[int, int, int | None], ...]] = []
                feasible = True
                for q in range(M):
                    if remaining[q] > 0:
                        choices.append(((
                            indices[q], remaining[q] - 1,
                            phases[q][indices[q]][0],
                        ),))
                        continue
                    next_index = indices[q] + 1
                    if next_index >= len(phases[q]):
                        choices.append(((indices[q], 0, None),))
                        continue
                    pending = sum(length for _, length in phases[q][next_index:])
                    if pending > rows_left:
                        feasible = False
                        break
                    start_phase = (
                        next_index, phases[q][next_index][1] - 1,
                        phases[q][next_index][0],
                    )
                    choices.append(
                        (start_phase,) if pending == rows_left
                        else (start_phase, (indices[q], 0, None))
                    )
                if not feasible:
                    continue
                for combination in itertools.product(*choices):
                    work = tuple(item[2] for item in combination)
                    for config in near_configs(work, positions):
                        step_directions = tuple(
                            0 if before == after else (1 if after > before else -1)
                            for before, after in zip(positions, config)
                        )
                        new_directions = tuple(
                            step or old for step, old
                            in zip(step_directions, directions)
                        )
                        new_moves = moves + sum(step != 0 for step in step_directions)
                        new_reversals = reversals + sum(
                            step != 0 and old != 0 and step != old
                            for step, old in zip(step_directions, directions)
                        )
                        now_idle = tuple(bay is None for bay in work)
                        new_idle_starts = idle_starts + sum(
                            not was_idle[q] and now_idle[q] for q in range(M)
                        )
                        new_indices = tuple(item[0] for item in combination)
                        new_remaining = tuple(item[1] for item in combination)
                        key = (
                            new_indices, new_remaining, config,
                            new_directions, now_idle,
                        )
                        item = (
                            new_indices, new_remaining, config, new_directions,
                            now_idle, new_moves, new_reversals,
                            new_idle_starts, row_path + (config,),
                            work_path + (work,),
                        )
                        old = next_states.get(key)
                        if old is None or (
                            new_reversals, new_moves, new_idle_starts
                        ) < (old[6], old[5], old[7]):
                            next_states[key] = item
            stats["states"] += len(next_states)
            if not next_states:
                completed = False
                break
            states = sorted(
                next_states.values(),
                # With identical route cost, retain states that have not
                # prematurely consumed their final phase.  The per-crane
                # pending-work guard above will force a start at the latest
                # feasible row, reducing long completed suffixes without
                # fragmenting work.
                key=lambda item: (item[6], item[5], item[7], -sum(item[1])),
            )[:beam_width]
        if not completed:
            continue
        finals = [
            state for state in states
            if all(
                state[0][q] == len(phases[q]) - 1 and state[1][q] == 0
                for q in range(M)
            )
        ]
        for state in sorted(finals, key=lambda item: (item[6], item[5], item[7]))[:8]:
            history = list(state[8])
            history.append(history[-1])
            work_plan = {
                (t, q): state[9][t][q]
                for t in range(horizon) for q in range(M)
            }
            try:
                candidate = _candidate_from_rows_and_work_plan(
                    W, M, history, work_plan
                )
            except (RuntimeError, ValueError):
                continue
            stats["decoded"] += 1
            if not _candidate_passes_independent_verifier(W, M, starts, candidate):
                continue
            stats["verified"] += 1
            if _continuity_diagnostics(candidate, M)["work_revisit_count"] != 0:
                continue
            if best is None or _recommended_schedule_rank(candidate, M) < _recommended_schedule_rank(best, M):
                best = candidate
        if best is not None:
            stats.update({
                "status": "FOUND", "profile": profile_name,
                "objective": list(best.objective_key),
            })
            return best, stats
    if stats["status"] == "NOT_RUN":
        stats["status"] = "NO_CANDIDATE"
    return best, stats


def _contiguous_phase_handoff_search(
    W: Sequence[int], M: int, starts: Sequence[int],
    source: _CandidateSchedule, deadline: float,
) -> tuple[_CandidateSchedule | None, dict[str, Any]]:
    """Transfer small boundary workloads, then enforce indivisible phases.

    A revisited task is offered to the adjacent crane on the inward/right
    side.  If that crane already owns the bay, its existing chunk determines
    the first transfer size; otherwise its unused horizon capacity does.  The
    resulting ownership ledger is searched as complete phases and independently
    verified, so the heuristic cannot leak or duplicate work.
    """
    counts: list[dict[int, int]] = [dict() for _ in range(M)]
    for slot in source.slots:
        if slot.state == "work" and slot.work_bay is not None:
            q, bay = slot.crane - 1, int(slot.work_bay)
            counts[q][bay] = counts[q].get(bay, 0) + 1
    loads = [sum(item.values()) for item in counts]
    transfers: list[dict[str, int]] = []
    revisits = _continuity_diagnostics(source, M)["crane_work_revisits"]
    used: set[tuple[int, int]] = set()
    for revisit in revisits:
        donor = int(revisit["crane"]) - 1
        receiver = donor + 1
        bay = int(revisit["bay"])
        if receiver >= M or (donor, bay) in used:
            continue
        available = counts[donor].get(bay, 0)
        if available <= 1:
            continue
        receiver_owned = counts[receiver].get(bay, 0)
        receiver_slack = max(0, source.completion_time - loads[receiver])
        amount = min(
            available - 1,
            receiver_owned if receiver_owned > 0 else receiver_slack,
        )
        if amount <= 0:
            continue
        counts[donor][bay] -= amount
        counts[receiver][bay] = receiver_owned + amount
        loads[donor] -= amount
        loads[receiver] += amount
        used.add((donor, bay))
        transfers.append({
            "donor": donor + 1, "receiver": receiver + 1,
            "bay": bay, "amount": amount,
        })
    stats: dict[str, Any] = {
        "status": "NOT_APPLICABLE", "transfers": transfers,
        "loads": loads,
    }
    if not transfers:
        return None, stats
    # Outer cranes benefit from a monotone inward path.  The central cranes
    # retain their observed phase order because they mediate both tight bay
    # pairs and often need one controlled direction change.
    kinds = tuple(
        "source" if 2 <= q < M - 1 else "descending"
        for q in range(M)
    )
    candidate, phase_stats = _contiguous_phase_schedule_search(
        W, M, starts, source, deadline, beam_width=12_000,
        task_counts=counts, profile_vectors=(kinds,),
    )
    if candidate is not None:
        candidate = _right_shift_terminal_work_blocks(
            W, M, starts, candidate
        )
    stats.update(phase_stats)
    stats["transfers"] = transfers
    stats["loads"] = loads
    return candidate, stats


def _right_shift_terminal_work_blocks(
    W: Sequence[int], M: int, starts: Sequence[int],
    source: _CandidateSchedule,
) -> _CandidateSchedule:
    """Move a final work block right inside its unchanged position run.

    Only work labels move; crane positions and therefore safety, movements and
    reversals remain unchanged.  This turns avoidable completed suffixes into
    inter-phase waiting without reopening any crane/bay task.
    """
    horizon = source.completion_time
    rows, work = _transaction_rows_work_map(source, M)
    current = source
    for q in range(M):
        work_times = [
            t for t in range(horizon) if work.get((t, q)) is not None
        ]
        if not work_times or work_times[-1] >= horizon - 1:
            continue
        end = work_times[-1] + 1
        bay = int(work[(work_times[-1], q)])
        start = work_times[-1]
        while start > 0 and work.get((start - 1, q)) == bay:
            start -= 1
        length = end - start
        previous_work = max((t for t in work_times if t < start), default=-1)
        trial_work = dict(work)
        for t in range(start, end):
            trial_work[(t, q)] = None
        latest_start = None
        for proposed in range(horizon - length, previous_work, -1):
            interval = range(proposed, proposed + length)
            if not all(rows[t][q] == bay for t in interval):
                continue
            if any(trial_work.get((t, q)) is not None for t in interval):
                continue
            if any(
                trial_work.get((t, other)) == bay
                for t in interval for other in range(M) if other != q
            ):
                continue
            latest_start = proposed
            break
        if latest_start is None or latest_start <= start:
            continue
        for t in range(latest_start, latest_start + length):
            trial_work[(t, q)] = bay
        try:
            proposed_candidate = _candidate_from_rows_and_work_plan(
                W, M, rows, trial_work
            )
        except (RuntimeError, ValueError):
            continue
        if not _candidate_passes_independent_verifier(
            W, M, starts, proposed_candidate
        ):
            continue
        if _continuity_diagnostics(
            proposed_candidate, M
        )["work_revisit_count"] > _continuity_diagnostics(
            current, M
        )["work_revisit_count"]:
            continue
        work = trial_work
        current = proposed_candidate
    return current


def _global_safe_pattern_search(
    W: Sequence[int], M: int, starts: Sequence[int],
    source: _CandidateSchedule, deadline: float, *,
    max_attempts_per_horizon: int = 1,
) -> tuple[_CandidateSchedule | None, dict[str, Any]]:
    """Cover work with safe row patterns, then recover crane identities.

    This search is not restricted to one monotone bay segment per crane.
    Sparse rows are extended to complete safe crane configurations while idle
    identities are balanced.  Rows are then ordered to reduce moves, with a
    full-work terminal row so all cranes finish together when capacity allows.
    """
    total = sum(int(value) for value in W)
    lower_bound = _congestion_workload_lower_bound(W, M)
    stats: dict[str, Any] = {
        "status": "NOT_RUN",
        "congestion_lower_bound": lower_bound,
        "target_horizons": [],
        "attempts": 0,
        "patterns_generated": 0,
        "rows_selected": 0,
        "configuration_extensions": 0,
        "decoded": 0,
        "verified": 0,
        "best_objective": None,
        "best_loads": None,
        "best_balance": None,
        "stop_reason": None,
    }
    if source.move_time != 0:
        stats.update({"status": "UNSUPPORTED", "stop_reason": "nonzero_move_time"})
        return None, stats
    if not W or M <= 0 or source.completion_time <= lower_bound:
        stats.update({"status": "NOT_APPLICABLE", "stop_reason": "at_lower_bound"})
        return None, stats

    N = len(W)
    positive = tuple(index for index, amount in enumerate(W) if amount > 0)
    legal_configurations = _legal_configurations(N, M)
    initial_row = tuple(_candidate_position_rows(source, M)[0])
    required = set(int(bay) for bay in starts)
    if not required.issubset(initial_row):
        stats.update({"status": "NO_CANDIDATE", "stop_reason": "required_start_missing"})
        return None, stats

    patterns_by_size: dict[int, list[tuple[int, ...]]] = {
        size: [] for size in range(1, M + 1)
    }
    for size in range(1, M + 1):
        for pattern in itertools.combinations(positive, size):
            if all(right - left > 1 for left, right in zip(pattern, pattern[1:])):
                patterns_by_size[size].append(pattern)
    stats["patterns_generated"] = sum(map(len, patterns_by_size.values()))

    intervals = [
        (left, right, min(M, (right - left + 2) // 2))
        for left in range(N)
        for right in range(left, N)
    ]

    def remaining_is_possible(remaining: Sequence[int], rows: int) -> bool:
        if rows < 0 or any(value < 0 or value > rows for value in remaining):
            return False
        if sum(remaining) > rows * M:
            return False
        prefix = [0]
        for value in remaining:
            prefix.append(prefix[-1] + int(value))
        return all(
            prefix[right + 1] - prefix[left] <= rows * capacity
            for left, right, capacity in intervals
        )

    def compact_work_patterns_for_horizon(
        horizon: int, variant: int,
    ) -> list[tuple[int, ...]] | None:
        """Search event blocks before falling back to row-wise colouring."""
        opening = tuple(
            bay - 1 for bay in initial_row if int(W[bay - 1]) > 0
        )
        remaining = [int(value) for value in W]
        for index in opening:
            remaining[index] -= 1
        rows_left = horizon - 1
        if not remaining_is_possible(remaining, rows_left):
            return None

        # State: transition burden, remaining work, rows, previous pattern,
        # and the compact (duration, pattern) event list.
        states: list[
            tuple[int, tuple[int, ...], int, tuple[int, ...], list[tuple[int, tuple[int, ...]]]]
        ] = [(0, tuple(remaining), rows_left, opening, [])]
        max_blocks = max(12, 2 * len(positive) + 8)
        for _depth in range(max_blocks):
            expanded: list[
                tuple[int, tuple[int, ...], int, tuple[int, ...], list[tuple[int, tuple[int, ...]]]]
            ] = []
            seen_states: set[tuple[Any, ...]] = set()
            for burden, values, rows, previous, blocks in states:
                if rows == 0:
                    if not any(values):
                        result = [opening]
                        for duration, pattern in blocks:
                            result.extend([pattern] * duration)
                        return result
                    continue
                prefix = [0]
                for value in values:
                    prefix.append(prefix[-1] + int(value))
                tight_intervals = [
                    (left, right, capacity)
                    for left, right, capacity in intervals
                    if prefix[right + 1] - prefix[left] == rows * capacity
                ]
                idle_capacity = rows * M - sum(values)
                choices: list[tuple[Any, ...]] = []
                # When at most one idle cell per remaining row is sufficient,
                # forbid needlessly sparse rows.  Concentrating two or more
                # idle cranes in one row makes both load balancing and later
                # movement consolidation strictly harder for this compact
                # construction.
                minimum_pattern_size = (
                    M - 1 if idle_capacity <= rows else 1
                )
                for size in range(max(1, minimum_pattern_size), M + 1):
                    for pattern in patterns_by_size.get(size, []):
                        if any(values[index] <= 0 for index in pattern):
                            continue
                        if any(
                            sum(left <= index <= right for index in pattern)
                            != capacity
                            for left, right, capacity in tight_intervals
                        ):
                            continue
                        selected = set(pattern)
                        duration = min(int(values[index]) for index in pattern)
                        if size < M:
                            duration = min(
                                duration,
                                idle_capacity // max(1, M - size),
                            )
                        for index, value in enumerate(values):
                            if index not in selected:
                                duration = min(duration, rows - int(value))
                        for left, right, capacity in intervals:
                            selected_count = sum(
                                left <= index <= right for index in pattern
                            )
                            if selected_count < capacity:
                                slack = (
                                    rows * capacity
                                    - (prefix[right + 1] - prefix[left])
                                )
                                duration = min(
                                    duration,
                                    slack // (capacity - selected_count),
                                )
                        if duration <= 0:
                            continue
                        proposed = list(values)
                        for index in pattern:
                            proposed[index] -= duration
                        if not remaining_is_possible(proposed, rows - duration):
                            continue
                        transition = len(set(previous) ^ set(pattern))
                        choices.append((
                            transition, -duration, pattern, duration,
                            tuple(proposed), rows - duration,
                        ))
                choices.sort(key=lambda item: (
                    item[0], item[1],
                    sum(
                        (position + 1 + variant) * (index + 3)
                        for position, index in enumerate(item[2])
                    ) % 1009,
                    item[2],
                ))
                for transition, _negative_duration, pattern, duration, proposed, new_rows in choices[:50]:
                    state_key = (proposed, new_rows, pattern)
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    expanded.append((
                        burden + transition,
                        proposed,
                        new_rows,
                        pattern,
                        blocks + [(duration, pattern)],
                    ))
            expanded.sort(key=lambda item: (
                item[0], len(item[4]),
                -sum(duration * duration for duration, _pattern in item[4]),
            ))
            states = expanded[:120]
            if not states or time.perf_counter() >= deadline:
                break
        return None

    def work_patterns_for_horizon(
        horizon: int, attempt: int,
    ) -> list[tuple[int, ...]] | None:
        compact = compact_work_patterns_for_horizon(horizon, attempt)
        if compact is not None:
            return compact
        opening = tuple(
            bay - 1 for bay in initial_row if int(W[bay - 1]) > 0
        )
        if not required.issubset({index + 1 for index in opening}):
            return None
        remaining = [int(value) for value in W]
        for index in opening:
            remaining[index] -= 1
        if not remaining_is_possible(remaining, horizon - 1):
            return None

        rows: list[tuple[int, ...]] = [opening]
        previous = opening
        opening_idle = M - len(opening)
        remaining_idle = horizon * M - total - opening_idle
        if remaining_idle < 0:
            return None
        idle_plan: list[int] = []
        for offset in range(horizon - 1):
            before = (offset * remaining_idle) // max(1, horizon - 1)
            after = ((offset + 1) * remaining_idle) // max(1, horizon - 1)
            idle_plan.append(after - before)
        if attempt:
            shift = attempt % max(1, horizon - 1)
            idle_plan = idle_plan[shift:] + idle_plan[:shift]

        for time_index in range(1, horizon):
            if time.perf_counter() >= deadline:
                return None
            rows_left = horizon - time_index
            desired_size = M - idle_plan[time_index - 1]
            minimum_size = max(1, sum(remaining) - M * (rows_left - 1))
            candidate_sizes = sorted(
                range(max(1, minimum_size), M + 1),
                key=lambda size: (abs(size - desired_size), -size),
            )
            choices: list[tuple[tuple[Any, ...], tuple[int, ...], list[int]]] = []
            for size in candidate_sizes:
                for pattern in patterns_by_size.get(size, []):
                    if any(remaining[index] <= 0 for index in pattern):
                        continue
                    proposed = list(remaining)
                    for index in pattern:
                        proposed[index] -= 1
                    if not remaining_is_possible(proposed, rows_left - 1):
                        continue
                    proportional_error = sum(
                        (proposed[index] * horizon
                         - (rows_left - 1) * int(W[index])) ** 2
                        for index in positive
                    )
                    switch_cost = len(set(pattern) ^ set(previous))
                    rotated = (
                        positive[attempt % len(positive):]
                        + positive[:attempt % len(positive)]
                        if positive else ()
                    )
                    tie = tuple(-proposed[index] for index in rotated)
                    rank = (
                        abs(size - desired_size), proportional_error,
                        switch_cost, tie, pattern,
                    )
                    choices.append((rank, pattern, proposed))
                if choices and size == desired_size:
                    break
            if not choices:
                return None
            choices.sort(key=lambda item: item[0])
            _rank, selected, remaining = choices[0]
            rows.append(selected)
            previous = selected
        return rows if not any(remaining) else None

    extension_cache: dict[tuple[int, ...], list[tuple[int, ...]]] = {}

    def extensions(pattern: tuple[int, ...]) -> list[tuple[int, ...]]:
        bays = tuple(index + 1 for index in pattern)
        if bays not in extension_cache:
            required_bays = set(bays)
            extension_cache[bays] = [
                config for config in legal_configurations
                if required_bays.issubset(config)
            ]
        return extension_cache[bays]

    def decode(pattern_rows: list[tuple[int, ...]]) -> _CandidateSchedule | None:
        records: list[dict[str, Any] | None] = [None] * len(pattern_rows)
        idle_counts = [0] * M
        configuration_frequency: dict[tuple[int, ...], int] = {}
        option_rows: list[
            tuple[int, tuple[int, ...], list[tuple[int, ...]], int]
        ] = []
        for row_index, pattern in enumerate(pattern_rows):
            options = list(extensions(pattern))
            if row_index == 0:
                options = [config for config in options if config == initial_row]
            if not options:
                return None
            work_bays = {index + 1 for index in pattern}
            idle_variants = {
                tuple(q for q, bay in enumerate(config) if bay not in work_bays)
                for config in options
            }
            option_rows.append((row_index, pattern, options, len(idle_variants)))

        # Constrained work patterns must claim their feasible idle identities
        # first.  A chronological greedy pass lets flexible early rows consume
        # capacity that a later pattern cannot use, producing avoidable load
        # imbalance.
        option_rows.sort(key=lambda item: (
            0 if item[0] == 0 else 1,
            item[3], len(item[2]), item[1], item[0],
        ))
        for row_index, pattern, options, _variant_count in option_rows:
            ranked: list[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]] = []
            work_bays = {index + 1 for index in pattern}
            for config in options:
                idle_cranes = tuple(
                    q for q, bay in enumerate(config) if bay not in work_bays
                )
                projected = list(idle_counts)
                for q in idle_cranes:
                    projected[q] += 1
                rank = (
                    max(projected, default=0),
                    sum(value * value for value in projected),
                    -configuration_frequency.get(config, 0),
                    config,
                )
                ranked.append((rank, config, idle_cranes))
            ranked.sort(key=lambda item: item[0])
            _rank, config, idle_cranes = ranked[0]
            for q in idle_cranes:
                idle_counts[q] += 1
            configuration_frequency[config] = configuration_frequency.get(config, 0) + 1
            records[row_index] = {
                "config": config,
                "work": tuple(index + 1 for index in pattern),
                "idle_cranes": idle_cranes,
            }
            stats["configuration_extensions"] += 1

        if any(record is None for record in records):
            return None
        completed_records = [record for record in records if record is not None]
        opening = completed_records[0]
        remaining_records = list(completed_records[1:])
        full_indices = [
            index for index, record in enumerate(remaining_records)
            if len(record["work"]) == M
        ]
        terminal = None
        if full_indices:
            terminal_index = min(
                full_indices,
                key=lambda index: sum(
                    before != after
                    for before, after in zip(
                        remaining_records[index]["config"], opening["config"]
                    )
                ),
            )
            terminal = remaining_records.pop(terminal_index)

        movement_ordered = [opening]
        current = opening["config"]
        last_idle: tuple[int, ...] = ()
        while remaining_records:
            best_index = min(
                range(len(remaining_records)),
                key=lambda index: (
                    sum(
                        before != after
                        for before, after in zip(
                            current, remaining_records[index]["config"]
                        )
                    ),
                    int(
                        bool(last_idle)
                        and remaining_records[index]["idle_cranes"] == last_idle
                    ),
                    -configuration_frequency.get(
                        remaining_records[index]["config"], 0
                    ),
                    remaining_records[index]["config"],
                    remaining_records[index]["work"],
                ),
            )
            record = remaining_records.pop(best_index)
            movement_ordered.append(record)
            current = record["config"]
            last_idle = record["idle_cranes"]
        if terminal is not None:
            movement_ordered.append(terminal)

        # A second ordering preserves each compact work-pattern run.  Within a
        # run, equal configurations are grouped and configuration changes are
        # monotone, so a crane does not leave a bay merely to return to it a
        # few rows later because of the global nearest-neighbour tour.
        continuity_ordered = [opening]
        index = 1
        current = opening["config"]
        while index < len(completed_records):
            end = index + 1
            work = completed_records[index]["work"]
            while (
                end < len(completed_records)
                and completed_records[end]["work"] == work
            ):
                end += 1
            run = list(completed_records[index:end])
            while run:
                next_config = min(
                    {record["config"] for record in run},
                    key=lambda config: (
                        sum(
                            before != after
                            for before, after in zip(current, config)
                        ),
                        config,
                    ),
                )
                same = [record for record in run if record["config"] == next_config]
                continuity_ordered.extend(same)
                run = [record for record in run if record["config"] != next_config]
                current = next_config
            index = end
        if len(continuity_ordered[-1]["work"]) < M:
            full_index = next(
                (
                    index for index in range(len(continuity_ordered) - 2, 0, -1)
                    if len(continuity_ordered[index]["work"]) == M
                ),
                None,
            )
            if full_index is not None:
                continuity_ordered.append(continuity_ordered.pop(full_index))

        def continuity_beam_order() -> list[dict[str, Any]] | None:
            """Order compressed row groups while penalizing reopened work."""
            opening_key = (
                opening["config"], opening["work"], opening["idle_cranes"]
            )
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for record in completed_records[1:]:
                key = (
                    record["config"], record["work"], record["idle_cranes"]
                )
                grouped.setdefault(key, []).append(record)
            # If the opening row occurs again, keep its remaining copies as a
            # normal block that the beam can place immediately after t=0.
            groups = [
                {
                    "records": values,
                    "config": key[0],
                    "work": key[1],
                    "active": frozenset(
                        (q, bay)
                        for q, bay in enumerate(key[0])
                        if bay in set(key[1])
                    ),
                }
                for key, values in grouped.items()
            ]
            if not groups or len(groups) > 22:
                return None
            opening_active = frozenset(
                (q, bay)
                for q, bay in enumerate(opening["config"])
                if bay in set(opening["work"])
            )
            full_mask = (1 << len(groups)) - 1
            # (reopens, moves, mask, last, closed, path)
            states: list[tuple[Any, ...]] = [
                (0, 0, 0, -1, frozenset(), tuple())
            ]
            beam_width = 6000
            for depth in range(len(groups)):
                next_states: dict[tuple[Any, ...], tuple[Any, ...]] = {}
                for reopens, moves, mask, last, closed, path in states:
                    previous_config = (
                        opening["config"] if last < 0
                        else groups[last]["config"]
                    )
                    previous_active = (
                        opening_active if last < 0
                        else groups[last]["active"]
                    )
                    for index, group in enumerate(groups):
                        bit = 1 << index
                        if mask & bit:
                            continue
                        if depth == len(groups) - 1 and len(group["work"]) < M:
                            continue
                        active = group["active"]
                        new_reopens = reopens + len(active & closed)
                        new_moves = moves + sum(
                            before != after
                            for before, after in zip(
                                previous_config, group["config"]
                            )
                        )
                        new_closed = closed | (previous_active - active)
                        new_mask = mask | bit
                        new_path = path + (index,)
                        key = (new_mask, index, new_closed)
                        item = (
                            new_reopens, new_moves, new_mask, index,
                            new_closed, new_path,
                        )
                        old = next_states.get(key)
                        if old is None or item[:2] < old[:2]:
                            next_states[key] = item
                states = sorted(
                    next_states.values(),
                    key=lambda item: (
                        item[0], item[1],
                        -len(groups[item[3]]["active"] & item[4]),
                        item[5],
                    ),
                )[:beam_width]
                if not states or time.perf_counter() >= deadline:
                    return None
            complete = [state for state in states if state[2] == full_mask]
            if not complete:
                return None
            best_state = min(complete, key=lambda item: item[:2])
            ordered = [opening]
            for group_index in best_state[5]:
                ordered.extend(groups[group_index]["records"])
            return ordered

        beam_ordered = continuity_beam_order()

        def build_candidate(
            ordered: Sequence[dict[str, Any]],
        ) -> _CandidateSchedule | None:
            history = [tuple(record["config"]) for record in ordered]
            history.append(history[-1])
            work_plan: dict[tuple[int, int], int | None] = {}
            for time_index, record in enumerate(ordered):
                work_bays = set(record["work"])
                for q, bay in enumerate(record["config"]):
                    work_plan[(time_index, q)] = (
                        bay if bay in work_bays else None
                    )
            try:
                candidate = _candidate_from_rows_and_work_plan(
                    W, M, history, work_plan
                )
            except (RuntimeError, ValueError):
                return None
            stats["decoded"] += 1
            candidate = _normalize_completed_candidate(candidate)
            if not _candidate_passes_independent_verifier(
                W, M, starts, candidate
            ):
                return None
            stats["verified"] += 1
            return candidate

        candidates = [
            candidate
            for candidate in (
                build_candidate(movement_ordered),
                build_candidate(continuity_ordered),
                build_candidate(beam_ordered) if beam_ordered is not None else None,
            )
            if candidate is not None
        ]
        # Feed the lowest-move skeleton into the existing fixed-horizon polish;
        # continuity-aware synchronization is applied after that polish.  An
        # early continuity choice can trap the later search at a worse K.
        return min(candidates, key=lambda item: item.objective_key) if candidates else None

    best: _CandidateSchedule | None = None
    for horizon in range(lower_bound, source.completion_time):
        stats["target_horizons"].append(horizon)
        for attempt in range(max_attempts_per_horizon):
            if time.perf_counter() >= deadline:
                stats.update({"status": "UNKNOWN_DEADLINE", "stop_reason": "deadline"})
                return best, stats
            stats["attempts"] += 1
            pattern_rows = work_patterns_for_horizon(horizon, attempt)
            if pattern_rows is None:
                continue
            stats["rows_selected"] += len(pattern_rows)
            candidate = decode(pattern_rows)
            if candidate is None or candidate.objective_key >= source.objective_key:
                continue
            if best is None or _recommended_schedule_rank(
                candidate, M
            ) < _recommended_schedule_rank(best, M):
                best = candidate
        if best is not None and best.completion_time == horizon:
            stats.update({
                "status": "FOUND",
                "best_objective": list(best.objective_key),
                "best_loads": list(best.loads),
                "best_balance": _balanced_schedule_metrics(best, M),
                "stop_reason": "lower_horizon_verified",
            })
            return best, stats
    if best is None and stats["status"] == "NOT_RUN":
        stats.update({"status": "NO_CANDIDATE", "stop_reason": "bounded_search_exhausted"})
    return best, stats


def _global_balanced_assignment_search(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    source: _CandidateSchedule,
    deadline: float,
    *,
    max_partitions: int = 48,
    max_timing_variants: int = 96,
) -> tuple[_CandidateSchedule | None, dict[str, Any]]:
    """Build globally rebalanced, monotone work plans and schedule their phases.

    Work is viewed as one ordered sequence of bay-work units.  Candidate crane
    assignments cut that sequence into contiguous segments, so a bay can be
    shared only by adjacent cranes and at most two cranes.  A small beam
    explores balanced cut positions; a second bounded search tries leading
    waits that can resolve event-time safety conflicts.  Complete schedules
    are independently verified before they are returned.

    This operator is independent of source revisits and source work blocks.
    It is intended to supply a new global skeleton when local repair has no
    useful defect to focus on.
    """
    total = sum(int(value) for value in W)
    workload_lower_bound = max(
        max(W, default=0),
        math.ceil(total / max(1, M)),
    )
    stats: dict[str, Any] = {
        "status": "NOT_RUN",
        "workload_lower_bound": workload_lower_bound,
        "target_horizons": [],
        "partition_states": 0,
        "partitions_tested": 0,
        "timing_variants_tested": 0,
        "schedules_decoded": 0,
        "schedules_verified": 0,
        "best_objective": None,
        "best_loads": None,
        "stop_reason": None,
    }
    if source.move_time != 0:
        stats.update({"status": "UNSUPPORTED", "stop_reason": "nonzero_move_time"})
        return None, stats
    if M <= 0 or total <= 0 or not source.slots:
        stats.update({"status": "NO_CANDIDATE", "stop_reason": "empty_instance"})
        return None, stats
    if max_partitions <= 0 or max_timing_variants <= 0:
        stats.update({"status": "UNKNOWN_STATE_LIMIT", "stop_reason": "zero_search_limit"})
        return None, stats

    N = len(W)
    source_rows = _candidate_position_rows(source, M)
    initial_row = tuple(source_rows[0])
    eligibility = _bay_eligibility(N, M, [])
    mandatory_start_by_crane: dict[int, int] = {}
    for bay in starts:
        if bay not in initial_row:
            stats.update({
                "status": "NO_CANDIDATE",
                "stop_reason": f"required_start_missing_from_source:{bay}",
            })
            return None, stats
        crane = initial_row.index(bay)
        if W[bay - 1] <= 0 or crane in mandatory_start_by_crane:
            stats.update({
                "status": "NO_CANDIDATE",
                "stop_reason": f"invalid_required_start:{bay}",
            })
            return None, stats
        mandatory_start_by_crane[crane] = int(bay)

    # Reserve the source schedule's mandatory first work slot at each start bay,
    # then rebalance all remaining work globally.
    mandatory_work_by_bay = {bay: crane for crane, bay in mandatory_start_by_crane.items()}
    remaining_work = [
        int(amount) - int(index + 1 in mandatory_work_by_bay)
        for index, amount in enumerate(W)
    ]
    if any(amount < 0 for amount in remaining_work):
        stats.update({"status": "NO_CANDIDATE", "stop_reason": "invalid_required_start_work"})
        return None, stats
    residual_total = sum(remaining_work)
    mandatory_loads = [int(crane in mandatory_start_by_crane) for crane in range(M)]
    prefix = [0]
    for amount in remaining_work:
        prefix.append(prefix[-1] + amount)

    def bay_for_unit(unit: int) -> int:
        index = bisect.bisect_right(prefix, unit) - 1
        while index < N and remaining_work[index] <= 0:
            index += 1
        if not 0 <= index < N:
            raise ValueError(f"作业单位索引越界：{unit}")
        return index + 1

    def split_bay(cut: int) -> int | None:
        if cut <= 0 or cut >= residual_total:
            return None
        index = bisect.bisect_right(prefix, cut) - 1
        if 0 <= index < N and prefix[index] < cut < prefix[index + 1]:
            return index + 1
        return None

    def segment_is_eligible(crane: int, left: int, right: int) -> bool:
        if right <= left:
            return True
        first_bay = bay_for_unit(left)
        last_bay = bay_for_unit(right - 1)
        return all(
            remaining_work[index] <= 0 or crane in eligibility[index]
            for index in range(first_bay - 1, last_bay)
        )

    def candidate_cuts(left: int, low: int, high: int, ideal: int) -> list[int]:
        if high - low <= 256:
            return list(range(low, high + 1))
        values = {low, high, min(high, max(low, ideal))}
        for delta in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            values.add(min(high, max(low, ideal - delta)))
            values.add(min(high, max(low, ideal + delta)))
        for boundary in prefix[1:-1]:
            if low <= boundary <= high:
                values.add(boundary)
            for neighbor in (boundary - 1, boundary + 1):
                if low <= neighbor <= high:
                    values.add(neighbor)
        return sorted(values)

    def partition_candidates(horizon: int) -> list[tuple[int, ...]]:
        # State: end cuts (the previous crane's cumulative work boundary),
        # interior bay cuts already used, and their score.
        beam: list[tuple[tuple[int, ...], frozenset[int]]] = [((), frozenset())]
        for crane in range(M):
            remaining_cranes = M - crane - 1
            expanded: list[tuple[tuple[int, ...], frozenset[int], tuple[Any, ...]]] = []
            for cuts, split_bays in beam:
                left = cuts[-1] if cuts else 0
                next_capacity = sum(
                    max(0, horizon - mandatory_loads[index])
                    for index in range(crane + 1, M)
                )
                crane_capacity = max(0, horizon - mandatory_loads[crane])
                low = max(left, residual_total - next_capacity)
                high = min(residual_total, left + crane_capacity)
                if crane == M - 1:
                    low = high = residual_total
                if low > high:
                    continue
                ideal = round(total * (crane + 1) / M) - sum(mandatory_loads[:crane + 1])
                ideal = min(residual_total, max(0, ideal))
                for right in candidate_cuts(left, low, high, ideal):
                    stats["partition_states"] += 1
                    if right - left > crane_capacity or not segment_is_eligible(crane, left, right):
                        continue
                    split = split_bay(right)
                    if split is not None and split in split_bays:
                        continue
                    new_splits = split_bays | ({split} if split is not None else set())
                    new_cuts = cuts + (right,)
                    loads = [
                        new_cuts[index] - (new_cuts[index - 1] if index else 0)
                        for index in range(len(new_cuts))
                    ]
                    full_loads = [
                        load + mandatory_loads[index]
                        for index, load in enumerate(loads)
                    ]
                    balance = sum((M * load - total) ** 2 for load in full_loads)
                    ideal_deviation = sum(
                        abs(
                            new_cuts[index] + sum(mandatory_loads[:index + 1])
                            - round(total * (index + 1) / M)
                        )
                        for index in range(len(new_cuts))
                    )
                    rank = (
                        max(full_loads, default=0), balance, len(new_splits),
                        ideal_deviation, new_cuts,
                    )
                    expanded.append((new_cuts, frozenset(new_splits), rank))
            if not expanded:
                return []
            expanded.sort(key=lambda item: item[2])
            kept: list[tuple[tuple[int, ...], frozenset[int]]] = []
            seen: set[tuple[int, ...]] = set()
            for cuts, splits, _rank in expanded:
                if cuts in seen:
                    continue
                seen.add(cuts)
                kept.append((cuts, splits))
                if len(kept) >= max_partitions:
                    break
            beam = kept
        if residual_total == 0:
            return [tuple(0 for _ in range(M))]
        return [cuts for cuts, _splits in beam if cuts and cuts[-1] == residual_total]

    def phases_for_cuts(cuts: tuple[int, ...]) -> list[list[tuple[int, int]]]:
        phases: list[list[tuple[int, int]]] = []
        left = 0
        for crane, right in enumerate(cuts):
            crane_phases: list[tuple[int, int]] = []
            for index, amount in enumerate(remaining_work):
                if amount <= 0:
                    continue
                overlap = min(right, prefix[index + 1]) - max(left, prefix[index])
                if overlap > 0:
                    crane_phases.append((index + 1, overlap))
            phases.append(crane_phases)
            left = right
        owners_by_bay: dict[int, set[int]] = {}
        for bay, crane in mandatory_work_by_bay.items():
            owners_by_bay.setdefault(bay, set()).add(crane)
        for crane, crane_phases in enumerate(phases):
            for bay, _amount in crane_phases:
                owners_by_bay.setdefault(bay, set()).add(crane)
        if any(len(owners) > 2 for owners in owners_by_bay.values()):
            return []
        return phases

    def offset_vectors(
        phases: list[list[tuple[int, int]]], horizon: int,
    ) -> list[tuple[int, ...]]:
        choices_by_crane: list[list[int]] = []
        for crane, crane_phases in enumerate(phases):
            load = sum(amount for _bay, amount in crane_phases)
            if not crane_phases:
                choices_by_crane.append([0])
                continue
            first_bay = crane_phases[0][0]
            earliest_start = 1 if crane in mandatory_start_by_crane else (
                0 if first_bay == initial_row[crane] else 1
            )
            latest_start = horizon - load
            if latest_start < earliest_start:
                return []
            slack = latest_start - earliest_start
            if slack <= 8:
                delays = list(range(slack + 1))
            else:
                delays = sorted({0, 1, 2, 4, 8, slack})
            choices_by_crane.append([earliest_start + delay for delay in delays])

        vectors: list[tuple[int, ...]] = [()]
        for choices in choices_by_crane:
            expanded = [vector + (value,) for vector in vectors for value in choices]
            expanded.sort(key=lambda vector: (sum(vector), vector))
            vectors = expanded[:max_timing_variants]
        return vectors

    lower_targets = [workload_lower_bound]
    gap = max(0, source.completion_time - workload_lower_bound)
    if gap:
        for delta in (1, 2, 4, 8, 16, 32, max(1, gap // 4),
                      max(1, gap // 2), gap - 1):
            target = workload_lower_bound + delta
            if target < source.completion_time:
                lower_targets.append(target)
    targets = sorted(set(lower_targets))
    stats["target_horizons"] = targets

    best: _CandidateSchedule | None = None
    for horizon in targets:
        if time.perf_counter() >= deadline:
            stats.update({"status": "UNKNOWN_DEADLINE", "stop_reason": "deadline"})
            break
        partitions = partition_candidates(horizon)
        if not partitions:
            continue
        candidates_for_horizon: list[_CandidateSchedule] = []
        for cuts in partitions:
            if time.perf_counter() >= deadline:
                stats.update({"status": "UNKNOWN_DEADLINE", "stop_reason": "deadline"})
                break
            stats["partitions_tested"] += 1
            phases = phases_for_cuts(cuts)
            if not phases:
                continue
            for offsets in offset_vectors(phases, horizon):
                if time.perf_counter() >= deadline:
                    stats.update({"status": "UNKNOWN_DEADLINE", "stop_reason": "deadline"})
                    break
                stats["timing_variants_tested"] += 1
                rows: list[tuple[int, ...]] = []
                work_plan: dict[tuple[int, int], int | None] = {}
                for time_index in range(horizon):
                    row: list[int] = []
                    for crane, crane_phases in enumerate(phases):
                        phase_start = offsets[crane]
                        if time_index == 0 and crane in mandatory_start_by_crane:
                            position = initial_row[crane]
                            bay = mandatory_start_by_crane[crane]
                        elif time_index < phase_start or not crane_phases:
                            position = initial_row[crane]
                            bay = None
                        else:
                            elapsed = time_index - phase_start
                            phase_end = 0
                            position = crane_phases[-1][0]
                            bay = None
                            for phase_bay, amount in crane_phases:
                                phase_end += amount
                                if elapsed < phase_end:
                                    position = phase_bay
                                    bay = phase_bay
                                    break
                        row.append(position)
                        work_plan[(time_index, crane)] = bay
                    rows.append(tuple(row))
                if not rows or rows[0] != initial_row:
                    continue
                first_work = {
                    bay for (time_index, _crane), bay in work_plan.items()
                    if time_index == 0 and bay is not None
                }
                if not set(starts).issubset(first_work):
                    continue
                if any(
                    any(right - left < 2 for left, right in zip(row, row[1:]))
                    for row in rows
                ):
                    continue
                rows.append(rows[-1])
                try:
                    candidate = _candidate_from_rows_and_work_plan(
                        W, M, rows, work_plan
                    )
                except (RuntimeError, ValueError):
                    continue
                stats["schedules_decoded"] += 1
                candidate = _normalize_completed_candidate(candidate)
                if not _candidate_passes_independent_verifier(W, M, starts, candidate):
                    continue
                stats["schedules_verified"] += 1
                if candidate.objective_key >= source.objective_key:
                    continue
                candidates_for_horizon.append(candidate)
                if len(candidates_for_horizon) >= 16:
                    break
            if stats["status"] == "UNKNOWN_DEADLINE" or len(candidates_for_horizon) >= 16:
                break
        if candidates_for_horizon:
            candidates_for_horizon.sort(key=lambda item: (
                item.objective_key,
                _execution_rank(item, M),
            ))
            best = candidates_for_horizon[0]
            stats.update({
                "status": "FOUND",
                "best_objective": list(best.objective_key),
                "best_loads": list(best.loads),
                "stop_reason": "target_feasible",
            })
            break
        if stats["status"] == "UNKNOWN_DEADLINE":
            break

    if best is None and stats["status"] not in {"UNKNOWN_DEADLINE", "UNSUPPORTED"}:
        stats.update({"status": "NO_CANDIDATE", "stop_reason": "bounded_search_exhausted"})
    return best, stats


def _cumulative_local_trajectory_repair_iterative(
    W: Sequence[int],
    M: int,
    starts: Sequence[int],
    incumbent: _CandidateSchedule,
    deadline: float,
    seed: int,
    move_time: int = 1,
    attempt_trace: list[dict[str, Any]] | None = None,
    continuity_output: dict[str, Any] | None = None,
    *,
    enable_descent: bool = True,
    enable_operational_repairs: bool = True,
    preserve_horizon: bool = False,
    enable_work_transfer: bool = False,
    enable_fragmentation_repair: bool = False,
    enable_cyclic_exchange: bool = True,
    enable_phase_resequence: bool = True,
    enable_phase_closure: bool = True,
    enable_cross_crane_phase_relay: bool = False,
    enable_idle_capacity_rebalance: bool = False,
    enable_forced_prefix_consolidation: bool = True,
    local_state_limit: int = 256,
    protect_source_continuity: bool = False,
    strict_local_transactions: bool = False,
    source_hash: str | None = None,
    use_legacy_seed: bool = False,
    enable_multi_relay: bool = True,
    execution_pool_sort: bool = True,
    enable_global_rebalance: bool = True,
    local_windows_only: bool = False,
) -> tuple[
    _CandidateSchedule | None, int, _CandidateSchedule, _CandidateSchedule | None
]:
    """Run several bounded H-1 repairs under one shared deadline.

    The legacy cumulative routine stops after the first shortened schedule.
    This wrapper keeps the old local operators and changes their orchestration:
    each newly found horizon becomes the source for the next target horizon,
    while fixed-H polishing receives only a bounded slice of the remaining
    budget.  No call changes the declared local-window contract.
    """
    started = time.perf_counter()
    if local_windows_only:
        # Strict local mode keeps every mutation inside the critical-window
        # and adjacent-crane transaction families.  In particular it disables
        # global pattern reconstruction and the full-horizon post-processes.
        enable_global_rebalance = False
    if move_time != 0:
        if continuity_output is not None:
            continuity_output.update({
                "status": "UNSUPPORTED",
                "stop_reason": "nonzero_move_time_trajectory_descent_unsupported",
                "descent_history": [],
                "first_feasible_by_h": [],
            })
        return None, 0, incumbent, None

    incumbent = _normalize_completed_candidate(incumbent)
    safe_lower_bound = _congestion_workload_lower_bound(W, M)
    current = incumbent
    prepared = incumbent
    evaluated_total = 0
    first_feasible: _CandidateSchedule | None = None
    first_by_h: dict[int, _CandidateSchedule] = {incumbent.makespan: incumbent}
    formal_best = incumbent
    continuity_best = incumbent
    operational_best = incumbent
    execution_best = incumbent
    balanced_best = incumbent
    recommended_best = incumbent
    compression_best = incumbent
    candidate_pool: list[_CandidateSchedule] = [incumbent]
    descent_history: list[dict[str, Any]] = []
    round_index = 0
    stale_rounds = 0
    stop_reason = "deadline"
    descent_fraction = (
        0.55 if local_windows_only
        else (0.85 if enable_phase_closure else 0.90)
    )
    descent_phase_deadline = min(
        deadline,
        started + descent_fraction * max(0.0, deadline - started),
    )

    def add_pool(item: _CandidateSchedule) -> None:
        nonlocal candidate_pool
        signature = _trajectory_signature(item)
        for index, old in enumerate(candidate_pool):
            if _trajectory_signature(old) == signature:
                if item.objective_key < old.objective_key:
                    candidate_pool[index] = item
                return
        candidate_pool.append(item)
        ranked = sorted(candidate_pool, key=lambda value: (
            (
                _execution_rank(value, M)
                if execution_pool_sort else
                (value.completion_time, value.objective_key, _operational_rank(value, M))
            ),
            value.objective_key,
        ))
        if execution_pool_sort:
            protected = [
                formal_best, execution_best, balanced_best,
                recommended_best, compression_best,
            ]
            kept: dict[tuple[Any, ...], _CandidateSchedule] = {}
            for champion in protected:
                signature = _trajectory_signature(champion)
                kept[signature] = champion
            for value in ranked:
                if len(kept) >= 16:
                    break
                kept.setdefault(_trajectory_signature(value), value)
            candidate_pool = sorted(kept.values(), key=lambda value: (
                _execution_rank(value, M), value.objective_key,
            ))
        else:
            candidate_pool = ranked[:16]

    def preparation_score(item: _CandidateSchedule) -> tuple[int, ...]:
        """Score a same-H state by its ability to lose the next row."""
        return tuple(_shortening_potential(W, M, item)[0])

    def register(item: _CandidateSchedule, phase: str) -> None:
        nonlocal formal_best, continuity_best, operational_best, execution_best, balanced_best, recommended_best, compression_best, first_feasible
        workload = [0] * len(W)
        for slot in item.slots:
            if slot.state == "work" and slot.work_bay is not None:
                workload[int(slot.work_bay) - 1] += 1
        if workload != list(W):
            return
        completion = item.completion_time
        if completion not in first_by_h:
            first_by_h[completion] = item
            if completion < incumbent.completion_time and first_feasible is None:
                first_feasible = item
            if attempt_trace is not None:
                attempt_trace.append({
                    "phase": "first_feasible_by_completion_time",
                    "completion_time": completion,
                    "schedule_horizon": item.schedule_horizon,
                    "source_phase": phase,
                    "time_from_start": round(time.perf_counter() - started, 6),
                    "objective": list(item.objective_key),
                })
        if item.objective_key < formal_best.objective_key:
            formal_best = item
        if _continuity_rank(item, M) < _continuity_rank(continuity_best, M):
            continuity_best = item
        if _operational_rank(item, M) < _operational_rank(operational_best, M):
            operational_best = item
        if _execution_rank(item, M) < _execution_rank(execution_best, M):
            execution_best = item
        if _balanced_schedule_metrics(item, M)["key"] < _balanced_schedule_metrics(
            balanced_best, M
        )["key"]:
            balanced_best = item
        if _recommended_schedule_rank(item, M) < _recommended_schedule_rank(
            recommended_best, M
        ):
            recommended_best = item
        item_compression_rank = (
            tuple(_shortening_potential(W, M, item)[0]), item.completion_time,
        )
        champion_compression_rank = (
            tuple(_shortening_potential(W, M, compression_best)[0]),
            compression_best.completion_time,
        )
        if item_compression_rank < champion_compression_rank:
            compression_best = item
        add_pool(item)

    def polish(item: _CandidateSchedule, seconds: float) -> _CandidateSchedule:
        nonlocal evaluated_total, first_feasible
        if not enable_operational_repairs or seconds <= 0.05:
            return _normalize_completed_candidate(item)
        details: dict[str, Any] = {}
        local_deadline = min(deadline, time.perf_counter() + seconds)
        polished, evaluated = _refine_same_horizon_trajectory(
            W, M, starts, item, local_deadline,
            seed + 700_001 + round_index,
            move_time=0,
            attempt_trace=attempt_trace,
            continuity=True,
            result_box=details,
            enable_work_transfer=enable_work_transfer,
            enable_fragmentation_repair=enable_fragmentation_repair,
            enable_cyclic_exchange=enable_cyclic_exchange,
            enable_phase_resequence=(
                enable_phase_resequence and not local_windows_only
            ),
            enable_phase_closure=(enable_phase_closure and not local_windows_only),
            enable_cross_crane_phase_relay=(
                enable_cross_crane_phase_relay and not local_windows_only
            ),
            enable_idle_capacity_rebalance=enable_idle_capacity_rebalance,
            enable_forced_prefix_consolidation=enable_forced_prefix_consolidation,
            local_state_limit=local_state_limit,
            protect_source_continuity=protect_source_continuity,
            strict_local_transactions=strict_local_transactions,
            source_hash=source_hash,
            enable_multi_relay=enable_multi_relay,
        )
        evaluated_total += evaluated
        formal = details.get("formal_best", polished)
        continuity = details.get("continuity_best", polished)
        operational = details.get("operational_best", polished)
        execution = details.get("execution_best", operational)
        balanced = details.get("balanced_best", execution)
        for value, phase in (
            (formal, "polish_formal"),
            (continuity, "polish_continuity"),
            (operational, "polish_operational"),
            (execution, "polish_execution"),
            (balanced, "polish_balanced"),
        ):
            register(value, phase)
        if (
            first_feasible is None
            and item.completion_time < incumbent.completion_time
        ):
            first_feasible = item
        if continuity_output is not None:
            continuity_output.setdefault("polish_runs", []).append({
                "completion_time": item.completion_time,
                "schedule_horizon": item.schedule_horizon,
                "evaluated": evaluated,
                "stats": details.get("stats"),
                "paired_transactions": details.get("paired_transactions", []),
                "phase_transactions": details.get("phase_transactions", []),
                "phase_closure_transactions": details.get(
                    "phase_closure_transactions", []
                ),
                "cross_crane_phase_transactions": details.get(
                    "cross_crane_phase_transactions", []
                ),
                "idle_capacity_transactions": details.get(
                    "idle_capacity_transactions", []
                ),
                "forced_prefix_transactions": details.get(
                    "forced_prefix_transactions", []
                ),
                "stop_reason": details.get("stop_reason"),
            })
        return _normalize_completed_candidate(formal)

    def attempt_source(
        source: _CandidateSchedule,
        local_deadline: float,
        source_index: int,
    ) -> tuple[_CandidateSchedule | None, _CandidateSchedule, int, str]:
        windows = _critical_repair_windows(source, M)
        if not windows:
            return None, source, 0, "no_windows"
        ordered = (
            windows[source_index % len(windows):]
            + windows[:source_index % len(windows)]
        )
        best_prepared = source
        evaluated = 0
        for index, (chain, window) in enumerate(ordered):
            if time.perf_counter() >= local_deadline:
                break
            remaining_windows = max(1, len(ordered) - index)
            slice_deadline = min(
                local_deadline,
                time.perf_counter() + max(
                    0.05,
                    (local_deadline - time.perf_counter()) / remaining_windows,
                ),
            )
            targeted, targeted_evaluated = _targeted_local_preparation(
                W, M, source, chain, window
            )
            evaluated += targeted_evaluated
            if targeted is not None:
                best_prepared = targeted
                if targeted.makespan < source.makespan:
                    return targeted, best_prepared, evaluated, "targeted"

            prepare_deadline = min(
                slice_deadline,
                time.perf_counter()
                + 0.55 * max(0.0, slice_deadline - time.perf_counter()),
            )
            prepared_candidate, prepared_evaluated, _ = _trajectory_repair(
                W, M, starts, best_prepared, prepare_deadline,
                seed + round_index * 1009 + source_index * 17 + index,
                active_cranes=chain,
                window=window,
                return_state=True,
                move_time=0,
                preserve_horizon=True,
                preparation_mode=True,
            )
            evaluated += prepared_evaluated
            if prepared_candidate is not None:
                best_prepared = prepared_candidate
                shortened = _decode_best_shortening(W, M, prepared_candidate)
                if shortened is not None:
                    return shortened, best_prepared, evaluated, "prepared_decode"

            if time.perf_counter() < slice_deadline:
                shortened, shortened_evaluated, _ = _trajectory_repair(
                    W, M, starts, best_prepared, slice_deadline,
                    seed + 500_003 + round_index * 1009
                    + source_index * 17 + index,
                    active_cranes=chain,
                    window=window,
                    return_state=True,
                    move_time=0,
                )
                evaluated += shortened_evaluated
                if shortened is not None and shortened.makespan < source.makespan:
                    return shortened, best_prepared, evaluated, "window_shorten"
        return None, best_prepared, evaluated, "timeout_or_no_improvement"

    global_rebalance_stats: dict[str, Any] = {
        "status": "NOT_RUN",
        "workload_lower_bound": safe_lower_bound,
        "target_horizons": [],
        "partition_states": 0,
        "partitions_tested": 0,
        "timing_variants_tested": 0,
        "schedules_decoded": 0,
        "schedules_verified": 0,
        "best_objective": None,
        "best_loads": None,
        "stop_reason": "disabled_or_not_applicable",
    }
    global_gap = current.completion_time - safe_lower_bound
    if (
        enable_global_rebalance
        and move_time == 0
        and not preserve_horizon
        and global_gap >= 3
        and time.perf_counter() < deadline
    ):
        global_deadline = min(
            deadline,
            started + 0.55 * max(0.0, deadline - started),
        )
        pattern_candidate, pattern_stats = _global_safe_pattern_search(
            W, M, starts, current, global_deadline,
        )
        global_candidate = pattern_candidate
        global_rebalance_stats = {
            **global_rebalance_stats,
            "safe_pattern_search": pattern_stats,
            "status": pattern_stats.get("status", "NOT_RUN"),
            "workload_lower_bound": safe_lower_bound,
            "target_horizons": pattern_stats.get("target_horizons", []),
            "schedules_decoded": pattern_stats.get("decoded", 0),
            "schedules_verified": pattern_stats.get("verified", 0),
            "best_objective": pattern_stats.get("best_objective"),
            "best_loads": pattern_stats.get("best_loads"),
            "stop_reason": pattern_stats.get("stop_reason"),
        }
        if global_candidate is None and time.perf_counter() < global_deadline:
            global_candidate, legacy_global_stats = _global_balanced_assignment_search(
                W, M, starts, current, global_deadline,
            )
            global_rebalance_stats["monotone_partition_search"] = legacy_global_stats
            if global_candidate is not None:
                global_rebalance_stats.update({
                    "status": legacy_global_stats.get("status"),
                    "target_horizons": legacy_global_stats.get("target_horizons", []),
                    "best_objective": legacy_global_stats.get("best_objective"),
                    "best_loads": legacy_global_stats.get("best_loads"),
                    "stop_reason": legacy_global_stats.get("stop_reason"),
                })
        if global_candidate is not None:
            register(global_candidate, "global_rebalance")
            if global_candidate.completion_time < incumbent.completion_time:
                if first_feasible is None or (
                    global_candidate.completion_time < first_feasible.completion_time
                ):
                    first_feasible = global_candidate
            if global_candidate.objective_key < current.objective_key:
                current = global_candidate
                prepared = global_candidate
                first_by_h[global_candidate.completion_time] = global_candidate
            if attempt_trace is not None:
                attempt_trace.append({
                    "phase": "global_rebalance",
                    "time_from_start": round(time.perf_counter() - started, 6),
                    "objective": list(global_candidate.objective_key),
                    "loads": list(global_candidate.loads),
                    "status": global_rebalance_stats.get("status"),
                })

    legacy_seed_used = False
    if (
        use_legacy_seed
        and enable_descent
        and not preserve_horizon
        and current.makespan > safe_lower_bound
        and time.perf_counter() < descent_phase_deadline
    ):
        # The previous cumulative helper has already demonstrated that this
        # exact source can reach H-1.  Use it only as a bounded first seed and
        # stop it immediately after the first shortening; the new iterative
        # orchestration then continues from that state toward lower horizons.
        legacy_seed_used = True
        legacy_shortened, legacy_evaluated, legacy_prepared, legacy_first = (
            _cumulative_local_trajectory_repair(
                W, M, starts, current, descent_phase_deadline, seed,
                move_time=0, attempt_trace=attempt_trace,
                continuity_output={}, polish_after_first=False,
            )
        )
        evaluated_total += legacy_evaluated
        if legacy_shortened is not None:
            current = legacy_shortened
            prepared = legacy_shortened
            if legacy_first is not None and first_feasible is None:
                first_feasible = legacy_first
            register(legacy_shortened, "legacy_descent")
            descent_history.append({
                "round": round_index,
                "source_horizons": [incumbent.makespan],
                "target_horizon": incumbent.makespan - 1,
                "source_completion_times": [incumbent.completion_time],
                "source_schedule_horizons": [incumbent.schedule_horizon],
                "source_normalization_trimmed_slots": incumbent.normalization_trimmed_slots,
                "target_completion_time": incumbent.completion_time - 1,
                "result": "FOUND_LEGACY_SEED",
                "horizon": legacy_shortened.makespan,
                "result_completion_time": legacy_shortened.completion_time,
                "movement_count": legacy_shortened.completion_movement_count,
                "objective": list(legacy_shortened.objective_key),
                "evaluated": legacy_evaluated,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            })
            round_index += 1
        elif legacy_prepared is not None:
            current = legacy_prepared
            prepared = legacy_prepared
            register(legacy_prepared, "legacy_preparation")

    if not enable_descent:
        stop_reason = "descent_disabled"
    elif preserve_horizon:
        stop_reason = "preserve_horizon"
    else:
        while (
            time.perf_counter() < descent_phase_deadline
            and current.makespan > safe_lower_bound
        ):
            source_pool = sorted(
                [item for item in candidate_pool
                 if item.completion_time == current.completion_time],
                key=lambda item: (
                    item.makespan,
                    item.objective_key,
                    _operational_rank(item, M),
                ),
            )
            # One source gets a complete round.  Dividing a 20-second round
            # across all elite alternatives made each of the 48 bounded
            # windows too short to reproduce the old trajectory search.
            sources = [source_pool[round_index % len(source_pool)]] if source_pool else [current]
            remaining = descent_phase_deadline - time.perf_counter()
            round_deadline = min(
                descent_phase_deadline,
                time.perf_counter() + max(
                    0.20, min(20.0, remaining)
                ),
            )
            found = None
            best_prepared = current
            round_evaluated = 0
            reason = "no_candidate"
            for source_index, source in enumerate(sources or [current]):
                if time.perf_counter() >= round_deadline:
                    break
                proposed, prepared_candidate, evaluated, reason = attempt_source(
                    source, round_deadline, source_index
                )
                round_evaluated += evaluated
                if preparation_score(prepared_candidate) < preparation_score(best_prepared):
                    best_prepared = prepared_candidate
                if (
                    proposed is not None
                    and proposed.completion_time < current.completion_time
                ):
                    found = proposed
                    break
            evaluated_total += round_evaluated
            preparation_progress = (
                best_prepared.makespan < current.makespan
                or preparation_score(best_prepared) < preparation_score(current)
            )
            if preparation_progress:
                current = best_prepared
                prepared = best_prepared
                register(current, "preparation")
            if found is None:
                stale_rounds = 0 if preparation_progress else stale_rounds + 1
                descent_history.append({
                    "round": round_index,
                    "source_horizons": [item.makespan for item in sources],
                    "target_horizon": min((item.makespan for item in sources), default=current.makespan) - 1,
                    "source_completion_times": [item.completion_time for item in sources],
                    "source_schedule_horizons": [item.schedule_horizon for item in sources],
                    "source_normalization_trimmed_slots": [
                        item.normalization_trimmed_slots for item in sources
                    ],
                    "target_completion_time": min(
                        (item.completion_time for item in sources),
                        default=current.completion_time,
                    ) - 1,
                    "result": "PREPARED_NO_SHORTENING" if preparation_progress else "NO_IMPROVEMENT",
                    "result_completion_time": formal_best.completion_time,
                    "movement_count": formal_best.completion_movement_count,
                    "reason": reason,
                    "evaluated": round_evaluated,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                })
                round_index += 1
                # A bounded local round can legitimately fail while a later
                # window or random continuation succeeds.  Do not hand the
                # majority of the deadline to polishing after only a few
                # misses; leave early only after a long, recorded plateau.
                if stale_rounds >= 20:
                    stop_reason = "descent_stalled"
                    break
                continue

            stale_rounds = 0
            previous_h = current.makespan
            if first_feasible is None and found.makespan < incumbent.makespan:
                first_feasible = found
            if found.makespan not in first_by_h:
                first_by_h[found.makespan] = found
            current = found
            prepared = found
            register(found, "descent")
            current = polish(
                current,
                min(2.0, max(0.05, 0.12 * (deadline - time.perf_counter()))),
            )
            prepared = current
            descent_history.append({
                "round": round_index,
                "source_horizons": [item.makespan for item in sources],
                "target_horizon": previous_h - 1,
                "source_completion_times": [item.completion_time for item in sources],
                "source_schedule_horizons": [item.schedule_horizon for item in sources],
                "source_normalization_trimmed_slots": [
                    item.normalization_trimmed_slots for item in sources
                ],
                "source_completion_time": previous_h,
                "target_completion_time": previous_h - 1,
                "result": "FOUND",
                "horizon": current.makespan,
                "result_completion_time": current.completion_time,
                "movement_count": current.completion_movement_count,
                "objective": list(current.objective_key),
                "evaluated": round_evaluated,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            })
            round_index += 1
            if current.makespan <= safe_lower_bound:
                stop_reason = "safe_workload_lower_bound_reached"
                break

    # Once descent is exhausted, spend the rest of the shared deadline on the
    # best shortest-horizon candidate.  This stage never changes the horizon.
    if enable_operational_repairs and time.perf_counter() < deadline:
        shortest_h = min(item.makespan for item in candidate_pool)
        quality_source = min(
            (item for item in candidate_pool if item.makespan == shortest_h),
            key=(
                (lambda item: (_execution_rank(item, M), item.objective_key))
                if execution_pool_sort else
                (lambda item: (_operational_rank(item, M), item.objective_key))
            ),
            default=current,
        )
        quality = polish(quality_source, max(0.05, deadline - time.perf_counter()))
        prepared = quality
        if quality.makespan < current.makespan:
            current = quality

    if time.perf_counter() >= deadline:
        stop_reason = "deadline"
    elif current.makespan <= safe_lower_bound:
        stop_reason = "safe_workload_lower_bound_reached"

    if attempt_trace is not None:
        attempt_trace.append({
            "phase": "descent_complete",
            "stop_reason": stop_reason,
            "safe_workload_lower_bound": safe_lower_bound,
            "horizons_seen": sorted(first_by_h),
            "descent_rounds": len(descent_history),
            "legacy_seed_used": legacy_seed_used,
            "use_legacy_seed": use_legacy_seed,
            "preserve_horizon": preserve_horizon,
        })
    aggregate_stats = {
        "generated": 0,
        "unique_complete": 0,
        "deduplicated": 0,
        "capacity_rejected": 0,
        "safety_rejected": 0,
        "decoded": 0,
        "verified": 0,
        "accepted": 0,
        "formal_improvements": 0,
        "operational_improvements": 0,
        "execution_improvements": 0,
        "balanced_improvements": 0,
        "pure_sync_delay_rejected": 0,
        "operator": {},
        "continuity_rejected": 0,
        "continuity_component_rejected": 0,
        "work_transfer": {
            "rounds": 0,
            "generated": 0,
            "unique": 0,
            "capacity_rejected": 0,
            "safety_rejected": 0,
            "boundary_rejected": 0,
            "verified": 0,
            "accepted": 0,
            "operators": {},
        },
        "paired_window_cyclic": {
            "status": "NOT_RUN",
            "rounds": 0,
            "generated": 0,
            "unique": 0,
            "expanded_states": 0,
            "early_window_states": 0,
            "late_window_states": 0,
            "capacity_rejected": 0,
            "safety_rejected": 0,
            "boundary_rejected": 0,
            "ledger_rejected": 0,
            "verified": 0,
            "accepted": 0,
            "timeout": 0,
            "state_limit": 0,
            "operators": {},
            "status_counts": {},
        },
        "phase_block_resequence": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_revisits": 0,
            "multi_slot_revisits": 0,
            "active_bands_generated": 0,
            "phase_permutations_generated": 0,
            "phase_combinations_tested": 0,
            "states_expanded": 0,
            "states_pruned_horizon": 0,
            "states_pruned_safety": 0,
            "states_pruned_split": 0,
            "states_pruned_movement": 0,
            "complete_phase_plans": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_burden_migration": 0,
            "timeout": 0,
            "state_limit": 0,
            "operators": {},
            "status_counts": {},
        },
        "phase_closure_relay": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_revisits": 0,
            "activity_bands_tested": 0,
            "phase_orders_generated": 0,
            "event_schedules_generated": 0,
            "states_expanded": 0,
            "complete_phase_plans": 0,
            "ledger_closed": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_safety": 0,
            "rejected_horizon": 0,
            "rejected_ledger": 0,
            "timeout": 0,
            "state_limit": 0,
            "max_activity_width": 0,
            "activity_chain_expansions": [],
            "status_counts": {},
        },
        "cross_crane_phase_relay": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_revisits": 0,
            "activity_bands_tested": 0,
            "assignment_variants_generated": 0,
            "retimed_plans_generated": 0,
            "retimed_plans_solved": 0,
            "owner_change_branches": 0,
            "two_hop_relay_branches": 0,
            "complete_phase_plans": 0,
            "ledger_closed": 0,
            "states_expanded": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_overlap": 0,
            "rejected_eligibility": 0,
            "rejected_safety": 0,
            "rejected_ledger": 0,
            "rejected_horizon": 0,
            "timeout": 0,
            "state_limit": 0,
            "max_activity_width": 0,
            "activity_chain_expansions": [],
            "status_counts": {},
        },
        "idle_capacity_rebalance": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_revisits": 0,
            "idle_capacity_focuses": 0,
            "idle_capacity_chain_focuses": 0,
            "partial_transfer_focuses": 0,
            "focuses_without_revisits": 0,
            "activity_bands_tested": 0,
            "assignment_variants_generated": 0,
            "retimed_plans_generated": 0,
            "retimed_plans_solved": 0,
            "owner_change_branches": 0,
            "two_hop_relay_branches": 0,
            "complete_phase_plans": 0,
            "ledger_closed": 0,
            "states_expanded": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_overlap": 0,
            "rejected_eligibility": 0,
            "rejected_safety": 0,
            "rejected_ledger": 0,
            "rejected_horizon": 0,
            "timeout": 0,
            "state_limit": 0,
            "max_activity_width": 0,
            "activity_chain_expansions": [],
            "status_counts": {},
        },
        "forced_prefix_consolidation": {
            "status": "NOT_RUN",
            "rounds": 0,
            "focus_interruptions": 0,
            "focuses_started": 0,
            "focuses_completed": 0,
            "neighbor_bands_generated": 0,
            "focus_phase_orders": 0,
            "neighbor_phase_orders": 0,
            "idle_placements_tested": 0,
            "states_expanded": 0,
            "ledger_closed": 0,
            "decoded": 0,
            "verified": 0,
            "accepted": 0,
            "rejected_safety": 0,
            "rejected_ledger": 0,
            "rejected_burden_migration": 0,
            "timeout": 0,
            "state_limit": 0,
            "status_counts": {},
        },
    }
    if continuity_output is not None:
        for run in continuity_output.get("polish_runs", []):
            stats = run.get("stats") or {}
            for key in aggregate_stats:
                if key in {
                    "operator", "work_transfer", "paired_window_cyclic",
                    "phase_block_resequence", "phase_closure_relay",
                    "cross_crane_phase_relay",
                    "idle_capacity_rebalance",
                    "forced_prefix_consolidation",
                }:
                    continue
                if isinstance(stats.get(key), int):
                    aggregate_stats[key] += stats[key]
            for name, values in (stats.get("operator") or {}).items():
                destination = aggregate_stats["operator"].setdefault(name, {})
                for key, value in values.items():
                    if isinstance(value, int):
                        destination[key] = destination.get(key, 0) + value
            source_work_stats = stats.get("work_transfer") or {}
            target_work_stats = aggregate_stats["work_transfer"]
            for key in (
                "rounds", "generated", "unique", "capacity_rejected",
                "safety_rejected", "boundary_rejected", "verified",
                "accepted",
            ):
                if isinstance(source_work_stats.get(key), int):
                    target_work_stats[key] += source_work_stats[key]
            for name, value in (source_work_stats.get("operators") or {}).items():
                target_work_stats["operators"][name] = (
                    target_work_stats["operators"].get(name, 0) + int(value)
                )
            source_pair_stats = stats.get("paired_window_cyclic") or {}
            target_pair_stats = aggregate_stats["paired_window_cyclic"]
            if source_pair_stats.get("status") not in (None, "NOT_RUN"):
                target_pair_stats["status"] = source_pair_stats["status"]
            for key in (
                "rounds", "generated", "unique", "expanded_states",
                "early_window_states", "late_window_states",
                "capacity_rejected", "safety_rejected", "boundary_rejected",
                "ledger_rejected", "verified", "accepted", "timeout",
                "state_limit",
            ):
                if isinstance(source_pair_stats.get(key), int):
                    target_pair_stats[key] += source_pair_stats[key]
            for name, value in (source_pair_stats.get("operators") or {}).items():
                target_pair_stats["operators"][name] = (
                    target_pair_stats["operators"].get(name, 0) + int(value)
                )
            status = source_pair_stats.get("status")
            if status is not None:
                target_pair_stats["status_counts"][status] = (
                    target_pair_stats["status_counts"].get(status, 0) + 1
                )
            source_phase_stats = stats.get("phase_block_resequence") or {}
            target_phase_stats = aggregate_stats["phase_block_resequence"]
            if source_phase_stats.get("status") not in (None, "NOT_RUN"):
                target_phase_stats["status"] = source_phase_stats["status"]
            for key in (
                "rounds", "focus_revisits", "multi_slot_revisits",
                "active_bands_generated", "phase_permutations_generated",
                "phase_combinations_tested", "states_expanded",
                "states_pruned_horizon", "states_pruned_safety",
                "states_pruned_split", "states_pruned_movement",
                "complete_phase_plans", "decoded", "verified", "accepted",
                "rejected_burden_migration", "timeout", "state_limit",
            ):
                if isinstance(source_phase_stats.get(key), int):
                    target_phase_stats[key] += source_phase_stats[key]
            for name, value in (source_phase_stats.get("operators") or {}).items():
                target_phase_stats["operators"][name] = (
                    target_phase_stats["operators"].get(name, 0) + int(value)
                )
            phase_status = source_phase_stats.get("status")
            if phase_status is not None:
                target_phase_stats["status_counts"][phase_status] = (
                    target_phase_stats["status_counts"].get(phase_status, 0) + 1
                )
            source_closure_stats = stats.get("phase_closure_relay") or {}
            target_closure_stats = aggregate_stats["phase_closure_relay"]
            if source_closure_stats.get("status") not in (None, "NOT_RUN"):
                target_closure_stats["status"] = source_closure_stats["status"]
            for key in (
                "rounds", "focus_revisits", "activity_bands_tested",
                "phase_orders_generated", "event_schedules_generated",
                "states_expanded", "complete_phase_plans", "ledger_closed",
                "decoded", "verified", "accepted", "rejected_safety",
                "rejected_horizon", "rejected_ledger", "timeout",
                "state_limit", "max_activity_width",
            ):
                if isinstance(source_closure_stats.get(key), int):
                    if key == "max_activity_width":
                        target_closure_stats[key] = max(
                            target_closure_stats[key],
                            source_closure_stats[key],
                        )
                    else:
                        target_closure_stats[key] += source_closure_stats[key]
            target_closure_stats["activity_chain_expansions"].extend(
                source_closure_stats.get("activity_chain_expansions", [])
            )
            closure_status = source_closure_stats.get("status")
            if closure_status is not None:
                target_closure_stats["status_counts"][closure_status] = (
                    target_closure_stats["status_counts"].get(closure_status, 0) + 1
                )
            source_relay_stats = stats.get("cross_crane_phase_relay") or {}
            target_relay_stats = aggregate_stats["cross_crane_phase_relay"]
            if source_relay_stats.get("status") not in (None, "NOT_RUN"):
                target_relay_stats["status"] = source_relay_stats["status"]
            for key in (
                "rounds", "focus_revisits", "activity_bands_tested",
                "assignment_variants_generated", "retimed_plans_generated",
                "retimed_plans_solved", "owner_change_branches",
                "two_hop_relay_branches", "complete_phase_plans",
                "ledger_closed", "states_expanded", "decoded", "verified",
                "accepted", "rejected_overlap", "rejected_eligibility",
                "rejected_safety", "rejected_ledger", "rejected_horizon",
                "timeout", "state_limit",
            ):
                if isinstance(source_relay_stats.get(key), int):
                    target_relay_stats[key] += source_relay_stats[key]
            target_relay_stats["max_activity_width"] = max(
                target_relay_stats["max_activity_width"],
                int(source_relay_stats.get("max_activity_width", 0)),
            )
            target_relay_stats["activity_chain_expansions"].extend(
                source_relay_stats.get("activity_chain_expansions", [])
            )
            relay_status = source_relay_stats.get("status")
            if relay_status is not None:
                target_relay_stats["status_counts"][relay_status] = (
                    target_relay_stats["status_counts"].get(relay_status, 0) + 1
                )
            source_balance_stats = stats.get("idle_capacity_rebalance") or {}
            target_balance_stats = aggregate_stats["idle_capacity_rebalance"]
            if source_balance_stats.get("status") not in (None, "NOT_RUN"):
                target_balance_stats["status"] = source_balance_stats["status"]
            for key in (
                "rounds", "focus_revisits", "idle_capacity_focuses",
                "idle_capacity_chain_focuses", "partial_transfer_focuses",
                "focuses_without_revisits", "activity_bands_tested",
                "assignment_variants_generated", "retimed_plans_generated",
                "retimed_plans_solved",
                "owner_change_branches", "two_hop_relay_branches",
                "complete_phase_plans", "ledger_closed", "states_expanded",
                "decoded", "verified", "accepted", "rejected_overlap",
                "rejected_eligibility", "rejected_safety", "rejected_ledger",
                "rejected_horizon", "timeout", "state_limit",
            ):
                if isinstance(source_balance_stats.get(key), int):
                    target_balance_stats[key] += source_balance_stats[key]
            target_balance_stats["max_activity_width"] = max(
                target_balance_stats["max_activity_width"],
                int(source_balance_stats.get("max_activity_width", 0)),
            )
            target_balance_stats["activity_chain_expansions"].extend(
                source_balance_stats.get("activity_chain_expansions", [])
            )
            balance_status = source_balance_stats.get("status")
            if balance_status is not None:
                target_balance_stats["status_counts"][balance_status] = (
                    target_balance_stats["status_counts"].get(balance_status, 0) + 1
                )
            source_prefix_stats = stats.get("forced_prefix_consolidation") or {}
            target_prefix_stats = aggregate_stats["forced_prefix_consolidation"]
            if source_prefix_stats.get("status") not in (None, "NOT_RUN"):
                target_prefix_stats["status"] = source_prefix_stats["status"]
            for key in (
                "rounds", "focus_interruptions", "focuses_started",
                "focuses_completed", "neighbor_bands_generated",
                "focus_phase_orders", "neighbor_phase_orders",
                "idle_placements_tested", "states_expanded", "ledger_closed",
                "decoded", "verified", "accepted", "rejected_safety",
                "rejected_ledger", "rejected_burden_migration", "timeout",
                "state_limit",
            ):
                if isinstance(source_prefix_stats.get(key), int):
                    target_prefix_stats[key] += source_prefix_stats[key]
            prefix_status = source_prefix_stats.get("status")
            if prefix_status is not None:
                target_prefix_stats["status_counts"][prefix_status] = (
                    target_prefix_stats["status_counts"].get(prefix_status, 0) + 1
                )
        aggregate_stats["polish_runs"] = len(continuity_output.get("polish_runs", []))
        paired_transactions = [
            transaction
            for run in continuity_output.get("polish_runs", [])
            for transaction in run.get("paired_transactions", [])
        ]
        phase_transactions = [
            transaction
            for run in continuity_output.get("polish_runs", [])
            for transaction in run.get("phase_transactions", [])
        ]
        phase_closure_transactions = [
            transaction
            for run in continuity_output.get("polish_runs", [])
            for transaction in run.get("phase_closure_transactions", [])
        ]
        cross_crane_phase_transactions = [
            transaction
            for run in continuity_output.get("polish_runs", [])
            for transaction in run.get("cross_crane_phase_transactions", [])
        ]
        idle_capacity_transactions = [
            transaction
            for run in continuity_output.get("polish_runs", [])
            for transaction in run.get("idle_capacity_transactions", [])
        ]
        forced_prefix_transactions = [
            transaction
            for run in continuity_output.get("polish_runs", [])
            for transaction in run.get("forced_prefix_transactions", [])
        ]
    else:
        paired_transactions = []
        phase_transactions = []
        phase_closure_transactions = []
        cross_crane_phase_transactions = []
        idle_capacity_transactions = []
        forced_prefix_transactions = []
    synchronization_candidate = None
    synchronization_stats: dict[str, Any] = {
        "status": "NOT_RUN", "runs": [],
    }
    if (
        not local_windows_only
        and
        first_feasible is not None
        and any(
            _balanced_schedule_metrics(item, M)["max_trailing_idle"] > 0
            or _balanced_schedule_metrics(item, M)["max_internal_idle_blocks"] > 1
            for item in (formal_best, balanced_best)
        )
    ):
        # Balance search may improve the amount of work assigned to an idle
        # crane while introducing several short gaps.  Synchronize both that
        # candidate and the formal low-move incumbent under one bounded tail
        # budget, prioritising the balanced candidate that can actually remove
        # under-loading rather than merely relocating trailing idle.
        synchronization_deadline = time.perf_counter() + 8.0
        synchronization_sources = sorted(
            {id(item): item for item in (balanced_best, formal_best)}.values(),
            key=lambda item: _recommended_schedule_rank(item, M),
        )
        synchronized: list[_CandidateSchedule] = []
        run_stats: list[dict[str, Any]] = []
        for sync_source in synchronization_sources:
            if time.perf_counter() >= synchronization_deadline:
                break
            sync_candidate, sync_stats = _synchronize_by_compressed_row_groups(
                W, M, starts, sync_source, synchronization_deadline,
            )
            run_stats.append({
                "source_objective": list(sync_source.objective_key),
                "source_loads": list(sync_source.loads),
                **sync_stats,
            })
            if sync_candidate is not None:
                synchronized.append(sync_candidate)
        if synchronized:
            synchronization_candidate = min(
                synchronized,
                key=lambda item: _recommended_schedule_rank(item, M),
            )
            recommended_best = min(
                (recommended_best, *synchronized),
                key=lambda item: _recommended_schedule_rank(item, M),
            )
        synchronization_stats = {
            "status": "FOUND" if synchronized else (
                run_stats[-1]["status"] if run_stats else "NOT_RUN"
            ),
            "runs": run_stats,
            "selected_objective": (
                list(synchronization_candidate.objective_key)
                if synchronization_candidate is not None else None
            ),
        }
    contiguous_phase_candidate = None
    contiguous_phase_stats: dict[str, Any] = {"status": "NOT_RUN"}
    if first_feasible is not None and not local_windows_only:
        contiguous_phase_candidate, contiguous_phase_stats = (
            _contiguous_phase_handoff_search(
                W, M, starts, formal_best,
                time.perf_counter() + 15.0,
            )
        )
        if contiguous_phase_candidate is not None:
            recommended_best = min(
                (recommended_best, contiguous_phase_candidate),
                key=lambda item: _recommended_schedule_rank(item, M),
            )
    local_terminal_alignment_stats: dict[str, Any] = {
        "status": "NOT_RUN", "attempted": 0, "improved": 0,
        "runs": [],
    }
    local_terminal_candidates: list[_CandidateSchedule] = []
    if local_windows_only and first_feasible is not None:
        # This is a time-local cleanup, not a global reschedule: only the
        # final contiguous work block may move, and only inside a position run
        # where the crane was already stationary.  Work ownership, movement,
        # reversals and the actual completion time are invariant.
        unique_sources = {
            id(item): item
            for item in (
                recommended_best, formal_best, execution_best, balanced_best,
            )
        }.values()
        for align_source in unique_sources:
            local_terminal_alignment_stats["attempted"] += 1
            before_balance = _balanced_schedule_metrics(align_source, M)
            aligned = _right_shift_terminal_work_blocks(
                W, M, starts, align_source
            )
            after_balance = _balanced_schedule_metrics(aligned, M)
            improved = (
                aligned.objective_key == align_source.objective_key
                and after_balance["max_trailing_idle"]
                < before_balance["max_trailing_idle"]
            )
            local_terminal_alignment_stats["improved"] += int(improved)
            local_terminal_alignment_stats["runs"].append({
                "objective": list(align_source.objective_key),
                "before_finish_times": before_balance["finish_times"],
                "after_finish_times": after_balance["finish_times"],
                "before_max_trailing_idle": before_balance[
                    "max_trailing_idle"
                ],
                "after_max_trailing_idle": after_balance[
                    "max_trailing_idle"
                ],
                "changed": aligned is not align_source,
            })
            local_terminal_candidates.append(aligned)
        local_terminal_alignment_stats["status"] = (
            "IMPROVED"
            if local_terminal_alignment_stats["improved"]
            else "NO_CHANGE"
        )
        aligned_balanced = min(
            local_terminal_candidates,
            key=lambda item: _balanced_schedule_metrics(item, M)["key"],
        )
        if _balanced_schedule_metrics(
            aligned_balanced, M
        )["key"] < _balanced_schedule_metrics(balanced_best, M)["key"]:
            balanced_best = aligned_balanced
    recommended_best = min(
        (
            recommended_best, formal_best, execution_best, balanced_best,
            *local_terminal_candidates,
        ),
        key=lambda item: _recommended_schedule_rank(item, M),
    )
    if continuity_output is not None:
        continuity_output.update({
            "formal_best": formal_best,
            "continuity_best": continuity_best,
            "operational_best": operational_best,
            "execution_best": execution_best,
            "balanced_best": balanced_best,
            "balanced_best_metrics": _balanced_schedule_metrics(balanced_best, M),
            "recommended_best": recommended_best,
            "recommended_best_metrics": _balanced_schedule_metrics(
                recommended_best, M
            ),
            "synchronization_candidate": synchronization_candidate,
            "synchronization_stats": synchronization_stats,
            "contiguous_phase_candidate": contiguous_phase_candidate,
            "contiguous_phase_stats": contiguous_phase_stats,
            "local_terminal_alignment": local_terminal_alignment_stats,
            "compression_best": compression_best,
            "paired_transactions": paired_transactions,
            "phase_transactions": phase_transactions,
            "phase_closure_transactions": phase_closure_transactions,
            "cross_crane_phase_transactions": cross_crane_phase_transactions,
            "idle_capacity_transactions": idle_capacity_transactions,
            "forced_prefix_transactions": forced_prefix_transactions,
            "pool_size": len(candidate_pool),
            "stop_reason": stop_reason,
            "descent_history": descent_history,
            "first_feasible_by_h": [
                {"horizon": horizon, "candidate": item}
                for horizon, item in sorted(first_by_h.items(), reverse=True)
            ],
            "safe_workload_lower_bound": safe_lower_bound,
            "global_rebalance": global_rebalance_stats,
            "local_windows_only": local_windows_only,
            "descent_rounds": len(descent_history),
            "stats": aggregate_stats,
        })
    return (
        recommended_best if first_feasible is not None else None,
        evaluated_total,
        prepared,
        first_feasible,
    )


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
    critical_mode: str = "local_only",
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
    if critical_mode not in {
        "off_reallocate", "off_reserved", "beam", "trajectory", "both",
        "local_only",
    }:
        raise ValueError(
            "critical_mode 必须是 off_reallocate、off_reserved、beam、"
            "trajectory、both 或 local_only。"
        )
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
        if (
            best is not None
            and candidate.completion_time > best.completion_time + 2
            and candidate is not best
        ):
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
        kept = [
            item for item in elite_pool
            if item is best or item.completion_time <= best.completion_time + 2
        ]
        elite_pool[:] = sorted(kept, key=lambda item: item.objective_key)[:16]
        elite_signatures.clear()
        elite_signatures.update(schedule_signature(item) for item in elite_pool)

    def publish(candidate):
        nonlocal published_key
        if on_incumbent is None:
            return
        if published_key is not None and candidate.objective_key >= published_key:
            return
        timed_candidate = _normalize_completed_candidate(
            _retime_candidate(W, M, candidate, move_time)
        )
        optimal = timed_candidate.completion_time == lower_bound
        snapshot = Solution(
            status="COMPLETION_TIME_OPTIMAL_BY_LOWER_BOUND" if optimal else "HEURISTIC_FEASIBLE",
            method="dp_dispatch_priority_evolution_trajectory_repair_mcts_exact_no_solver",
            makespan=timed_candidate.completion_time,
            schedule_horizon=timed_candidate.schedule_horizon,
            schedule_movement_count=timed_candidate.movement_count,
            makespan_lower_bound=lower_bound,
            lower_bound_components=lower_bound_components,
            makespan_proven_optimal=optimal, proven_lexicographic_optimal=False,
            assignment_count=timed_candidate.assignment_count, split_bay_count=timed_candidate.split_bay_count,
            load_deviation=timed_candidate.load_deviation, reversal_count=timed_candidate.reversal_count,
            movement_count=timed_candidate.completion_movement_count,
            crane_loads=timed_candidate.loads,
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
        published_key = timed_candidate.objective_key

    # Experimental checkpoint entry: resume exactly at the boundary before
    # the critical-window phase.  The checkpoint is an already validated
    # complete candidate and is intentionally kept private to the ablation
    # harness; normal construction remains unchanged when it is absent.
    if checkpoint is not None:
        checkpoint = _normalize_completed_candidate(checkpoint)
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
            if best is not None and best.completion_time == lower_bound:
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
            and best.completion_time == lower_bound
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
        previous_best_completion = best.completion_time if best is not None else max_steps
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
        candidate = _normalize_completed_candidate(candidate)
        completed += 1
        operator_calls["construction"] += 1
        remember_elite(candidate)
        gap = max(0, candidate.completion_time - lower_bound)
        reward = (
            1.0 / (1 + gap)
            + 0.25 * max(0, previous_best_completion - candidate.completion_time)
            + 0.002 / (1 + candidate.completion_movement_count)
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
        best = _normalize_completed_candidate(dp_incumbent)
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
    while best.completion_time > lower_bound and time.perf_counter() < general_repair_deadline:
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
        if improved is not None:
            improved = _normalize_completed_candidate(improved)
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
        and critical_mode in {"trajectory", "both", "local_only"}
        and best.completion_time > lower_bound
        and time.perf_counter() < repair_phase_deadline
    ):
        cumulative_deadline = repair_phase_deadline
        if critical_mode == "both":
            cumulative_deadline = time.perf_counter() + 0.65 * (
                repair_phase_deadline - time.perf_counter()
            )
        operator_calls["trajectory"] += 1
        cumulative, evaluated, prepared, first_feasible = _cumulative_local_trajectory_repair_iterative(
            W, M, starts, best, cumulative_deadline, seed,
            move_time=move_time,
            enable_cross_crane_phase_relay=True,
            enable_idle_capacity_rebalance=True,
            local_windows_only=(critical_mode == "local_only"),
        )
        critical_repair_iterations += evaluated
        if prepared is not best:
            elite_repairs += 1
            remember_elite(prepared)
        if cumulative is not None:
            cumulative = _normalize_completed_candidate(cumulative)
            elite_repairs += 1
            remember_elite(cumulative)
            if cumulative.objective_key < best.objective_key:
                best = cumulative
                critical_repair_improvements += 1
                refresh_elite_pool()
                publish(best)
    while (
        not stop_before_critical
        and critical_mode in {"beam", "trajectory", "both", "local_only"}
        and
        best.completion_time > lower_bound
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
        if critical_mode in {"beam", "local_only"} or (
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
            improved = _normalize_completed_candidate(improved)
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
    gap_after_construction = best.completion_time - lower_bound
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
        and best.completion_time > lower_bound
        and time.perf_counter() < mcts_deadline
    ):
        operator_calls["mcts"] += 1
        improved, evaluated = _mcts_horizon_search(
            W, M, starts, eligibility, bay_criticality, best,
            mcts_deadline, seed + 97,
        )
        mcts_iterations += evaluated
        if improved is not None:
            improved = _normalize_completed_candidate(improved)
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
            or best.completion_time <= lower_bound
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
        if improved is not None:
            improved = _normalize_completed_candidate(improved)
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
        best.completion_time - lower_bound <= 1
        and len(configurations) <= 20_000
        and math.comb(len(W) - M + 1, M) <= 20_000
    )
    exact_started = time.perf_counter()
    while (
        use_exact_search
        and exact_is_promising
        and best.completion_time > lower_bound
        and time.perf_counter() < improvement_deadline
    ):
        operator_calls["exact"] += 1
        improved, exact_status, evaluated = _exact_horizon_search(
            W, M, starts, configurations, initial_configs, eligibility,
            bay_criticality, best, improvement_deadline,
        )
        exact_search_nodes += evaluated
        if improved is not None:
            improved = _normalize_completed_candidate(improved)
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
    best = _normalize_completed_candidate(
        _retime_candidate(W, M, best, move_time)
    )
    makespan_optimal = (
        best.completion_time == lower_bound or exact_search_proved_optimal
    )
    bay_cranes = {
        i + 1: [q + 1 for q in sorted(bay_owners)]
        for i, bay_owners in enumerate(best.owners)
        if bay_owners
    }
    return Solution(
        status=(
            "COMPLETION_TIME_OPTIMAL_BY_LOWER_BOUND"
            if makespan_optimal and best.completion_time == lower_bound
            else "COMPLETION_TIME_OPTIMAL_BY_EXACT_SEARCH"
            if exact_search_proved_optimal
            else "HEURISTIC_FEASIBLE"
        ),
        method=(
            "dp_dispatch_priority_construction_only_no_solver"
            if stop_before_critical
            else "dp_dispatch_priority_evolution_trajectory_repair_mcts_exact_no_solver"
        ),
        makespan=best.completion_time,
        schedule_horizon=best.schedule_horizon,
        schedule_movement_count=best.movement_count,
        makespan_lower_bound=lower_bound,
        lower_bound_components=lower_bound_components,
        makespan_proven_optimal=makespan_optimal,
        proven_lexicographic_optimal=False,
        assignment_count=best.assignment_count,
        split_bay_count=best.split_bay_count,
        load_deviation=best.load_deviation,
        reversal_count=best.reversal_count,
        movement_count=best.completion_movement_count,
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
    schedule_horizon = getattr(solution, "schedule_horizon", None)
    if schedule_horizon is None:
        schedule_horizon = solution.makespan

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
    actual_completion = _work_completion_time(solution.slots)
    if actual_completion != solution.makespan:
        raise AssertionError(
            f"报告完工时间 {solution.makespan} 与最后工作边界 {actual_completion} 不一致。"
        )
    if len(by_time) != schedule_horizon:
        raise AssertionError("时间槽数量与保存的排程时域不一致。")

    for t in range(schedule_horizon):
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
        completed_moves = 0
        for t in range(1, schedule_horizon):
            previous = sorted(by_time[t - 1], key=lambda row: row.crane)
            current = sorted(by_time[t], key=lambda row: row.crane)
            for q, (before, after) in enumerate(zip(previous, current)):
                if before.end_bay != after.start_bay:
                    visible_moves += 1
                    directions[q].append(1 if after.start_bay > before.end_bay else -1)
                    if t < solution.makespan:
                        completed_moves += 1
        reversals = sum(
            a != b
            for crane_directions in directions
            for a, b in zip(crane_directions, crane_directions[1:])
        )
        if completed_moves != solution.movement_count:
            raise AssertionError("完工前瞬时移动次数汇总不一致。")
        expected_schedule_moves = getattr(
            solution, "schedule_movement_count", None
        )
        if expected_schedule_moves is None:
            expected_schedule_moves = solution.movement_count
        if visible_moves != expected_schedule_moves:
            raise AssertionError("完整排程时域的瞬时移动次数汇总不一致。")
    else:
        reversals = _count_reversals(solution.slots, M)
        if move_time == 1:
            movement_count = sum(
                slot.state == "move" and slot.time < solution.makespan
                for slot in solution.slots
            )
            schedule_movement_count = sum(
                slot.state == "move" for slot in solution.slots
            )
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
            movement_count = sum(
                min(slot.time for slot in group) < solution.makespan
                for group in groups.values()
            )
            schedule_movement_count = len(groups)
        if movement_count != solution.movement_count:
            raise AssertionError("完工前移动事件汇总不一致。")
        expected_schedule_moves = getattr(
            solution, "schedule_movement_count", None
        )
        if expected_schedule_moves is not None and schedule_movement_count != expected_schedule_moves:
            raise AssertionError("完整排程时域的移动事件汇总不一致。")
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
    proof_text = "completion-time optimal" if solution.makespan_proven_optimal else "heuristic"
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
    if solution.status == "COMPLETION_TIME_OPTIMAL_BY_LOWER_BOUND":
        proof = "达到有效下界，完工时间已证明最优；移动次数未证明最优"
    elif solution.status == "COMPLETION_TIME_OPTIMAL_BY_EXACT_SEARCH":
        proof = "精确搜索已排除更短完工时间；移动次数未证明最优"
    else:
        proof = "当前最好可行解，未证明全局最优"
    print(f"状态: {solution.status}（{proof}）")
    print(f"方法: {solution.method}")
    horizon = solution.schedule_horizon or solution.makespan
    print(
        f"实际完工时间 C: {solution.makespan}；保存时域 H: {horizon}；"
        f"完工后空槽: {max(0, horizon - solution.makespan)}；"
        f"完工时间下界: {solution.makespan_lower_bound}"
    )
    print(f"下界组成: {solution.lower_bound_components}")
    print("正式目标: (实际完工时间 C, 完工前移动次数 K)，严格词典序最小化")
    print(f"桥吊-贝位分配数（统计）: {solution.assignment_count}")
    print(f"被多吊拆分的贝位数: {solution.split_bay_count}")
    print(f"各桥吊作业负荷: {solution.crane_loads}")
    print(f"中间重载目标权重: {solution.target_weights}")
    print(f"中间重载偏差: {solution.load_deviation}")
    print(f"折返次数（统计）: {solution.reversal_count}")
    print(
        f"完工前移动次数 K: {solution.movement_count}；"
        f"完整保存时域移动次数: {solution.schedule_movement_count}"
    )
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
        choices=(
            "off_reallocate", "off_reserved", "beam", "trajectory", "both",
            "local_only",
        ),
        default="local_only",
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
