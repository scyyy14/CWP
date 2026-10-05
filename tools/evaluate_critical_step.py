"""Reproducible experiment harness for Step 8 local repair operators.

The harness treats a complete source schedule as an immutable artifact.  The
direct experiment can run the bounded legacy operators together with the
atomic cross-crane phase-relay operator; the full experiment uses the solver's
critical_mode switch.  It never turns a timeout into an infeasibility claim
and writes one JSON record per run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cwp_solver as solver


INSTANCES = {
    "current20x6": (ROOT / "example_input.json", ROOT / "output" / "schedule.json"),
    "historical39x9": (
        ROOT / "experiments" / "local_block_repair_20260914" / "historical39x9_input.json",
        ROOT / "experiments" / "large39" / "schedule.json",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def objective(candidate: solver._CandidateSchedule) -> list[int]:
    return list(candidate.objective_key)


def source_candidate(W, M, S, path: Path, move_time=1, allow_edge_exit=False):
    data = json.loads(path.read_text(encoding="utf-8"))
    # Older polished experiment records omitted move_time.  In that case the
    # caller's instance configuration is the only unambiguous default.
    source_move_time = data.get("move_time", move_time)
    if source_move_time != move_time:
        raise ValueError(
            f"固定源 move_time={source_move_time} 与输入 move_time={move_time} 不一致；"
            "请用相同移动时间生成固定源。"
        )
    slots = [solver.Slot(**item) for item in data["slots"]]
    schedule_horizon = int(data.get("schedule_horizon", data["makespan"]))
    if allow_edge_exit:
        slots = solver.apply_completed_edge_exits(
            slots, M, len(W), schedule_horizon, move_time
        )
    owners = [set() for _ in W]
    loads = [0] * M
    for slot in slots:
        if slot.state == "work":
            owners[slot.work_bay - 1].add(slot.crane - 1)
            loads[slot.crane - 1] += 1
    makespan = schedule_horizon
    if move_time == 0:
        by_time = {
            t: sorted((slot for slot in slots if slot.time == t), key=lambda slot: slot.crane)
            for t in range(makespan)
        }
        directions = [[] for _ in range(M)]
        movement_count = 0
        for t in range(1, makespan):
            for q, (before, after) in enumerate(zip(by_time[t - 1], by_time[t])):
                if before.end_bay != after.start_bay:
                    movement_count += 1
                    directions[q].append(1 if after.start_bay > before.end_bay else -1)
        reversal_count = sum(
            left != right
            for crane_directions in directions
            for left, right in zip(crane_directions, crane_directions[1:])
        )
    else:
        reversal_count = solver._count_reversals(slots, M)
        movement_count = (
            sum(slot.state == "move" for slot in slots)
            if move_time == 1
            else len({
                (slot.crane, slot.move_id)
                for slot in slots if slot.state == "move"
            })
        )
    weights = [min(q + 1, M - q) for q in range(M)]
    assignment_count = sum(len(item) for item in owners)
    split_bay_count = sum(len(item) > 1 for item in owners)
    load_deviation = solver._load_deviation(loads, weights, sum(W))
    completion_time = solver._work_completion_time(slots)
    objective_movement_count = solver._movement_count_until_completion(
        slots, M, move_time, completion_time
    )
    check = SimpleNamespace(
        slots=slots, makespan=completion_time, schedule_horizon=makespan,
        crane_loads=loads, reversal_count=reversal_count,
        movement_count=objective_movement_count,
        schedule_movement_count=movement_count, move_time=move_time,
    )
    solver.verify_solution(W, M, S, check)
    candidate = solver._CandidateSchedule(
        slots=slots, makespan=makespan,
        assignment_count=assignment_count,
        split_bay_count=split_bay_count,
        load_deviation=load_deviation,
        reversal_count=reversal_count,
        movement_count=movement_count,
        loads=loads, owners=owners, move_time=move_time,
    )
    normalized = solver._normalize_completed_candidate(candidate)
    normalized.normalization_source_horizon = makespan
    normalized.normalization_trimmed_slots = max(
        0, makespan - normalized.schedule_horizon
    )
    return normalized


def summarize(candidate, starts=None):
    continuity = solver._continuity_diagnostics(
        candidate, len(candidate.loads), starts
    )
    long_revisit_threshold = 8
    fragmentation_details = [
        {
            "bay": int(bay),
            "blocks": continuity["work_blocks_by_bay"].get(bay, []),
            "gaps": gaps,
        }
        for bay, gaps in continuity["bay_gaps_by_bay"].items()
        if gaps
    ]
    return {
        "objective": objective(candidate),
        "makespan": candidate.completion_time,
        "schedule_horizon": candidate.schedule_horizon,
        "normalization_source_horizon": (
            candidate.normalization_source_horizon
            if candidate.normalization_source_horizon is not None
            else candidate.schedule_horizon
        ),
        "normalization_trimmed_slots": candidate.normalization_trimmed_slots,
        "trailing_idle_after_work": max(
            0, candidate.schedule_horizon - candidate.completion_time
        ),
        "source_trailing_idle_after_work": max(
            0,
            (candidate.normalization_source_horizon
             if candidate.normalization_source_horizon is not None
             else candidate.schedule_horizon) - candidate.completion_time,
        ),
        "move_time": candidate.move_time,
        "split_bay_count": candidate.split_bay_count,
        "load_deviation": candidate.load_deviation,
        "movement_count": candidate.completion_movement_count,
        "schedule_movement_count": candidate.movement_count,
        "smoothness": list(solver._trajectory_smoothness(
            candidate, len(candidate.loads)
        )),
        "assignment_count": candidate.assignment_count,
        "reversal_count": candidate.reversal_count,
        "continuity": continuity,
        "execution_key": list(solver._execution_rank(
            candidate, len(candidate.loads)
        )),
        "execution_warning": (
            continuity["max_work_revisit_gap"] >= long_revisit_threshold
        ),
        "execution_warning_reason": (
            f"max_work_revisit_gap>={long_revisit_threshold}"
            if continuity["max_work_revisit_gap"] >= long_revisit_threshold
            else None
        ),
        "work_revisit_details": continuity["crane_work_revisits"],
        "fragmentation_details": fragmentation_details,
        "idle": solver._idle_diagnostics(candidate, len(candidate.loads)),
        "operational_rank": list(solver._operational_rank(
            candidate, len(candidate.loads)
        )),
    }


def slot_dicts(candidate):
    return [
        {"time": s.time, "crane": s.crane, "state": s.state,
         "start_bay": s.start_bay, "end_bay": s.end_bay,
         "work_bay": s.work_bay, "move_id": s.move_id,
         "move_step": s.move_step, "move_steps": s.move_steps}
        for s in candidate.slots
    ]


def verify_candidate(W, M, S, candidate, move_time=1):
    """Run the project's independent verifier on a private candidate."""
    owners = {bay: sorted(q + 1 for q in cranes)
              for bay, cranes in enumerate(candidate.owners, 1) if cranes}
    check = SimpleNamespace(
        slots=candidate.slots, makespan=candidate.completion_time,
        schedule_horizon=candidate.schedule_horizon,
        crane_loads=candidate.loads, reversal_count=candidate.reversal_count,
        assignment_count=candidate.assignment_count,
        split_bay_count=candidate.split_bay_count, bay_cranes=owners,
        movement_count=candidate.completion_movement_count,
        schedule_movement_count=candidate.movement_count,
        move_time=move_time,
    )
    solver.verify_solution(W, M, S, check)


def plot_view(candidate):
    """Minimal solution-shaped view accepted by the project plotter."""
    return SimpleNamespace(
        slots=candidate.slots,
        makespan=candidate.completion_time,
        makespan_proven_optimal=False,
        reversal_count=candidate.reversal_count,
        movement_count=candidate.completion_movement_count,
        split_bay_count=candidate.split_bay_count,
        crane_loads=candidate.loads,
        move_time=candidate.move_time,
    )


def direct_run(
    W, M, S, source, mode, budget, seed, *, move_time=1,
    verbose_windows=False, plot_dir: Path | None = None,
    artifact_dir: Path | None = None,
    preserve_horizon=False, cumulative_local=True,
    enable_descent=True, enable_operational_repairs=True,
    enable_work_transfer=False, protect_source_continuity=False,
    enable_fragmentation_repair=False, enable_cyclic_exchange=True,
    enable_phase_resequence=True,
    enable_phase_closure=True,
    enable_cross_crane_phase_relay=True,
    enable_idle_capacity_rebalance=True,
    enable_forced_prefix_consolidation=True,
    enable_global_rebalance=True,
    execution_output=True, local_state_limit=256, execution_pool_sort=True,
    strict_local_transactions=False,
    source_hash=None,
    use_legacy_seed=False,
    enable_multi_relay=True,
    local_windows_only=False,
):
    start = time.perf_counter()
    deadline = start + budget
    # Reserve a small, explicit tail for the optional diagnostic plots.  The
    # search deadline is otherwise extended while rendering, so a nominal
    # 300-second run can exceed its advertised wall-clock budget.
    plot_reserve = 0.0
    if plot_dir is not None:
        plot_reserve = min(15.0, max(2.0, 0.06 * budget))
        deadline = start + max(0.01, budget - plot_reserve)
    windows = solver._critical_repair_windows(source, M)
    limit = max(1, min(48, len(windows)))
    best = None
    calls = 0
    attempts = 0
    updates = []
    window_results = []
    phase_closure_transaction_paths = []
    cross_crane_phase_transaction_paths = []
    source_key = tuple(source.objective_key)
    if (
        cumulative_local and mode == "trajectory"
        and move_time == 0
    ):
        trace = []
        continuity_output = {}
        candidate, evaluated, prepared, first_feasible = solver._cumulative_local_trajectory_repair_iterative(
            W, M, S, source, deadline, seed,
            move_time=move_time, attempt_trace=trace,
            continuity_output=continuity_output,
            enable_descent=enable_descent,
            enable_operational_repairs=enable_operational_repairs,
            preserve_horizon=preserve_horizon,
            enable_work_transfer=enable_work_transfer,
            enable_fragmentation_repair=enable_fragmentation_repair,
            enable_cyclic_exchange=enable_cyclic_exchange,
            enable_phase_resequence=enable_phase_resequence,
            enable_phase_closure=enable_phase_closure,
            enable_cross_crane_phase_relay=enable_cross_crane_phase_relay,
            enable_idle_capacity_rebalance=enable_idle_capacity_rebalance,
            enable_forced_prefix_consolidation=enable_forced_prefix_consolidation,
            local_state_limit=local_state_limit,
            execution_pool_sort=execution_pool_sort,
            protect_source_continuity=protect_source_continuity,
            strict_local_transactions=strict_local_transactions,
            source_hash=source_hash,
            use_legacy_seed=use_legacy_seed,
            enable_multi_relay=enable_multi_relay,
            enable_global_rebalance=enable_global_rebalance,
            local_windows_only=local_windows_only,
        )
        formal_best = continuity_output.get("formal_best", candidate)
        continuity_best = continuity_output.get("continuity_best", candidate)
        operational_best = continuity_output.get("operational_best", candidate)
        execution_best = continuity_output.get("execution_best", operational_best)
        balanced_best = continuity_output.get("balanced_best", execution_best)
        recommended_best = continuity_output.get("recommended_best", balanced_best)
        if formal_best is None:
            formal_best = source
        if continuity_best is None:
            continuity_best = formal_best
        if operational_best is None:
            operational_best = continuity_best
        if execution_best is None:
            execution_best = operational_best
        formal_best = solver._normalize_completed_candidate(formal_best)
        continuity_best = solver._normalize_completed_candidate(continuity_best)
        operational_best = solver._normalize_completed_candidate(operational_best)
        execution_best = solver._normalize_completed_candidate(execution_best)
        balanced_best = solver._normalize_completed_candidate(balanced_best)
        recommended_best = solver._normalize_completed_candidate(recommended_best)
        reported_best = (
            formal_best if candidate is None
            else solver._normalize_completed_candidate(candidate)
        )
        if candidate is not None:
            run_status = "FOUND"
        elif solver._execution_rank(execution_best, M) < solver._execution_rank(source, M):
            run_status = "EXECUTION_IMPROVED"
        elif formal_best.objective_key < source.objective_key:
            run_status = "NO_SHORTENING_SAME_H_IMPROVEMENT"
        else:
            run_status = "TIMEOUT"
        verify_candidate(W, M, S, prepared, move_time)
        for polished_candidate in (
            candidate, formal_best, continuity_best, operational_best,
            execution_best, balanced_best, recommended_best,
        ):
            if polished_candidate is not None:
                verify_candidate(W, M, S, polished_candidate, move_time)
        first_event = next(
            (item for item in trace if item.get("phase") == "first_feasible"),
            None,
        )
        polish_event = next(
            (item for item in reversed(trace)
             if item.get("phase") == "polish_complete"),
            None,
        )
        plot_path = None
        source_path = None
        prepared_path = None
        first_feasible_path = None
        polished_path = None
        formal_best_path = None
        continuity_best_path = None
        operational_best_path = None
        execution_best_path = None
        balanced_best_path = None
        recommended_best_path = None
        paired_transaction_paths = []
        phase_transaction_paths = []
        cross_crane_phase_transaction_paths = []
        forced_prefix_transaction_paths = []
        # Beam is measured as a 300-second search and gets one final artifact
        # below.  Rendering all 48 intermediate beam windows during the search
        # makes plotting part of the algorithm and can consume the reserve.
        if plot_dir is not None and mode != "beam":
            plot_dir.mkdir(parents=True, exist_ok=True)
            source_path = plot_dir / "source.json"
            source_path.write_text(json.dumps({
                **summarize(source),
                "slots": slot_dicts(source),
            }, indent=2), encoding="utf-8")
            source_plot = plot_dir / "source.png"
            solver.plot_schedule(
                plot_view(source), len(W), source_plot, show=False,
                diagnostic_title="Step 8 trajectory — adapted source schedule",
            )
            source_path = str(source_path)
            prepared_path = plot_dir / "cumulative_prepared.json"
            prepared_path.write_text(json.dumps({
                **summarize(prepared),
                "slots": slot_dicts(prepared),
            }, indent=2), encoding="utf-8")
            if first_feasible is not None:
                first_feasible_path = plot_dir / "first_feasible.json"
                first_feasible_path.write_text(json.dumps({
                    **summarize(first_feasible),
                    "slots": slot_dicts(first_feasible),
                }, indent=2), encoding="utf-8")
                first_feasible_plot = plot_dir / "first_feasible.png"
                solver.plot_schedule(
                    plot_view(first_feasible), len(W), first_feasible_plot,
                    show=False,
                    diagnostic_title=(
                        "Step 8 trajectory — first feasible H-1 schedule"
                    ),
                )
                first_feasible_path = str(first_feasible_path)
                plot_path = str(first_feasible_plot)
            if candidate is not None:
                polished_path = plot_dir / "polished_best.json"
                polished_path.write_text(json.dumps({
                    **summarize(candidate),
                    "slots": slot_dicts(candidate),
                }, indent=2), encoding="utf-8")
                polished_plot = plot_dir / "polished_best.png"
                solver.plot_schedule(
                    plot_view(candidate), len(W), polished_plot, show=False,
                    diagnostic_title=(
                        "Step 8 trajectory — fixed-horizon polished H-1 schedule"
                    ),
                )
                polished_path = str(polished_path)
                plot_path = str(polished_plot)
            if formal_best is not None:
                formal_best_path = plot_dir / "formal_best.json"
                formal_best_path.write_text(json.dumps({
                    **summarize(formal_best),
                    "slots": slot_dicts(formal_best),
                }, indent=2), encoding="utf-8")
                formal_best_plot = plot_dir / "formal_best.png"
                solver.plot_schedule(
                    plot_view(formal_best), len(W), formal_best_plot,
                    show=False,
                    diagnostic_title=(
                        "Step 8 trajectory — formal best H-1 schedule"
                    ),
                )
                formal_best_path = str(formal_best_path)
            if continuity_best is not None:
                continuity_best_path = plot_dir / "continuity_best.json"
                continuity_best_path.write_text(json.dumps({
                    **summarize(continuity_best),
                    "slots": slot_dicts(continuity_best),
                }, indent=2), encoding="utf-8")
                continuity_best_plot = plot_dir / "continuity_best.png"
                solver.plot_schedule(
                    plot_view(continuity_best), len(W), continuity_best_plot,
                    show=False,
                    diagnostic_title=(
                        "Step 8 trajectory — continuity best H-1 schedule"
                    ),
                )
                continuity_best_path = str(continuity_best_path)
            if operational_best is not None:
                operational_best_path = plot_dir / "operational_best.json"
                operational_best_path.write_text(json.dumps({
                    **summarize(operational_best),
                    "slots": slot_dicts(operational_best),
                }, indent=2), encoding="utf-8")
                operational_best_plot = plot_dir / "operational_best.png"
                solver.plot_schedule(
                    plot_view(operational_best), len(W), operational_best_plot,
                    show=False,
                    diagnostic_title=(
                        "Step 8 trajectory — operational best schedule"
                    ),
                )
                operational_best_path = str(operational_best_path)
            if execution_output and execution_best is not None:
                execution_best_path = plot_dir / "execution_best.json"
                execution_best_path.write_text(json.dumps({
                    **summarize(execution_best),
                    "slots": slot_dicts(execution_best),
                }, indent=2), encoding="utf-8")
                execution_best_plot = plot_dir / "execution_best.png"
                solver.plot_schedule(
                    plot_view(execution_best), len(W), execution_best_plot,
                    show=False,
                    diagnostic_title=(
                        "Step 8 trajectory — execution best schedule"
                    ),
                )
                execution_best_path = str(execution_best_path)
                plot_path = execution_best_plot
            if balanced_best is not None:
                balanced_best_path = plot_dir / "balanced_best.json"
                balanced_best_path.write_text(json.dumps({
                    **summarize(balanced_best),
                    "balance": solver._balanced_schedule_metrics(balanced_best, M),
                    "slots": slot_dicts(balanced_best),
                }, indent=2), encoding="utf-8")
                balanced_best_plot = plot_dir / "balanced_best.png"
                solver.plot_schedule(
                    plot_view(balanced_best), len(W), balanced_best_plot,
                    show=False,
                    diagnostic_title=(
                        "Step 8 trajectory — balanced idle-capacity alternative"
                    ),
                )
                balanced_best_path = str(balanced_best_path)
            if recommended_best is not None:
                recommended_best_path = artifact_dir / "recommended_best.json"
                recommended_best_path.write_text(json.dumps({
                    **summarize(recommended_best, S),
                    "balance": solver._balanced_schedule_metrics(
                        recommended_best, M
                    ),
                    "recommended_rank": list(
                        solver._recommended_schedule_rank(recommended_best, M)
                    ),
                    "slots": slot_dicts(recommended_best),
                }, indent=2), encoding="utf-8")
                recommended_best_path = str(recommended_best_path)
            for transaction_index, transaction in enumerate(
                continuity_output.get("paired_transactions", []), start=1
            ):
                transaction_dir = plot_dir / "paired_transactions"
                transaction_dir.mkdir(parents=True, exist_ok=True)
                before = transaction["before"]
                after = transaction["after"]
                details = transaction.get("details", {})
                stem = f"paired_{transaction_index:02d}"
                before_json = transaction_dir / f"{stem}_before.json"
                after_json = transaction_dir / f"{stem}_after.json"
                before_json.write_text(json.dumps({
                    **summarize(before),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(before),
                }, indent=2), encoding="utf-8")
                after_json.write_text(json.dumps({
                    **summarize(after),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(after),
                }, indent=2), encoding="utf-8")
                before_png = transaction_dir / f"{stem}_before.png"
                after_png = transaction_dir / f"{stem}_after.png"
                interval_text = ", ".join(
                    f"Q{region['crane']}:[{region['start']},{region['end_exclusive']})"
                    for region in transaction.get("regions", [])
                )
                solver.plot_schedule(
                    plot_view(before), len(W), before_png, show=False,
                    diagnostic_title=(
                        f"Step 8 paired exchange {transaction_index} — before; "
                        f"{interval_text}"
                    ),
                )
                solver.plot_schedule(
                    plot_view(after), len(W), after_png, show=False,
                    diagnostic_title=(
                        f"Step 8 paired exchange {transaction_index} — after; "
                        f"{interval_text}"
                    ),
                )
                paired_transaction_paths.append({
                    "details": details,
                    "regions": transaction.get("regions", []),
                    "before_json": str(before_json),
                    "before_png": str(before_png),
                    "after_json": str(after_json),
                    "after_png": str(after_png),
                })
            for transaction_index, transaction in enumerate(
                continuity_output.get("phase_transactions", []), start=1
            ):
                transaction_dir = plot_dir / "phase_transactions"
                transaction_dir.mkdir(parents=True, exist_ok=True)
                before = transaction["before"]
                after = transaction["after"]
                details = transaction.get("details", {})
                stem = f"phase_{transaction_index:02d}"
                before_json = transaction_dir / f"{stem}_before.json"
                after_json = transaction_dir / f"{stem}_after.json"
                before_json.write_text(json.dumps({
                    **summarize(before),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(before),
                }, indent=2), encoding="utf-8")
                after_json.write_text(json.dumps({
                    **summarize(after),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(after),
                }, indent=2), encoding="utf-8")
                before_png = transaction_dir / f"{stem}_before.png"
                after_png = transaction_dir / f"{stem}_after.png"
                interval_text = ", ".join(
                    f"Q{region['crane']}:[{region['start']},{region['end_exclusive']})"
                    for region in transaction.get("regions", [])
                )
                solver.plot_schedule(
                    plot_view(before), len(W), before_png, show=False,
                    diagnostic_title=(
                        f"Step 8 phase resequence {transaction_index} — before; "
                        f"{interval_text}"
                    ),
                )
                solver.plot_schedule(
                    plot_view(after), len(W), after_png, show=False,
                    diagnostic_title=(
                        f"Step 8 phase resequence {transaction_index} — after; "
                        f"{interval_text}"
                    ),
                )
                phase_transaction_paths.append({
                    "details": details,
                    "regions": transaction.get("regions", []),
                    "before_json": str(before_json),
                    "before_png": str(before_png),
                    "after_json": str(after_json),
                    "after_png": str(after_png),
                })
            for transaction_index, transaction in enumerate(
                continuity_output.get("cross_crane_phase_transactions", []), start=1
            ):
                transaction_dir = plot_dir / "cross_crane_phase_transactions"
                transaction_dir.mkdir(parents=True, exist_ok=True)
                before = transaction["before"]
                after = transaction["after"]
                details = transaction.get("details", {})
                stem = f"cross_relay_{transaction_index:02d}"
                before_json = transaction_dir / f"{stem}_before.json"
                after_json = transaction_dir / f"{stem}_after.json"
                before_json.write_text(json.dumps({
                    **summarize(before),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(before),
                }, indent=2), encoding="utf-8")
                after_json.write_text(json.dumps({
                    **summarize(after),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(after),
                }, indent=2), encoding="utf-8")
                before_png = transaction_dir / f"{stem}_before.png"
                after_png = transaction_dir / f"{stem}_after.png"
                interval_text = ", ".join(
                    f"Q{region['crane']}:[{region['start']},{region['end_exclusive']})"
                    for region in transaction.get("regions", [])
                )
                solver.plot_schedule(
                    plot_view(before), len(W), before_png, show=False,
                    diagnostic_title=(
                        f"Step 8 cross-crane phase relay {transaction_index} — before; "
                        f"{interval_text}"
                    ),
                )
                solver.plot_schedule(
                    plot_view(after), len(W), after_png, show=False,
                    diagnostic_title=(
                        f"Step 8 cross-crane phase relay {transaction_index} — after; "
                        f"{interval_text}"
                    ),
                )
                cross_crane_phase_transaction_paths.append({
                    "details": details,
                    "regions": transaction.get("regions", []),
                    "before_json": str(before_json),
                    "before_png": str(before_png),
                    "after_json": str(after_json),
                    "after_png": str(after_png),
                })
            if candidate is None:
                plot_path = plot_dir / "cumulative_prepared.png"
                solver.plot_schedule(
                    plot_view(prepared), len(W), plot_path, show=False,
                    diagnostic_title=(
                        "Step 8 cumulative local trajectory — "
                        "prepared H schedule; no legal shortening"
                    ),
                )
                plot_path = str(plot_path)
        elif artifact_dir is not None and execution_output:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            source_path = artifact_dir / "source.json"
            source_path.write_text(json.dumps({
                **summarize(source, S), "slots": slot_dicts(source),
            }, indent=2), encoding="utf-8")
            source_path = str(source_path)
            if formal_best is not None:
                formal_best_path = artifact_dir / "formal_best.json"
                formal_best_path.write_text(json.dumps({
                    **summarize(formal_best, S), "slots": slot_dicts(formal_best),
                }, indent=2), encoding="utf-8")
                formal_best_path = str(formal_best_path)
            if execution_best is not None:
                execution_best_path = artifact_dir / "execution_best.json"
                execution_best_path.write_text(json.dumps({
                    **summarize(execution_best, S), "slots": slot_dicts(execution_best),
                }, indent=2), encoding="utf-8")
                execution_best_path = str(execution_best_path)
            if balanced_best is not None:
                balanced_best_path = artifact_dir / "balanced_best.json"
                balanced_best_path.write_text(json.dumps({
                    **summarize(balanced_best, S),
                    "balance": solver._balanced_schedule_metrics(balanced_best, M),
                    "slots": slot_dicts(balanced_best),
                }, indent=2), encoding="utf-8")
                balanced_best_path = str(balanced_best_path)
            if recommended_best is not None:
                recommended_best_path = artifact_dir / "recommended_best.json"
                recommended_best_path.write_text(json.dumps({
                    **summarize(recommended_best, S),
                    "balance": solver._balanced_schedule_metrics(
                        recommended_best, M
                    ),
                    "recommended_rank": list(
                        solver._recommended_schedule_rank(recommended_best, M)
                    ),
                    "slots": slot_dicts(recommended_best),
                }, indent=2), encoding="utf-8")
                recommended_best_path = str(recommended_best_path)
            for transaction_index, transaction in enumerate(
                continuity_output.get("paired_transactions", []), start=1
            ):
                transaction_dir = artifact_dir / "paired_transactions"
                transaction_dir.mkdir(parents=True, exist_ok=True)
                details = transaction.get("details", {})
                stem = f"paired_{transaction_index:02d}"
                before_json = transaction_dir / f"{stem}_before.json"
                after_json = transaction_dir / f"{stem}_after.json"
                before_json.write_text(json.dumps({
                    **summarize(transaction["before"]),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction["before"]),
                }, indent=2), encoding="utf-8")
                after_json.write_text(json.dumps({
                    **summarize(transaction["after"]),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction["after"]),
                }, indent=2), encoding="utf-8")
                paired_transaction_paths.append({
                    "details": details,
                    "regions": transaction.get("regions", []),
                    "before_json": str(before_json),
                    "after_json": str(after_json),
                })
            for transaction_index, transaction in enumerate(
                continuity_output.get("phase_transactions", []), start=1
            ):
                transaction_dir = artifact_dir / "phase_transactions"
                transaction_dir.mkdir(parents=True, exist_ok=True)
                details = transaction.get("details", {})
                stem = f"phase_{transaction_index:02d}"
                before_json = transaction_dir / f"{stem}_before.json"
                after_json = transaction_dir / f"{stem}_after.json"
                before_json.write_text(json.dumps({
                    **summarize(transaction["before"]),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction["before"]),
                }, indent=2), encoding="utf-8")
                after_json.write_text(json.dumps({
                    **summarize(transaction["after"]),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction["after"]),
                }, indent=2), encoding="utf-8")
                phase_transaction_paths.append({
                    "details": details,
                    "regions": transaction.get("regions", []),
                    "before_json": str(before_json),
                    "after_json": str(after_json),
                })
            for transaction_index, transaction in enumerate(
                continuity_output.get("cross_crane_phase_transactions", []), start=1
            ):
                transaction_dir = artifact_dir / "cross_crane_phase_transactions"
                transaction_dir.mkdir(parents=True, exist_ok=True)
                details = transaction.get("details", {})
                stem = f"cross_relay_{transaction_index:02d}"
                before_json = transaction_dir / f"{stem}_before.json"
                after_json = transaction_dir / f"{stem}_after.json"
                before_json.write_text(json.dumps({
                    **summarize(transaction["before"], S),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction["before"]),
                }, indent=2), encoding="utf-8")
                after_json.write_text(json.dumps({
                    **summarize(transaction["after"], S),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction["after"]),
                }, indent=2), encoding="utf-8")
                cross_crane_phase_transaction_paths.append({
                    "details": details,
                    "regions": transaction.get("regions", []),
                    "before_json": str(before_json),
                    "after_json": str(after_json),
                })
            for transaction_index, transaction in enumerate(
                continuity_output.get("phase_closure_transactions", []), start=1
            ):
                transaction_dir = artifact_dir / "phase_closure_transactions"
                transaction_dir.mkdir(parents=True, exist_ok=True)
                details = transaction.get("details", {})
                stem = f"closure_{transaction_index:02d}"
                before_json = transaction_dir / f"{stem}_before.json"
                after_json = transaction_dir / f"{stem}_after.json"
                for state_name, path in (("before", before_json), ("after", after_json)):
                    candidate_state = transaction[state_name]
                    path.write_text(json.dumps({
                        **summarize(candidate_state, S),
                        "transaction": details,
                        "regions": transaction.get("regions", []),
                        "slots": slot_dicts(candidate_state),
                    }, indent=2), encoding="utf-8")
                phase_closure_transaction_paths.append({
                    "details": details,
                    "regions": transaction.get("regions", []),
                    "before_json": str(before_json),
                    "after_json": str(after_json),
                })
            for transaction_index, transaction in enumerate(
                continuity_output.get("forced_prefix_transactions", []), start=1
            ):
                transaction_dir = artifact_dir / "forced_prefix_transactions"
                transaction_dir.mkdir(parents=True, exist_ok=True)
                details = transaction.get("details", {})
                stem = f"forced_prefix_{transaction_index:02d}"
                before_json = transaction_dir / f"{stem}_before.json"
                after_json = transaction_dir / f"{stem}_after.json"
                before_json.write_text(json.dumps({
                    **summarize(transaction["before"], S),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction["before"]),
                }, indent=2), encoding="utf-8")
                after_json.write_text(json.dumps({
                    **summarize(transaction["after"], S),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction["after"]),
                }, indent=2), encoding="utf-8")
                forced_prefix_transaction_paths.append({
                    "details": details,
                    "regions": transaction.get("regions", []),
                    "before_json": str(before_json),
                    "after_json": str(after_json),
                })
        prepared_potential, remove_at, deficits = solver._shortening_potential(
            W, M, prepared
        )
        updates = []
        if candidate is not None:
            updates.append({
                "chain": None, "window": None, "remove_at": remove_at,
                "evaluated": evaluated, "attempt_trace": trace,
                "candidate": summarize(candidate),
                "strict_source_improvement": candidate.objective_key < source_key,
                "slots": slot_dicts(candidate),
                "first_feasible": (
                    summarize(first_feasible) if first_feasible is not None else None
                ),
            })
        if verbose_windows:
            print(
                f"[trajectory-cumulative] calls={len(trace)} evaluated={evaluated} "
                f"result={'FOUND' if candidate is not None else 'NO_CANDIDATE'} "
                f"prepared_potential={list(prepared_potential)} "
                f"deficit_bays={[bay for bay, amount in enumerate(deficits, 1) if amount]}",
                flush=True,
            )
        return {
            "experiment": "direct", "mode": mode, "move_time": move_time,
            "target_mode": "same_horizon" if preserve_horizon else "shorten",
            "local_strategy": "iterative_cumulative",
            "budget_seconds": budget, "seed": seed,
            "search_budget_seconds": round(deadline - start, 6),
            "plot_reserve_seconds": round(plot_reserve, 6),
            "source": summarize(source, S), "source_objective": list(source_key),
            "calls": len(trace), "evaluated": evaluated,
            "status": run_status,
            "best": summarize(reported_best) if reported_best is not None else None,
            "best_objective": (
                list(reported_best.objective_key) if reported_best is not None else None
            ),
            "best_smoothness": (
                list(solver._trajectory_smoothness(candidate, M))
                if candidate is not None else None
            ),
            "formal_best": summarize(formal_best, S) if formal_best is not None else None,
            "continuity_best": summarize(continuity_best, S) if continuity_best is not None else None,
            "operational_best": summarize(operational_best, S) if operational_best is not None else None,
            "execution_best": summarize(execution_best, S) if execution_best is not None else None,
            "balanced_best": summarize(balanced_best, S) if balanced_best is not None else None,
            "balanced_best_metrics": continuity_output.get("balanced_best_metrics"),
            "recommended_best": (
                summarize(recommended_best, S)
                if recommended_best is not None else None
            ),
            "recommended_best_metrics": continuity_output.get(
                "recommended_best_metrics"
            ),
            "synchronization_stats": continuity_output.get(
                "synchronization_stats"
            ),
            "contiguous_phase_stats": continuity_output.get(
                "contiguous_phase_stats"
            ),
            "local_terminal_alignment": continuity_output.get(
                "local_terminal_alignment"
            ),
            "compression_best": (
                summarize(continuity_output["compression_best"])
                if continuity_output.get("compression_best") is not None else None
            ),
            "execution_warning": (
                summarize(execution_best, S)["execution_warning"]
                if execution_best is not None else True
            ),
            "continuity_operator_stats": continuity_output.get("stats"),
            "idle_capacity_rebalance": (
                (continuity_output.get("stats") or {}).get("idle_capacity_rebalance")
            ),
            "global_rebalance": continuity_output.get("global_rebalance"),
            "continuity_stop_reason": continuity_output.get("stop_reason"),
            "descent_history": continuity_output.get("descent_history", []),
            "first_feasible_by_h": [
                {
                    "completion_time": item["horizon"],
                    "horizon": item["horizon"],
                    "candidate": summarize(item["candidate"]),
                }
                for item in continuity_output.get("first_feasible_by_h", [])
            ],
            "safe_workload_lower_bound": continuity_output.get("safe_workload_lower_bound"),
            "descent_rounds": continuity_output.get("descent_rounds", 0),
            "enable_descent": enable_descent,
            "enable_operational_repairs": enable_operational_repairs,
            "enable_work_transfer": enable_work_transfer,
            "enable_fragmentation_repair": enable_fragmentation_repair,
            "enable_cyclic_exchange": enable_cyclic_exchange,
            "enable_phase_resequence": enable_phase_resequence,
            "enable_phase_closure": enable_phase_closure,
            "enable_cross_crane_phase_relay": enable_cross_crane_phase_relay,
            "enable_idle_capacity_rebalance": enable_idle_capacity_rebalance,
            "enable_forced_prefix_consolidation": enable_forced_prefix_consolidation,
            "enable_global_rebalance": enable_global_rebalance,
            "local_windows_only": local_windows_only,
            "execution_output": execution_output,
            "local_state_limit": local_state_limit,
            "execution_pool_sort": execution_pool_sort,
            "protect_source_continuity": protect_source_continuity,
            "strict_local_transactions": strict_local_transactions,
            "source_hash": source_hash,
            "use_legacy_seed": use_legacy_seed,
            "enable_multi_relay": enable_multi_relay,
            "first_feasible": (
                summarize(first_feasible) if first_feasible is not None else None
            ),
            "time_to_first_feasible": (
                first_event.get("time_to_first_feasible")
                if first_event is not None else None
            ),
            "first_feasible_objective": (
                first_event.get("objective") if first_event is not None else None
            ),
            "first_feasible_smoothness": (
                first_event.get("smoothness") if first_event is not None else None
            ),
            "polish_seconds": (
                polish_event.get("polish_seconds")
                if polish_event is not None else 0.0
            ),
            "polish_attempts": (
                polish_event.get("polish_attempts")
                if polish_event is not None else 0
            ),
            "polish_complete_candidates": (
                polish_event.get("polish_complete_candidates")
                if polish_event is not None else 0
            ),
            "polish_legal_improvements": (
                polish_event.get("polish_legal_improvements")
                if polish_event is not None else 0
            ),
            "prepared": summarize(prepared),
            "prepared_potential": list(prepared_potential),
            "best_remove_at": remove_at,
            "deficit_bays": [
                bay for bay, amount in enumerate(deficits, 1) if amount
            ],
            "window_results": trace,
            "strict_improvements": updates,
            "candidate_updates": updates,
            "plot_path": str(plot_path) if plot_path is not None else None,
            "prepared_path": str(prepared_path) if prepared_path is not None else None,
            "source_path": source_path,
            "first_feasible_path": first_feasible_path,
            "polished_path": polished_path,
            "formal_best_path": formal_best_path,
            "continuity_best_path": continuity_best_path,
            "operational_best_path": operational_best_path,
            "execution_best_path": execution_best_path,
            "balanced_best_path": balanced_best_path,
            "recommended_best_path": recommended_best_path,
            "paired_transactions": paired_transaction_paths,
            "phase_transactions": phase_transaction_paths,
            "cross_crane_phase_transactions": cross_crane_phase_transaction_paths,
            "idle_capacity_transactions": [
                {
                    "details": item.get("details", {}),
                    "regions": item.get("regions", []),
                    "before": summarize(item["before"], S),
                    "after": summarize(item["after"], S),
                    "balance_before": solver._balanced_schedule_metrics(
                        item["before"], M
                    ),
                    "balance_after": solver._balanced_schedule_metrics(
                        item["after"], M
                    ),
                }
                for item in continuity_output.get("idle_capacity_transactions", [])
            ],
            "forced_prefix_transactions": forced_prefix_transaction_paths,
            "forced_prefix_consolidation": (
                (continuity_output.get("stats") or {}).get("forced_prefix_consolidation")
            ),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }
    beam_polish_reserve = 0.0
    beam_search_deadline = deadline
    if (
        mode == "beam"
        and (enable_work_transfer or enable_fragmentation_repair)
        and move_time == 0
    ):
        beam_polish_reserve = min(15.0, max(1.0, 0.05 * budget))
        beam_search_deadline = max(start + 0.01, deadline - beam_polish_reserve)
    for index in range(limit):
        if time.perf_counter() >= beam_search_deadline:
            break
        chain, window = windows[index]
        window_started = time.perf_counter()
        remaining = beam_search_deadline - time.perf_counter()
        # Divide the complete requested budget across the windows.  A former
        # seven-second cap silently reduced a nominal five-minute experiment
        # to at most 7 * window_count seconds.
        slice_deadline = min(
            beam_search_deadline,
            time.perf_counter()
            + max(0.02, remaining / max(1, limit - index)),
        )
        trace = []
        calls += 1
        if mode == "beam" or (mode == "both" and index % 2 == 1):
            candidate, evaluated = solver._critical_window_beam_repair(
                W, M, S, source, chain, slice_deadline, seed + 1009 + index,
                window=window, attempt_trace=trace, move_time=move_time,
                preserve_horizon=preserve_horizon,
            )
        else:
            candidate, evaluated, _ = solver._trajectory_repair(
                W, M, S, source, slice_deadline, seed + 1009 + index,
                active_cranes=chain, window=window, return_state=True,
                attempt_trace=trace, move_time=move_time,
                preserve_horizon=preserve_horizon,
            )
        attempts += evaluated
        if trace:
            selected = trace[-1]
        else:
            selected = {"remove_at": None, "window": window}
        window_record = {
            "window_index": index,
            "chain": list(chain),
            "cranes": [q + 1 for q in chain],
            "window": list(window),
            "remove_at": selected.get("remove_at"),
            "remove_attempts": [item.get("remove_at") for item in trace],
            "evaluated": evaluated,
            "elapsed_seconds": round(time.perf_counter() - window_started, 6),
            "result": "FOUND" if candidate is not None else "NO_CANDIDATE",
            "candidate": summarize(candidate) if candidate is not None else None,
            "strict_source_improvement": (
                candidate is not None
                and tuple(candidate.objective_key) < source_key
            ),
        }
        window_results.append(window_record)
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_candidate = candidate if candidate is not None else source
            plot_path = plot_dir / (
                f"window_{index + 1:02d}_{window_record['result'].lower()}.png"
            )
            result_note = (
                "modified feasible schedule"
                if candidate is not None
                else "no feasible modification; source schedule retained"
            )
            solver.plot_schedule(
                plot_view(plot_candidate),
                len(W),
                plot_path,
                show=False,
                diagnostic_title=(
                    f"Step 8 {mode}, window {index + 1}/{limit}, "
                    f"target={'same H' if preserve_horizon else 'H-1'}, "
                    f"cranes={window_record['cranes']}, range=[{window[0]},{window[1]}), "
                    f"{result_note}"
                ),
                highlight_window=tuple(window),
            )
            window_record["plot_path"] = str(plot_path)
        if verbose_windows:
            candidate_text = (
                str(window_record["candidate"]["objective"])
                if window_record["candidate"] is not None
                else "None"
            )
            print(
                f"[{mode}] window {index + 1}/{limit} "
                f"cranes={window_record['cranes']} range=[{window[0]},{window[1]}) "
                f"remove_at={window_record['remove_attempts']} "
                f"evaluated={evaluated} elapsed={window_record['elapsed_seconds']:.6f}s "
                f"result={window_record['result']} objective={candidate_text}",
                flush=True,
            )
            if window_record.get("plot_path"):
                print(f"  plot={window_record['plot_path']}", flush=True)
        if candidate is None:
            continue
        verify_candidate(W, M, S, candidate, move_time)
        record = {
            "chain": list(chain), "window": list(window),
            "remove_at": selected.get("remove_at"), "evaluated": evaluated,
            "attempt_trace": trace,
            "candidate": summarize(candidate),
            "strict_source_improvement": list(candidate.objective_key) < list(source_key),
        }
        if record["strict_source_improvement"]:
            record["slots"] = slot_dicts(candidate)
        updates.append(record)
        if best is None or candidate.objective_key < best.objective_key:
            best = candidate
    beam_search_found = best is not None
    beam_polish_details = {}
    if (
        (
            enable_work_transfer or enable_fragmentation_repair
            or enable_phase_closure or enable_idle_capacity_rebalance
        )
        and strict_local_transactions
        and move_time == 0
        and time.perf_counter() < deadline
    ):
        polish_source = best if best is not None else source
        polished, polished_evaluated = solver._refine_same_horizon_trajectory(
            W, M, S, polish_source, deadline, seed + 900_001,
            move_time=0, continuity=True, result_box=beam_polish_details,
            enable_work_transfer=enable_work_transfer,
            enable_fragmentation_repair=enable_fragmentation_repair,
            enable_cyclic_exchange=enable_cyclic_exchange,
            enable_phase_resequence=enable_phase_resequence,
            enable_phase_closure=enable_phase_closure,
            enable_cross_crane_phase_relay=enable_cross_crane_phase_relay,
            enable_idle_capacity_rebalance=enable_idle_capacity_rebalance,
            local_state_limit=local_state_limit,
            protect_source_continuity=protect_source_continuity,
            strict_local_transactions=True,
            source_hash=source_hash,
            enable_multi_relay=enable_multi_relay,
        )
        attempts += polished_evaluated
        verify_candidate(W, M, S, polished, move_time)
        formal = beam_polish_details.get("formal_best", polished)
        if best is None or formal.objective_key < best.objective_key:
            best = formal
        if best is None:
            best = polished
    found_best = beam_search_found or (
        best is not None and best is not source
        and best.objective_key < source.objective_key
    )
    reported_best = best if found_best else source
    formal_best = beam_polish_details.get("formal_best", reported_best)
    continuity_best = beam_polish_details.get("continuity_best", formal_best)
    operational_best = beam_polish_details.get("operational_best", continuity_best)
    execution_best = beam_polish_details.get("execution_best", operational_best)
    for polished_candidate in (formal_best, continuity_best, operational_best, execution_best):
        if polished_candidate is not None:
            verify_candidate(W, M, S, polished_candidate, move_time)
    execution_best_path = None
    phase_transaction_paths = []
    if plot_dir is not None and mode == "beam":
        plot_dir.mkdir(parents=True, exist_ok=True)
        source_json = plot_dir / "source.json"
        source_json.write_text(json.dumps({
            **summarize(source), "slots": slot_dicts(source),
        }, indent=2), encoding="utf-8")
        solver.plot_schedule(
            plot_view(source), len(W), plot_dir / "source.png", show=False,
            diagnostic_title="Step 8 beam — adapted source schedule",
        )
        best_json = plot_dir / "best.json"
        best_json.write_text(json.dumps({
            **summarize(reported_best), "slots": slot_dicts(reported_best),
        }, indent=2), encoding="utf-8")
        solver.plot_schedule(
            plot_view(reported_best), len(W), plot_dir / "best.png", show=False,
            diagnostic_title=(
                "Step 8 beam — best feasible schedule"
                if found_best else
                "Step 8 beam — no feasible modification; source retained"
            ),
        )
        if execution_output and execution_best is not None:
            execution_best_path = plot_dir / "execution_best.json"
            execution_best_path.write_text(json.dumps({
                **summarize(execution_best), "slots": slot_dicts(execution_best),
            }, indent=2), encoding="utf-8")
            solver.plot_schedule(
                plot_view(execution_best), len(W),
                plot_dir / "execution_best.png", show=False,
                diagnostic_title="Step 8 beam — execution best schedule",
            )
            execution_best_path = str(execution_best_path)
        for transaction_index, transaction in enumerate(
            beam_polish_details.get("phase_transactions", []), start=1
        ):
            transaction_dir = plot_dir / "phase_transactions"
            transaction_dir.mkdir(parents=True, exist_ok=True)
            before = transaction["before"]
            after = transaction["after"]
            details = transaction.get("details", {})
            stem = f"phase_{transaction_index:02d}"
            before_json = transaction_dir / f"{stem}_before.json"
            after_json = transaction_dir / f"{stem}_after.json"
            before_json.write_text(json.dumps({
                **summarize(before),
                "transaction": details,
                "regions": transaction.get("regions", []),
                "slots": slot_dicts(before),
            }, indent=2), encoding="utf-8")
            after_json.write_text(json.dumps({
                **summarize(after),
                "transaction": details,
                "regions": transaction.get("regions", []),
                "slots": slot_dicts(after),
            }, indent=2), encoding="utf-8")
            before_png = transaction_dir / f"{stem}_before.png"
            after_png = transaction_dir / f"{stem}_after.png"
            interval_text = ", ".join(
                f"Q{region['crane']}:[{region['start']},{region['end_exclusive']})"
                for region in transaction.get("regions", [])
            )
            solver.plot_schedule(
                plot_view(before), len(W), before_png, show=False,
                diagnostic_title=(
                    f"Step 8 beam phase resequence {transaction_index} — before; "
                    f"{interval_text}"
                ),
            )
            solver.plot_schedule(
                plot_view(after), len(W), after_png, show=False,
                diagnostic_title=(
                    f"Step 8 beam phase resequence {transaction_index} — after; "
                    f"{interval_text}"
                ),
            )
            phase_transaction_paths.append({
                "details": details,
                "regions": transaction.get("regions", []),
                "before_json": str(before_json),
                "before_png": str(before_png),
                "after_json": str(after_json),
                "after_png": str(after_png),
            })
    elif artifact_dir is not None and execution_output:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "source.json").write_text(json.dumps({
            **summarize(source), "slots": slot_dicts(source),
        }, indent=2), encoding="utf-8")
        if formal_best is not None:
            (artifact_dir / "formal_best.json").write_text(json.dumps({
                **summarize(formal_best), "slots": slot_dicts(formal_best),
            }, indent=2), encoding="utf-8")
        if execution_best is not None:
            execution_best_path = artifact_dir / "execution_best.json"
            execution_best_path.write_text(json.dumps({
                **summarize(execution_best), "slots": slot_dicts(execution_best),
            }, indent=2), encoding="utf-8")
            execution_best_path = str(execution_best_path)
        for transaction_index, transaction in enumerate(
            beam_polish_details.get("paired_transactions", []), start=1
        ):
            transaction_dir = artifact_dir / "paired_transactions"
            transaction_dir.mkdir(parents=True, exist_ok=True)
            stem = f"paired_{transaction_index:02d}"
            details = transaction.get("details", {})
            for state_name in ("before", "after"):
                transaction_path = transaction_dir / f"{stem}_{state_name}.json"
                transaction_path.write_text(json.dumps({
                    **summarize(transaction[state_name]),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction[state_name]),
                }, indent=2), encoding="utf-8")
        for transaction_index, transaction in enumerate(
            beam_polish_details.get("phase_transactions", []), start=1
        ):
            transaction_dir = artifact_dir / "phase_transactions"
            transaction_dir.mkdir(parents=True, exist_ok=True)
            stem = f"phase_{transaction_index:02d}"
            details = transaction.get("details", {})
            for state_name in ("before", "after"):
                transaction_path = transaction_dir / f"{stem}_{state_name}.json"
                transaction_path.write_text(json.dumps({
                    **summarize(transaction[state_name]),
                    "transaction": details,
                    "regions": transaction.get("regions", []),
                    "slots": slot_dicts(transaction[state_name]),
                }, indent=2), encoding="utf-8")
            phase_transaction_paths.append({
                "details": details,
                "regions": transaction.get("regions", []),
                "before_json": str(transaction_dir / f"{stem}_before.json"),
                "after_json": str(transaction_dir / f"{stem}_after.json"),
            })
    return {
        "experiment": "direct",
        "mode": mode, "move_time": move_time,
        "target_mode": "same_horizon" if preserve_horizon else "shorten",
        "budget_seconds": budget, "seed": seed,
        "search_budget_seconds": round(deadline - start, 6),
        "plot_reserve_seconds": round(plot_reserve, 6),
        "source": summarize(source), "source_objective": list(source_key),
        "calls": calls, "evaluated": attempts,
        "status": "FOUND" if found_best else ("TIMEOUT" if time.perf_counter() >= deadline else "NO_CANDIDATE"),
        "best": summarize(reported_best),
        "formal_best": summarize(formal_best),
        "continuity_best": summarize(continuity_best),
        "operational_best": summarize(operational_best),
        "execution_best": summarize(execution_best) if execution_best is not None else None,
        "execution_warning": (
            summarize(execution_best)["execution_warning"]
            if execution_best is not None else True
        ),
        "best_is_source": not found_best,
        "beam_search_found": beam_search_found,
        "beam_polish_reserve_seconds": round(beam_polish_reserve, 6),
        "beam_polish": {
            "evaluated": beam_polish_details.get("stats", {}).get("verified", 0),
            "stats": beam_polish_details.get("stats"),
            "stop_reason": beam_polish_details.get("stop_reason"),
            "formal_best": (
                summarize(beam_polish_details["formal_best"])
                if beam_polish_details.get("formal_best") is not None else None
            ),
        },
        "enable_work_transfer": enable_work_transfer,
        "enable_fragmentation_repair": enable_fragmentation_repair,
        "enable_cyclic_exchange": enable_cyclic_exchange,
        "enable_phase_resequence": enable_phase_resequence,
        "enable_phase_closure": enable_phase_closure,
        "enable_cross_crane_phase_relay": enable_cross_crane_phase_relay,
        "enable_idle_capacity_rebalance": enable_idle_capacity_rebalance,
        "execution_output": execution_output,
        "execution_best_path": execution_best_path,
        "phase_transactions": phase_transaction_paths,
        "phase_closure_transactions": phase_closure_transaction_paths,
        "cross_crane_phase_transactions": cross_crane_phase_transaction_paths,
        "local_state_limit": local_state_limit,
        "protect_source_continuity": protect_source_continuity,
        "strict_local_transactions": strict_local_transactions,
        "source_hash": source_hash,
        "use_legacy_seed": use_legacy_seed,
        "enable_multi_relay": enable_multi_relay,
        "window_results": window_results,
        "strict_improvements": [item for item in updates if item["strict_source_improvement"]],
        "candidate_updates": updates,
        "elapsed_seconds": round(time.perf_counter() - start, 6),
    }


def _write_continuity_artifact(
    out: Path,
    label: str,
    candidate,
    W,
    *,
    highlight_window=None,
):
    """Write one verified candidate and its post-search diagnostic plot."""
    json_path = out / f"{label}.json"
    json_path.write_text(json.dumps({
        **summarize(candidate),
        "slots": slot_dicts(candidate),
    }, indent=2), encoding="utf-8")
    png_path = out / f"{label}.png"
    solver.plot_schedule(
        plot_view(candidate), len(W), png_path, show=False,
        diagnostic_title=f"Step 8 continuity — {label}",
        highlight_window=highlight_window,
    )
    return str(json_path), str(png_path)


def continuity_run(
    W, M, S, source, budget, seed, out: Path,
):
    """Run only the fixed-H continuity stage on an immutable source."""
    started = time.perf_counter()
    # Keep a small explicit tail for validation and serialization.  A large
    # deterministic proposal can finish just after its loop check, so the
    # reserve also prevents the advertised search budget from being exceeded
    # by the last decode.
    search_reserve = min(2.0, max(0.25, 0.01 * budget))
    deadline = started + max(0.01, budget - search_reserve)
    trace: list[dict] = []
    details: dict = {}
    result, evaluated = solver._refine_same_horizon_trajectory(
        W, M, S, source, deadline, seed,
        move_time=source.move_time,
        attempt_trace=trace,
        continuity=True,
        result_box=details,
        enable_phase_resequence=True,
        enable_idle_capacity_rebalance=True,
    )
    search_finished = time.perf_counter()
    formal_best = details.get("formal_best", result)
    continuity_best = details.get("continuity_best", result)
    validation_started = time.perf_counter()
    for candidate in (source, formal_best, continuity_best):
        verify_candidate(W, M, S, candidate, source.move_time)
    validation_seconds = time.perf_counter() - validation_started

    out.mkdir(parents=True, exist_ok=True)
    source_json, source_png = _write_continuity_artifact(
        out, "source", source, W
    )
    first_json, first_png = _write_continuity_artifact(
        out, "first_feasible", source, W
    )
    formal_json, formal_png = _write_continuity_artifact(
        out, "formal_best", formal_best, W
    )
    continuity_json, continuity_png = _write_continuity_artifact(
        out, "continuity_best", continuity_best, W
    )
    plot_seconds = time.perf_counter() - validation_started - validation_seconds
    before_rows = solver._candidate_position_rows(source, M)
    after_rows = solver._candidate_position_rows(continuity_best, M)
    changed_regions = solver._history_diff_regions(
        [tuple(row) for row in before_rows],
        [tuple(row) for row in after_rows],
    )
    total_seconds = time.perf_counter() - started
    polish_event = next(
        (item for item in reversed(trace)
         if item.get("phase") == "polish_complete"),
        None,
    )
    return {
        "experiment": "continuity",
        "mode": "trajectory_continuity",
        "move_time": source.move_time,
        "budget_seconds": budget,
        "search_budget_seconds": round(budget - search_reserve, 6),
        "search_reserve_seconds": round(search_reserve, 6),
        "seed": seed,
        "validated": True,
        "search_seconds": round(search_finished - started, 6),
        "validation_seconds": round(validation_seconds, 6),
        "plot_seconds": round(plot_seconds, 6),
        "total_seconds": round(total_seconds, 6),
        "evaluated": evaluated,
        "pool_size": details.get("pool_size"),
        "stop_reason": details.get("stop_reason"),
        "operator_stats": details.get("stats"),
        "source": summarize(source),
        "first_feasible": summarize(source),
        "formal_best": summarize(formal_best),
        "continuity_best": summarize(continuity_best),
        "formal_best_objective": objective(formal_best),
        "continuity_best_objective": objective(continuity_best),
        "changed_regions": changed_regions,
        "source_path": source_json,
        "source_plot": source_png,
        "first_feasible_path": first_json,
        "first_feasible_plot": first_png,
        "formal_best_path": formal_json,
        "formal_best_plot": formal_png,
        "continuity_best_path": continuity_json,
        "continuity_best_plot": continuity_png,
        "attempt_trace": trace,
        "polish_complete": polish_event,
        "elapsed_seconds": round(total_seconds, 6),
    }


def full_run(W, M, S, source, mode, budget, seed, move_time=1):
    start = time.perf_counter()
    solution = solver.solve_cwp_bounded(
        W, M, S, time_limit=budget, seed=seed, critical_mode=mode,
        checkpoint=source,
        move_time=move_time,
    )
    solver.verify_solution(W, M, S, solution)
    key = [solution.makespan, solution.movement_count]
    return {
        "experiment": "full", "mode": mode, "move_time": move_time,
        "budget_seconds": budget,
        "seed": seed, "validated": True, "objective": key,
        "makespan": solution.makespan,
        "schedule_horizon": solution.schedule_horizon,
        "schedule_movement_count": solution.schedule_movement_count,
        "split_bay_count": solution.split_bay_count,
        "load_deviation": solution.load_deviation,
        "reversal_count": solution.reversal_count,
        "makespan_lower_bound": solution.makespan_lower_bound,
        "operator_calls": solution.operator_calls,
        "phase_seconds": solution.phase_seconds,
        "critical_repair_iterations": solution.critical_repair_iterations,
        "critical_repair_improvements": solution.critical_repair_improvements,
        "status": solution.status,
        "elapsed_seconds": round(time.perf_counter() - start, 6),
    }


def paired_summary(records):
    grouped = {}
    for item in records:
        if item.get("experiment") != "full":
            continue
        grouped.setdefault((item["instance"], item["budget_seconds"], item["seed"]), {})[item["mode"]] = item
    result = []
    for key, values in sorted(grouped.items()):
        if "both" not in values or "off_reallocate" not in values:
            continue
        a = tuple(values["both"]["objective"])
        b = tuple(values["off_reallocate"]["objective"])
        result.append({"instance": key[0], "budget_seconds": key[1], "seed": key[2],
                       "both_vs_off_reallocate": "win" if a < b else "loss" if a > b else "tie"})
    grouped_results = {}
    for item in result:
        grouped_results.setdefault((item["instance"], item["budget_seconds"]), []).append(item["both_vs_off_reallocate"])
    for key, values in grouped_results.items():
        wins = values.count("win")
        losses = values.count("loss")
        diffs = [1 if value == "win" else -1 if value == "loss" else 0 for value in values]
        rng = random.Random(20260914 + sum(ord(ch) for ch in f"{key[0]}:{key[1]}"))
        boot = []
        if diffs:
            for _ in range(10000):
                boot.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
        boot.sort()
        result.append({"instance": key[0], "budget_seconds": key[1],
                       "paired_counts": {"win": wins, "loss": losses, "tie": values.count("tie")},
                       "win_minus_loss": wins - losses,
                       "bootstrap_95ci_win_minus_loss_rate": [boot[250], boot[9749]] if boot else None})
    return result


def write_report(out: Path, records):
    lines = [
        "# Step 8 local-repair experiment",
        "",
        "正式目标：`(actual_completion_time, movement_count)`，按词典序比较。",
        "正式 `(C,K)` 冠军单独保留；推荐输出在同一最短工期内先消除同桥吊同贝位回访，再要求同步完工并减少位置回访、方向反转和移动，最后比较负载与停工。",
        "",
    ]
    direct = [r for r in records if r.get("experiment") == "direct"]
    full = [r for r in records if r.get("experiment") == "full"]
    lines.append(f"Direct records: {len(direct)}; full records: {len(full)}.")
    lines.append("")
    for item in direct:
        formal = item.get("formal_best")
        execution = item.get("execution_best")
        if formal is None or execution is None:
            continue
        lines.extend([
            f"## {item.get('instance', 'instance')} / {item.get('mode')} / seed {item.get('seed')}",
            "",
            f"- status: `{item.get('status')}`; search {item.get('search_budget_seconds')}s / "
            f"elapsed {item.get('elapsed_seconds')}s",
            f"- source: `{item.get('source', {}).get('objective')}`; stored H="
            f"{item.get('source', {}).get('schedule_horizon')}, original H="
            f"{item.get('source', {}).get('normalization_source_horizon')}, trimmed slots="
            f"{item.get('source', {}).get('normalization_trimmed_slots')}",
            f"- formal: `{formal.get('objective')}`, warning={formal.get('execution_warning')}",
            f"- execution: `{execution.get('objective')}`, key=`{execution.get('execution_key')}`, "
            f"warning={execution.get('execution_warning')}",
            f"- execution movement/reversals: {execution.get('movement_count')} / "
            f"{execution.get('reversal_count')}; revisits={execution.get('continuity', {}).get('work_revisit_count')}; "
            f"max gap={execution.get('continuity', {}).get('max_work_revisit_gap')}",
        ])
        balanced = item.get("balanced_best")
        balance_metrics = item.get("balanced_best_metrics") or {}
        recommended = item.get("recommended_best")
        recommended_metrics = item.get("recommended_best_metrics") or {}
        if recommended is not None and recommended_metrics:
            recommended_continuity = recommended.get("continuity", {})
            lines.append(
                f"- recommended output: `{recommended.get('objective')}`, loads="
                f"`{recommended_metrics.get('loads')}`, max trailing idle="
                f"{recommended_metrics.get('max_trailing_idle')}, finish gap="
                f"{recommended_metrics.get('finish_gap')}, Umax="
                f"{recommended_metrics.get('max_nonwork_capacity')}, max internal idle blocks="
                f"{recommended_metrics.get('max_internal_idle_blocks')}, total internal idle blocks="
                f"{recommended_metrics.get('total_internal_idle_blocks')}, revisits="
                f"{recommended_continuity.get('work_revisit_count')}"
            )
        balance_transactions = item.get("idle_capacity_transactions") or []
        balance_before = (
            balance_transactions[0].get("balance_before", {})
            if balance_transactions else {}
        )
        if balanced is not None and balance_metrics:
            before_u = balance_before.get("max_nonwork_capacity")
            before_gap = balance_before.get("finish_gap")
            before_key = balanced.get("objective")
            lines.append(
                f"- balanced alternative: `{before_key}`, loads="
                f"`{balance_metrics.get('loads')}`, Umax="
                f"{balance_metrics.get('max_nonwork_capacity')}"
                + (f" (source {before_u})" if before_u is not None else "")
                + f", finish gap={balance_metrics.get('finish_gap')}"
                + (f" (source {before_gap})" if before_gap is not None else "")
                + f", total non-work capacity={balance_metrics.get('total_nonwork_capacity')}; "
                "this is also eligible for the synchronized-finish recommended output."
            )
            balanced_path = item.get("balanced_best_path")
            if balanced_path:
                lines.append(f"- balanced alternative artifact: `{balanced_path}`")
                balanced_plot = str(Path(balanced_path).with_suffix(".png"))
                if (ROOT / balanced_plot).exists():
                    lines.append(f"- balanced schedule plot: `{balanced_plot}`")
        idle_stats = item.get("idle_capacity_rebalance") or {}
        if idle_stats:
            lines.append(
                f"- idle-capacity rebalance: `{idle_stats.get('status')}`, "
                f"no-revisit source={bool(idle_stats.get('focuses_without_revisits'))}, "
                f"focuses={idle_stats.get('idle_capacity_focuses')}, "
                f"partial-transfer focuses={idle_stats.get('partial_transfer_focuses')}, "
                f"verified={idle_stats.get('verified')}, "
                f"state expansions={idle_stats.get('states_expanded')}"
            )
            if balance_transactions:
                sample = balance_transactions[0].get("details", {})
                focus = sample.get("focus", {})
                transfers = sample.get("owner_changes", [])
                lines.append(
                    f"  - best recorded hand-off: Q{focus.get('crane')} bay "
                    f"{focus.get('bay')} -> Q{focus.get('target_crane')}, "
                    f"{focus.get('transfer_length')} units; moves="
                    f"{sample.get('candidate_movement_count')}; ledger closed="
                    f"{sample.get('work_ledger', {}).get('closed')}"
                )
        q3_blocks = [
            block
            for block in execution.get("continuity", {})
            .get("work_blocks_by_crane_bay", {})
            .get("3", [])
            if int(block.get("bay", 0)) == 13
        ]
        q3_moves = execution.get("continuity", {}).get(
            "movement_count_by_crane", {}
        ).get("3")
        lines.append(
            f"- Q3@bay13 blocks: `{q3_blocks}`; Q3 moves={q3_moves}"
        )
        artifact = item.get("execution_best_path")
        if artifact:
            lines.append(f"- execution artifact: `{artifact}`")
        pair_stats = (item.get("continuity_operator_stats") or {}).get(
            "paired_window_cyclic", {}
        )
        if pair_stats:
            lines.append(
                f"- paired exchange: `{pair_stats.get('status')}`, "
                f"generated={pair_stats.get('generated')}, "
                f"verified={pair_stats.get('verified')}, "
                f"accepted={pair_stats.get('accepted')}, "
                f"states early/late={pair_stats.get('early_window_states')}/"
                f"{pair_stats.get('late_window_states')}, "
                f"state_limit={pair_stats.get('state_limit')}"
            )
        phase_stats = (item.get("continuity_operator_stats") or {}).get(
            "phase_block_resequence", {}
        )
        if phase_stats:
            lines.append(
                f"- phase resequence: `{phase_stats.get('status')}`, "
                f"focus={phase_stats.get('focus_revisits')}, "
                f"multi_slot={phase_stats.get('multi_slot_revisits')}, "
                f"complete={phase_stats.get('complete_phase_plans')}, "
                f"verified={phase_stats.get('verified')}, "
                f"accepted={phase_stats.get('accepted')}, "
                f"states={phase_stats.get('states_expanded')}, "
                f"state_limit={phase_stats.get('state_limit')}"
            )
        closure_stats = (item.get("continuity_operator_stats") or {}).get(
            "phase_closure_relay", {}
        )
        if closure_stats:
            observed_max_width = max(
                (
                    int(expansion.get("width", 0))
                    for expansion in closure_stats.get(
                        "activity_chain_expansions", []
                    )
                ),
                default=int(closure_stats.get("max_activity_width", 0)),
            )
            lines.append(
                f"- phase closure: `{closure_stats.get('status')}`, "
                f"focus={closure_stats.get('focus_revisits')}, "
                f"bands={closure_stats.get('activity_bands_tested')}, "
                f"max_width={observed_max_width}, "
                f"complete={closure_stats.get('complete_phase_plans')}, "
                f"ledger_closed={closure_stats.get('ledger_closed')}, "
                f"verified={closure_stats.get('verified')}, "
                f"accepted={closure_stats.get('accepted')}, "
                f"states={closure_stats.get('states_expanded')}, "
                f"state_limit={closure_stats.get('state_limit')}"
            )
            lines.append(
                "  - scope: preserves source bay ownership; work transfer is "
                "handled only by the separate explicit-transfer operators."
            )
            for expansion in closure_stats.get("activity_chain_expansions", [])[:8]:
                lines.append(
                    "  - activity chain: "
                    f"focus={expansion.get('focus')}, "
                    f"width={expansion.get('width')}, "
                    f"active={expansion.get('active_cranes')}, "
                    f"conflicts={len(expansion.get('resolved_conflicts', []))}"
                )
        relay_stats = (item.get("continuity_operator_stats") or {}).get(
            "cross_crane_phase_relay", {}
        )
        if relay_stats:
            observed_relay_width = max(
                (
                    len(expansion.get("active_cranes", []))
                    for expansion in relay_stats.get(
                        "activity_chain_expansions", []
                    )
                ),
                default=int(relay_stats.get("max_activity_width", 0)),
            )
            lines.append(
                f"- cross-crane phase relay: `{relay_stats.get('status')}`, "
                f"focus={relay_stats.get('focus_revisits')}, "
                f"bands={relay_stats.get('activity_bands_tested')}, "
                f"max_width={observed_relay_width}, "
                f"owner_changes={relay_stats.get('owner_change_branches')}, "
                f"two_hop={relay_stats.get('two_hop_relay_branches')}, "
                f"complete={relay_stats.get('complete_phase_plans')}, "
                f"verified={relay_stats.get('verified')}, "
                f"accepted={relay_stats.get('accepted')}, "
                f"states={relay_stats.get('states_expanded')}, "
                f"state_limit={relay_stats.get('state_limit')}"
            )
            lines.append(
                "  - scope: one atomic ledger; phase timing, donor displacement "
                "and receiver ownership are searched together."
            )
            for expansion in relay_stats.get("activity_chain_expansions", [])[:8]:
                lines.append(
                    "  - relay band: "
                    f"focus={expansion.get('focus')}, "
                    f"active={expansion.get('active_cranes')}, "
                    f"tasks={expansion.get('task_count')}, "
                    f"variants={expansion.get('assignment_variants')}, "
                    f"verified={expansion.get('verified_candidates')}"
                )
        prefix_stats = (item.get("continuity_operator_stats") or {}).get(
            "forced_prefix_consolidation", {}
        )
        if prefix_stats:
            lines.append(
                f"- forced-prefix consolidation: `{prefix_stats.get('status')}`, "
                f"focus={prefix_stats.get('focus_interruptions')}, "
                f"ledger_closed={prefix_stats.get('ledger_closed')}, "
                f"verified={prefix_stats.get('verified')}, "
                f"accepted={prefix_stats.get('accepted')}, "
                f"states={prefix_stats.get('states_expanded')}, "
                f"state_limit={prefix_stats.get('state_limit')}"
            )
        rebalance = item.get("global_rebalance") or {}
        if rebalance:
            lines.append(
                f"- global rebalance: `{rebalance.get('status')}`, "
                f"LB={rebalance.get('workload_lower_bound')}, "
                f"targets={rebalance.get('target_horizons')}, "
                f"verified={rebalance.get('schedules_verified')}, "
                f"objective={rebalance.get('best_objective')}, "
                f"loads={rebalance.get('best_loads')}"
            )
        lines.append("")
    for item in paired_summary(records):
        if "paired_counts" in item:
            lines.append(f"- {item['instance']} / {item['budget_seconds']}s: {item['paired_counts']}; CI={item['bootstrap_95ci_win_minus_loss_rate']}.")
    lines.extend(["", "`TIMEOUT` 表示预算内没有得到候选，不表示无解。成功排程只以独立校验通过的记录为证据。"])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--experiment", choices=("prepare", "continuity", "direct", "full", "both"),
        default="both",
    )
    parser.add_argument("--instances", nargs="+", choices=tuple(INSTANCES), default=list(INSTANCES))
    parser.add_argument(
        "--fixed-input", type=Path,
        help="Use this frozen input instead of the named instance inputs.",
    )
    parser.add_argument(
        "--fixed-source", type=Path,
        help="Use this frozen source schedule together with --fixed-input.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budgets", nargs="+", type=float, default=[5.0, 20.0, 60.0])
    parser.add_argument(
        "--modes", nargs="+", choices=("beam", "trajectory", "both"),
        default=["beam", "trajectory", "both"],
        help="Critical-window operators to run in direct experiments.",
    )
    parser.add_argument(
        "--target-mode", choices=("auto", "shorten", "same_horizon"),
        default="auto",
        help=(
            "Step 8 target: shorten actual completion time by default; "
            "same_horizon is a fixed-horizon diagnostic mode."
        ),
    )
    parser.add_argument(
        "--allow-edge-exit", action="store_true",
        help=(
            "For move_time=0, legalize completed edge cranes before source "
            "validation. Prefer outward on-rail moves so Step 8 can reuse "
            "their capacity; exit only when the rail has no legal space."
        ),
    )
    parser.add_argument(
        "--verbose-windows", action="store_true",
        help="Print one result line after every critical-window call.",
    )
    parser.add_argument(
        "--plot-windows", action="store_true",
        help="Save one annotated schedule PNG after every direct window call.",
    )
    parser.add_argument(
        "--independent-windows", action="store_true",
        help=(
            "Disable cumulative local preparation and test every trajectory "
            "window independently from the frozen source."
        ),
    )
    parser.add_argument(
        "--descent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue trying H-1, H-2, ... after the first feasible shortening.",
    )
    parser.add_argument(
        "--operational-repairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run fixed-H block and operational continuity repairs after descent.",
    )
    parser.add_argument(
        "--work-transfer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable explicit local work/position transfer transactions.",
    )
    parser.add_argument(
        "--fragmentation-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bounded paired-window fragmentation and revisit repair.",
    )
    parser.add_argument(
        "--cyclic-exchange",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the adjacent-crane cyclic work-exchange proposal family.",
    )
    parser.add_argument(
        "--phase-resequence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bounded complete-work-phase resequencing for long revisits.",
    )
    parser.add_argument(
        "--phase-closure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable event-time whole-phase closure with progressively expanded "
            "adjacent-crane bands."
        ),
    )
    parser.add_argument(
        "--cross-crane-phase-relay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the atomic full-band search that jointly changes phase "
            "order, work ownership, relay path and safety positions."
        ),
    )
    parser.add_argument(
        "--forced-prefix-consolidation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Consolidate a short interruption after a forced t=0 work prefix.",
    )
    parser.add_argument(
        "--idle-capacity-rebalance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Search whole and partial work hand-offs from early-finished cranes "
            "and preserve a separate same-completion balanced alternative."
        ),
    )
    parser.add_argument(
        "--global-rebalance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Search balanced bay-work partitions and global event schedules "
            "before local descent."
        ),
    )
    parser.add_argument(
        "--local-windows-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable global reconstruction and full-horizon post-processing; "
            "use only cumulative critical-window and adjacent-crane repairs."
        ),
    )
    parser.add_argument(
        "--execution-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a separate execution_best JSON; plots are included when enabled.",
    )
    parser.add_argument(
        "--local-state-limit", type=int, default=256,
        help="Maximum cyclic-exchange local states/proposals per fixed-H round.",
    )
    parser.add_argument(
        "--execution-pool-sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Protect formal/execution/compression champions in the candidate pool.",
    )
    parser.add_argument(
        "--continuity-protection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject fixed-H candidates that add work revisits or bay fragments.",
    )
    parser.add_argument(
        "--strict-local-transactions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only explicit transactions in fixed-H polish; disable legacy decoders.",
    )
    parser.add_argument(
        "--legacy-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ablation: spend the first descent slice on the old cumulative seed.",
    )
    parser.add_argument(
        "--multi-relay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable depth-two adjacent multi-crane relay transactions.",
    )
    parser.add_argument("--source-budget", type=float, default=5.0)
    parser.add_argument(
        "--source-restarts", type=int, default=100_000,
        help="Maximum construction attempts while preparing generated sources.",
    )
    parser.add_argument("--generate-sources", action="store_true")
    parser.add_argument(
        "--source-without-step7", action="store_true",
        help=(
            "Prepare sources using construction only: skip general trajectory "
            "repair and stop immediately before the critical-window step."
        ),
    )
    args = parser.parse_args()
    if args.local_state_limit < 1:
        parser.error("--local-state-limit must be a positive integer")
    if (args.fixed_input is None) != (args.fixed_source is None):
        parser.error("--fixed-input and --fixed-source must be supplied together")
    if args.experiment == "prepare" and not args.generate_sources:
        parser.error("--experiment prepare requires --generate-sources")
    selected_instances = (
        {"fixed": (args.fixed_input, args.fixed_source)}
        if args.fixed_input is not None
        else {name: INSTANCES[name] for name in args.instances}
    )
    args.out.mkdir(parents=True, exist_ok=True)
    backup = args.out / "backup"
    backup.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "cwp_solver.py", backup / "cwp_solver.py")
    shutil.copy2(ROOT / "example_input.json", backup / "example_input.json")
    records = []
    manifest = {"git": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "command": list(sys.argv),
                "code_sha256": sha256(ROOT / "cwp_solver.py"),
                "harness_sha256": sha256(Path(__file__).resolve()),
                "objective": "(actual_completion_time, movement_count)",
                "source_generation": {
                    "budget_seconds": args.source_budget,
                    "restarts": args.source_restarts,
                "without_step7": args.source_without_step7,
                "descent": args.descent,
                "operational_repairs": args.operational_repairs,
                "work_transfer": args.work_transfer,
                "fragmentation_repair": args.fragmentation_repair,
                "cyclic_exchange": args.cyclic_exchange,
                "phase_resequence": args.phase_resequence,
                "phase_closure": args.phase_closure,
                "cross_crane_phase_relay": args.cross_crane_phase_relay,
                "idle_capacity_rebalance": args.idle_capacity_rebalance,
                "forced_prefix_consolidation": args.forced_prefix_consolidation,
                "global_rebalance": args.global_rebalance,
                "execution_output": args.execution_output,
                "local_state_limit": args.local_state_limit,
                "execution_pool_sort": args.execution_pool_sort,
                "continuity_protection": args.continuity_protection,
                "strict_local_transactions": args.strict_local_transactions,
                "legacy_seed": args.legacy_seed,
                "multi_relay": args.multi_relay,
                },
                "instances": {}}
    for name, (input_path, source_path) in selected_instances.items():
        shutil.copy2(input_path, backup / f"{name}_input.json")
        shutil.copy2(source_path, backup / f"{name}_source.json")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        W, M, S = payload["W"], payload["M"], payload.get("S", [])
        move_time = payload.get("move_time", 1)
        manifest["instances"][name] = {"input_sha256": sha256(input_path), "source_sha256": sha256(source_path),
                                       "source_path": str(source_path), "N": len(W), "M": M,
                                       "move_time": move_time, "total_work": sum(W)}
        manifest["instances"][name]["generated_sources"] = {}
        for seed in args.seeds:
            selected_source = source_path
            if args.generate_sources:
                generated_dir = args.out / "sources" / name
                generated_dir.mkdir(parents=True, exist_ok=True)
                selected_source = generated_dir / f"seed_{seed}.json"
                generated = solver.solve_cwp_bounded(
                    W, M, S, time_limit=args.source_budget, seed=seed,
                    restarts=args.source_restarts,
                    critical_mode="off_reallocate",
                    skip_general_repair=args.source_without_step7,
                    stop_before_critical=args.source_without_step7,
                    move_time=move_time,
                )
                solver.verify_solution(W, M, S, generated)
                selected_source.write_text(json.dumps(generated.to_dict(), indent=2), encoding="utf-8")
                print(
                    f"[prepare] instance={name} seed={seed} "
                    f"objective={[generated.makespan, generated.movement_count]} "
                    f"construction_calls={generated.operator_calls.get('construction', 0)} "
                    f"step7_calls={generated.operator_calls.get('trajectory', 0)} "
                    f"step8_calls={generated.operator_calls.get('critical_beam', 0)} "
                    f"source={selected_source}",
                    flush=True,
                )
                manifest["instances"][name]["generated_sources"][str(seed)] = {
                    "path": str(selected_source), "sha256": sha256(selected_source),
                    "method": generated.method,
                    "objective": [generated.makespan, generated.movement_count],
                    "schedule_horizon": generated.schedule_horizon,
                    "schedule_movement_count": generated.schedule_movement_count,
                    "split_bay_count": generated.split_bay_count,
                    "load_deviation": generated.load_deviation,
                }
            source = source_candidate(
                W, M, S, selected_source, move_time, args.allow_edge_exit
            )
            if args.experiment == "continuity":
                for budget in args.budgets:
                    continuity_out = (
                        args.out / "continuity_plots" / name
                        / f"seed_{seed}" / f"budget_{budget:g}s"
                    )
                    item = continuity_run(
                        W, M, S, source, budget, seed, continuity_out
                    )
                    item["instance"] = name
                    item["input_sha256"] = sha256(input_path)
                    item["source_sha256"] = sha256(selected_source)
                    records.append(item)
                continue
            if args.experiment in ("direct", "both"):
                _, starts, configurations = solver._validate_input(W, M, S, move_time)
                initial = solver._initial_configurations(W, starts, configurations)
                eligibility = solver._bay_eligibility(len(W), M, configurations)
                lower_bound = solver._makespan_lower_bound(
                    W, M, initial, eligibility, starts, move_time
                )[0]
                preserve_horizon = (
                    args.target_mode == "same_horizon"
                    or args.target_mode == "auto" and source.makespan <= lower_bound
                )
                if source.makespan > lower_bound or preserve_horizon:
                    for budget in args.budgets:
                        for mode in args.modes:
                            item = direct_run(
                                W, M, S, source, mode, budget, seed,
                                verbose_windows=args.verbose_windows,
                                move_time=move_time,
                                preserve_horizon=preserve_horizon,
                                cumulative_local=not args.independent_windows,
                                enable_descent=args.descent,
                                enable_operational_repairs=args.operational_repairs,
                                enable_work_transfer=args.work_transfer,
                                enable_fragmentation_repair=args.fragmentation_repair,
                                enable_cyclic_exchange=args.cyclic_exchange,
                                enable_phase_resequence=args.phase_resequence,
                                enable_phase_closure=args.phase_closure,
                                enable_cross_crane_phase_relay=args.cross_crane_phase_relay,
                                enable_idle_capacity_rebalance=args.idle_capacity_rebalance,
                                enable_forced_prefix_consolidation=args.forced_prefix_consolidation,
                                enable_global_rebalance=args.global_rebalance,
                                local_windows_only=args.local_windows_only,
                                execution_output=args.execution_output,
                                local_state_limit=args.local_state_limit,
                                execution_pool_sort=args.execution_pool_sort,
                                protect_source_continuity=args.continuity_protection,
                                strict_local_transactions=args.strict_local_transactions,
                                source_hash=sha256(selected_source),
                                use_legacy_seed=args.legacy_seed,
                                enable_multi_relay=args.multi_relay,
                                plot_dir=(
                                    args.out / "window_plots" / name
                                    / f"seed_{seed}" / f"budget_{budget:g}s" / mode
                                    if args.plot_windows else None
                                ),
                                artifact_dir=(
                                    args.out / "schedule_artifacts" / name
                                    / f"seed_{seed}" / f"budget_{budget:g}s" / mode
                                    if args.execution_output and not args.plot_windows
                                    else None
                                ),
                            )
                            item["instance"] = name
                            records.append(item)
                else:
                    records.append({"experiment": "direct", "instance": name, "seed": seed,
                                    "status": "LOWER_BOUND_NO_CALL", "source": summarize(source)})
            if args.experiment in ("full", "both"):
                for budget in (b for b in args.budgets if b in (20.0, 60.0)):
                    modes = ["off_reallocate", "off_reserved", "beam", "trajectory", "both"]
                    random.Random(1000003 + seed).shuffle(modes)
                    for mode in modes:
                        item = full_run(W, M, S, source, mode, budget, seed, move_time)
                        item["instance"] = name
                        item["checkpoint_restored"] = True
                        item["checkpoint_objective"] = summarize(source)
                        item["checkpoint_source_sha256"] = sha256(selected_source)
                        records.append(item)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.out / "paired_summary.json").write_text(json.dumps(paired_summary(records), indent=2), encoding="utf-8")
    write_report(args.out, records)
    print(json.dumps({"records": len(records), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
