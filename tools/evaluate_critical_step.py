"""Reproducible ablation harness for the original critical-window step.

The harness treats a complete source schedule as an immutable artifact.  The
direct experiment calls only the two legacy operators; the full experiment
uses the solver's critical_mode switch.  It never turns a timeout into an
infeasibility claim and writes one JSON record per run.
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
    if allow_edge_exit:
        slots = solver.apply_completed_edge_exits(
            slots, M, len(W), int(data["makespan"]), move_time
        )
    owners = [set() for _ in W]
    loads = [0] * M
    for slot in slots:
        if slot.state == "work":
            owners[slot.work_bay - 1].add(slot.crane - 1)
            loads[slot.crane - 1] += 1
    makespan = int(data["makespan"])
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
    check = SimpleNamespace(
        slots=slots, makespan=makespan, crane_loads=loads,
        reversal_count=reversal_count, movement_count=movement_count,
        move_time=move_time,
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
    return candidate


def summarize(candidate):
    return {
        "objective": objective(candidate),
        "makespan": candidate.makespan,
        "move_time": candidate.move_time,
        "split_bay_count": candidate.split_bay_count,
        "load_deviation": candidate.load_deviation,
        "movement_count": candidate.movement_count,
        "smoothness": list(solver._trajectory_smoothness(
            candidate, len(candidate.loads)
        )),
        "assignment_count": candidate.assignment_count,
        "reversal_count": candidate.reversal_count,
        "continuity": solver._continuity_diagnostics(
            candidate, len(candidate.loads)
        ),
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
        slots=candidate.slots, makespan=candidate.makespan,
        crane_loads=candidate.loads, reversal_count=candidate.reversal_count,
        assignment_count=candidate.assignment_count,
        split_bay_count=candidate.split_bay_count, bay_cranes=owners,
        movement_count=candidate.movement_count, move_time=move_time,
    )
    solver.verify_solution(W, M, S, check)


def plot_view(candidate):
    """Minimal solution-shaped view accepted by the project plotter."""
    return SimpleNamespace(
        slots=candidate.slots,
        makespan=candidate.makespan,
        makespan_proven_optimal=False,
        reversal_count=candidate.reversal_count,
        movement_count=candidate.movement_count,
        split_bay_count=candidate.split_bay_count,
        crane_loads=candidate.loads,
        move_time=candidate.move_time,
    )


def direct_run(
    W, M, S, source, mode, budget, seed, *, move_time=1,
    verbose_windows=False, plot_dir: Path | None = None,
    preserve_horizon=False, cumulative_local=True,
):
    start = time.perf_counter()
    deadline = start + budget
    # Reserve a small, explicit tail for the optional diagnostic plots.  The
    # search deadline is otherwise extended while rendering, so a nominal
    # 300-second run can exceed its advertised wall-clock budget.
    plot_reserve = 0.0
    if plot_dir is not None and cumulative_local and mode == "trajectory":
        plot_reserve = min(15.0, max(2.0, 0.06 * budget))
        deadline = start + max(0.01, budget - plot_reserve)
    windows = solver._critical_repair_windows(source, M)
    limit = max(1, min(48, len(windows)))
    best = None
    calls = 0
    attempts = 0
    updates = []
    window_results = []
    source_key = tuple(source.objective_key)
    if (
        cumulative_local and mode == "trajectory"
        and move_time == 0 and not preserve_horizon
    ):
        trace = []
        continuity_output = {}
        candidate, evaluated, prepared, first_feasible = solver._cumulative_local_trajectory_repair(
            W, M, S, source, deadline, seed,
            move_time=move_time, attempt_trace=trace,
            continuity_output=continuity_output,
        )
        formal_best = continuity_output.get("formal_best", candidate)
        continuity_best = continuity_output.get("continuity_best", candidate)
        verify_candidate(W, M, S, prepared, move_time)
        if candidate is not None:
            verify_candidate(W, M, S, candidate, move_time)
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
        prepared_path = None
        first_feasible_path = None
        polished_path = None
        formal_best_path = None
        continuity_best_path = None
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
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
            "target_mode": "shorten", "local_strategy": "cumulative",
            "budget_seconds": budget, "seed": seed,
            "search_budget_seconds": round(deadline - start, 6),
            "plot_reserve_seconds": round(plot_reserve, 6),
            "source": summarize(source), "source_objective": list(source_key),
            "calls": len(trace), "evaluated": evaluated,
            "status": "FOUND" if candidate is not None else "TIMEOUT",
            "best": summarize(candidate) if candidate is not None else None,
            "best_objective": (
                list(candidate.objective_key) if candidate is not None else None
            ),
            "best_smoothness": (
                list(solver._trajectory_smoothness(candidate, M))
                if candidate is not None else None
            ),
            "formal_best": summarize(formal_best) if formal_best is not None else None,
            "continuity_best": summarize(continuity_best) if continuity_best is not None else None,
            "continuity_operator_stats": continuity_output.get("stats"),
            "continuity_stop_reason": continuity_output.get("stop_reason"),
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
            "first_feasible_path": first_feasible_path,
            "polished_path": polished_path,
            "formal_best_path": formal_best_path,
            "continuity_best_path": continuity_best_path,
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }
    for index in range(limit):
        if time.perf_counter() >= deadline:
            break
        chain, window = windows[index]
        window_started = time.perf_counter()
        remaining = deadline - time.perf_counter()
        # Divide the complete requested budget across the windows.  A former
        # seven-second cap silently reduced a nominal five-minute experiment
        # to at most 7 * window_count seconds.
        slice_deadline = min(
            deadline,
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
            plot_started = time.perf_counter()
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
            # Diagnostic rendering must not reduce the search budget available
            # to later windows in the same experiment.
            deadline += time.perf_counter() - plot_started
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
    return {
        "experiment": "direct",
        "mode": mode, "move_time": move_time,
        "target_mode": "same_horizon" if preserve_horizon else "shorten",
        "budget_seconds": budget, "seed": seed,
        "search_budget_seconds": round(deadline - start, 6),
        "plot_reserve_seconds": round(plot_reserve, 6),
        "source": summarize(source), "source_objective": list(source_key),
        "calls": calls, "evaluated": attempts,
        "status": "FOUND" if best is not None else ("TIMEOUT" if time.perf_counter() >= deadline else "NO_CANDIDATE"),
        "best": summarize(best) if best is not None else None,
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
    deadline = started + budget
    trace: list[dict] = []
    details: dict = {}
    result, evaluated = solver._refine_same_horizon_trajectory(
        W, M, S, source, deadline, seed,
        move_time=source.move_time,
        attempt_trace=trace,
        continuity=True,
        result_box=details,
    )
    search_finished = time.perf_counter()
    formal_best = details.get("formal_best", result)
    continuity_best = details.get("continuity_best", result)
    validation_started = time.perf_counter()
    for candidate in (source, formal_best, continuity_best):
        verify_candidate(W, M, S, candidate, source.move_time)
    validation_seconds = time.perf_counter() - validation_started

    out.mkdir(parents=True, exist_ok=True)
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
    key = [solution.makespan, solution.split_bay_count,
           solution.movement_count, solution.load_deviation]
    return {
        "experiment": "full", "mode": mode, "move_time": move_time,
        "budget_seconds": budget,
        "seed": seed, "validated": True, "objective": key,
        "makespan": solution.makespan,
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
    lines = ["# Critical-window ablation", "", "四目标：`(makespan, split_bay_count, movement_count, load_deviation)`", ""]
    direct = [r for r in records if r.get("experiment") == "direct"]
    full = [r for r in records if r.get("experiment") == "full"]
    lines.append(f"Direct records: {len(direct)}; full records: {len(full)}.")
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
            "Step 8 target: auto shortens above the lower bound and keeps the "
            "horizon at the lower bound; same_horizon optimizes secondary objectives."
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
                "code_sha256": sha256(ROOT / "cwp_solver.py"), "objective": "(makespan, split_bay_count, movement_count, load_deviation)",
                "source_generation": {
                    "budget_seconds": args.source_budget,
                    "restarts": args.source_restarts,
                    "without_step7": args.source_without_step7,
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
                    f"objective={[generated.makespan, generated.split_bay_count, generated.movement_count, generated.load_deviation]} "
                    f"construction_calls={generated.operator_calls.get('construction', 0)} "
                    f"step7_calls={generated.operator_calls.get('trajectory', 0)} "
                    f"step8_calls={generated.operator_calls.get('critical_beam', 0)} "
                    f"source={selected_source}",
                    flush=True,
                )
                manifest["instances"][name]["generated_sources"][str(seed)] = {
                    "path": str(selected_source), "sha256": sha256(selected_source),
                    "method": generated.method,
                    "objective": [generated.makespan, generated.split_bay_count,
                                   generated.movement_count, generated.load_deviation],
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
                                plot_dir=(
                                    args.out / "window_plots" / name
                                    / f"seed_{seed}" / f"budget_{budget:g}s" / mode
                                    if args.plot_windows else None
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
