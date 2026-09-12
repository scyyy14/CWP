#!/usr/bin/env python3
"""Sequential, equal-budget before/after benchmark for the Luna handoff.

The historical solver is loaded from git in memory, so no checkout or working
tree file is overwritten.  Results are written only to the requested report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "6229e27dc432f1cc7bc53dff4a21aa87840c91f1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CASES = {
    "current_39x9": json.loads((ROOT / "example_input.json").read_text()),
    "small_18x5": {
        "W": [9, 7, 4, 4, 9, 0, 4, 6, 2, 7, 4, 4, 3, 8, 3, 4, 2, 0],
        "M": 5,
        "S": [5, 10, 12],
    },
    "small_18x3": {
        "W": [2, 3, 7, 5, 5, 2, 2, 9, 9, 6, 7, 3, 3, 0, 5, 1, 3, 6],
        "M": 3,
        "S": [2],
    },
    "small_21x3": {
        "W": [4, 13, 0, 8, 5, 16, 3, 11, 0, 6, 0, 18, 5, 0, 14, 7, 2, 12, 4, 0, 9],
        "M": 3,
        "S": [2, 6],
    },
}


def load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:cwp_solver.py"], cwd=ROOT
    )
    module = types.ModuleType("cwp_solver_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, "cwp_solver_baseline/cwp_solver.py", "exec"), module.__dict__)
    return module


def run_one(module, version, case_name, payload, seed, budget):
    started = time.perf_counter()
    solution = module.solve_cwp(
        payload["W"], payload["M"], payload.get("S", []),
        time_limit=budget, seed=seed,
    )
    wall = time.perf_counter() - started
    module.verify_solution(payload["W"], payload["M"], payload.get("S", []), solution)
    return {
        "version": version,
        "case": case_name,
        "seed": seed,
        "budget_seconds": budget,
        "wall_seconds": round(wall, 6),
        "makespan": solution.makespan,
        "lower_bound": solution.makespan_lower_bound,
        "proven": solution.makespan_proven_optimal,
        "search_seconds": solution.search_seconds,
        "trajectory_repair_improvements": solution.trajectory_repair_improvements,
        "critical_repair_improvements": solution.critical_repair_improvements,
        "critical_repair_iterations": solution.critical_repair_iterations,
        "phase_seconds": getattr(solution, "phase_seconds", {}),
        "operator_calls": getattr(solution, "operator_calls", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=12.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[42, 7, 20260910])
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    args = parser.parse_args()
    selected = args.case or list(CASES)
    baseline = load_baseline()
    import cwp_solver as current

    records = []
    for case_name in selected:
        payload = CASES[case_name]
        for seed in args.seed:
            # Sequential execution keeps CPU contention out of the comparison.
            records.append(run_one(baseline, "baseline", case_name, payload, seed, args.budget))
            records.append(run_one(current, "current", case_name, payload, seed, args.budget))
            print(json.dumps(records[-2:], ensure_ascii=False), flush=True)
    report = {
        "baseline_commit": BASELINE_COMMIT,
        "budget_seconds": args.budget,
        "seeds": args.seed,
        "cases": selected,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
