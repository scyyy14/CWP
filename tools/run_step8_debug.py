#!/usr/bin/env python3
"""One-click PyCharm entry point for the latest Step 8 local search.

Running this file without parameters evaluates the bundled H208 source for
30 seconds.  Command-line options let a debugger switch fixtures, budgets,
seeds, and custom source schedules without editing the search harness.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_critical_step import main as evaluate_main  # noqa: E402
from render_reference_schedule import render  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the latest Step 8 neighborhood search.",
    )
    parser.add_argument(
        "--case",
        choices=("h208", "h280"),
        default="h208",
        help="Bundled source schedule to use when --input/--source are omitted.",
    )
    parser.add_argument("--input", type=Path, help="Custom instance JSON.")
    parser.add_argument("--source", type=Path, help="Matching source schedule JSON.")
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "step8_debug",
    )
    parser.add_argument(
        "--target-mode",
        choices=("auto", "shorten", "same_horizon"),
        default="shorten",
    )
    parser.add_argument("--local-state-limit", type=int, default=256)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate the schedule chart without opening it automatically.",
    )
    args = parser.parse_args()
    if (args.input is None) != (args.source is None):
        parser.error("--input and --source must be supplied together")
    if args.budget <= 0:
        parser.error("--budget must be positive")
    if args.local_state_limit < 1:
        parser.error("--local-state-limit must be positive")
    return args


def build_evaluator_args(args: argparse.Namespace) -> list[str]:
    if args.input is None:
        case_dir = ROOT / "debug_cases" / args.case
        input_path = case_dir / "instance.json"
        source_path = case_dir / "source_schedule.json"
    else:
        input_path = args.input.expanduser().resolve()
        source_path = args.source.expanduser().resolve()

    return [
        "evaluate_critical_step.py",
        "--experiment", "direct",
        "--fixed-input", str(input_path),
        "--fixed-source", str(source_path),
        "--out", str(args.out.expanduser().resolve()),
        "--budgets", str(args.budget),
        "--seeds", str(args.seed),
        "--modes", "trajectory",
        "--target-mode", args.target_mode,
        "--local-state-limit", str(args.local_state_limit),
        "--local-windows-only",
        "--no-global-rebalance",
        "--execution-output",
        "--idle-capacity-rebalance",
        "--fragmentation-repair",
        "--cyclic-exchange",
        "--phase-resequence",
        "--phase-closure",
        "--cross-crane-phase-relay",
        "--forced-prefix-consolidation",
    ]


def source_path_for(args: argparse.Namespace) -> Path:
    if args.source is not None:
        return args.source.expanduser().resolve()
    return ROOT / "debug_cases" / args.case / "source_schedule.json"


def render_and_show_result(args: argparse.Namespace) -> Path:
    out = args.out.expanduser().resolve()
    budget_label = f"{args.budget:g}"
    artifact_dir = (
        out / "schedule_artifacts" / "fixed" / f"seed_{args.seed}"
        / f"budget_{budget_label}s" / "trajectory"
    )
    result_json = artifact_dir / "execution_best.json"
    if not result_json.exists():
        result_json = artifact_dir / "source.json"
    if not result_json.exists():
        result_json = source_path_for(args)

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    chart_path = out / "step8_best_schedule.svg"
    title = (
        "Step 8 best schedule"
        f" — C{payload.get('makespan', '?')}, K{payload.get('movement_count', '?')}"
    )
    render(payload, chart_path, title=title)
    print(f"Step 8 schedule chart: {chart_path}", flush=True)
    if not args.no_show:
        opened = webbrowser.open(chart_path.as_uri(), new=1)
        if not opened:
            print(
                "The chart was generated but the system could not open it; "
                f"open this file manually: {chart_path}",
                flush=True,
            )
    return chart_path


def main() -> None:
    args = parse_args()
    evaluator_args = build_evaluator_args(args)
    print("Step 8 debug command:", " ".join(evaluator_args), flush=True)
    sys.argv = evaluator_args
    evaluate_main()
    render_and_show_result(args)


if __name__ == "__main__":
    main()
