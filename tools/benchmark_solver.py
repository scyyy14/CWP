"""Sequential equal-budget comparison; no third-party solver or dependencies."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--budget', type=float, default=12)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', choices=('before', 'after'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    modules = [('before', load('before', args.baseline)),
               ('after', load('after', root / 'cwp_solver.py'))]
    if args.only:
        modules = [(name, module) for name, module in modules if name == args.only]
    cases = {
        'current': json.loads((root / 'example_input.json').read_text()),
        'three_cranes': {'W': [2,3,7,5,5,2,2,9,9,6,7,3,3,0,5,1,3,6], 'M': 3, 'S': [2]},
        'large_work': {'W': [4,13,0,8,5,16,3,11,0,6,0,18,5,0,14,7,2,12,4,0,9], 'M': 3, 'S': [2,6]},
    }
    records = []
    for name, payload in cases.items():
        for seed in (42, 7, 20260910):
            for version, solver in modules:
                start = time.perf_counter()
                solution = solver.solve_cwp(**payload, time_limit=args.budget, seed=seed)
                solver.verify_solution(payload['W'], payload['M'], payload['S'], solution)
                row = dict(case=name, input=payload, seed=seed, version=version,
                           budget=args.budget, wall_seconds=round(time.perf_counter()-start, 6),
                           makespan=solution.makespan, lower_bound=solution.makespan_lower_bound,
                           proven=solution.makespan_proven_optimal, restarts=solution.restarts_completed)
                records.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
