import itertools
import json
import random
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import cwp_solver


def slow_worker(payload, options, checkpoint, error_path):
    result = cwp_solver.solve_cwp(**payload, restarts=1, time_limit=0.1)
    Path(checkpoint).write_text(json.dumps(result.to_dict()), encoding='utf-8')
    time.sleep(10)


class SolverImprovementTests(unittest.TestCase):
    def test_completed_edge_crane_stays_available_when_rail_has_space(self):
        slots = [
            cwp_solver.Slot(0, 1, "work", 1, 1, 1),
            cwp_solver.Slot(0, 2, "work", 3, 3, 3),
            cwp_solver.Slot(1, 1, "work", 1, 1, 1),
            cwp_solver.Slot(1, 2, "idle", 3, 3, None),
        ]
        adapted = cwp_solver.apply_completed_edge_exits(
            slots, M=2, N=3, makespan=2, move_time=0
        )
        q2_t1 = next(slot for slot in adapted if slot.time == 1 and slot.crane == 2)
        self.assertEqual(q2_t1.state, "idle")
        self.assertEqual(q2_t1.start_bay, 3)
        check = type("Check", (), {
            "slots": adapted,
            "makespan": 2,
            "crane_loads": [2, 1],
            "reversal_count": 0,
            "movement_count": 0,
            "move_time": 0,
        })()
        cwp_solver.verify_solution([2, 0, 1], 2, [1, 3], check)

    def test_completed_edge_crane_can_still_be_forced_to_exit_for_ablation(self):
        slots = [
            cwp_solver.Slot(0, 1, "work", 1, 1, 1),
            cwp_solver.Slot(0, 2, "work", 3, 3, 3),
            cwp_solver.Slot(1, 1, "work", 1, 1, 1),
            cwp_solver.Slot(1, 2, "idle", 3, 3, None),
        ]
        adapted = cwp_solver.apply_completed_edge_exits(
            slots, M=2, N=3, makespan=2, move_time=0, force_exit=True
        )
        q2_t1 = next(slot for slot in adapted if slot.time == 1 and slot.crane == 2)
        self.assertEqual(q2_t1.state, "offrail")
        self.assertGreater(q2_t1.start_bay, 3)

    def test_completed_suffix_moves_outward_and_remains_reusable(self):
        slots = []
        for t in range(2):
            positions = (7, 10, 13, 14, 16)
            for q, bay in enumerate(positions):
                working = t == 0 or q < 3
                slots.append(cwp_solver.Slot(
                    t, q + 1, "work" if working else "idle",
                    bay, bay, bay if working else None,
                ))
        adapted = cwp_solver.apply_completed_edge_exits(
            slots, M=5, N=21, makespan=2, move_time=0
        )
        final = sorted(
            (slot for slot in adapted if slot.time == 1),
            key=lambda slot: slot.crane,
        )
        self.assertEqual([slot.start_bay for slot in final], [7, 10, 13, 15, 17])
        self.assertTrue(all(slot.state != "offrail" for slot in final))

    def test_move_time_is_configurable_and_default_is_legacy_one(self):
        work = [1, 0, 1]
        results = {
            duration: cwp_solver.solve_cwp(
                work, 1, [1], restarts=20, time_limit=0.1,
                critical_mode="off_reallocate", move_time=duration,
            )
            for duration in (0, 1, 2)
        }
        legacy = cwp_solver.solve_cwp(
            work, 1, [1], restarts=20, time_limit=0.1,
            critical_mode="off_reallocate",
        )
        self.assertEqual([results[d].makespan for d in (0, 1, 2)], [2, 3, 4])
        self.assertEqual(legacy.makespan, results[1].makespan)
        self.assertFalse(any(slot.state == "move" for slot in results[0].slots))
        self.assertEqual(sum(slot.state == "move" for slot in results[2].slots), 2)
        self.assertEqual(results[2].movement_count, 1)
        for duration, solution in results.items():
            self.assertEqual(solution.move_time, duration)
            cwp_solver.verify_solution(work, 1, [1], solution)

    def test_invalid_move_time_is_rejected(self):
        for value in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "move_time"):
                cwp_solver.solve_cwp(
                    [1], 1, [1], restarts=1, time_limit=0.1,
                    move_time=value,
                )

    def test_step8_trajectory_uses_zero_time_movement_capacity(self):
        source = cwp_solver._CandidateSchedule(
            slots=[
                cwp_solver.Slot(0, 1, "work", 1, 1, 1),
                cwp_solver.Slot(1, 1, "idle", 1, 1, None),
                cwp_solver.Slot(2, 1, "work", 3, 3, 3),
                cwp_solver.Slot(3, 1, "idle", 3, 3, None),
            ],
            makespan=4, assignment_count=2, split_bay_count=0,
            load_deviation=0, reversal_count=0, movement_count=1,
            loads=[2], owners=[{0}, set(), {0}], move_time=0,
        )
        repaired, _ = cwp_solver._trajectory_repair(
            [1, 0, 1], 1, [1], source, time.perf_counter() + 0.2, 17,
            active_cranes=(0,), window=(1, 3), move_time=0,
        )
        self.assertIsNotNone(repaired)
        self.assertEqual(repaired.makespan, 2)
        check = type("Check", (), {
            "slots": repaired.slots,
            "makespan": repaired.makespan,
            "crane_loads": repaired.loads,
            "reversal_count": repaired.reversal_count,
            "movement_count": repaired.movement_count,
            "move_time": 0,
        })()
        cwp_solver.verify_solution([1, 0, 1], 1, [1], check)

    def test_step8_same_horizon_improves_secondary_objective(self):
        work = [0, 0, 4, 0, 0]
        slots = []
        for t, positions in enumerate(((1, 3), (1, 3), (3, 5), (3, 5))):
            for q, bay in enumerate(positions):
                is_work = (q == 1 and t < 2) or (q == 0 and t >= 2)
                slots.append(cwp_solver.Slot(
                    t, q + 1, "work" if is_work else "idle",
                    bay, bay, bay if is_work else None,
                ))
        source = cwp_solver._CandidateSchedule(
            slots=slots, makespan=4, assignment_count=2,
            split_bay_count=1, load_deviation=0, reversal_count=0,
            movement_count=2, loads=[2, 2],
            owners=[set(), set(), {0, 1}, set(), set()], move_time=0,
        )
        repaired, _ = cwp_solver._trajectory_repair(
            work, 2, [], source, time.perf_counter() + 0.2, 0,
            active_cranes=(0, 1), window=(1, 3), move_time=0,
            preserve_horizon=True,
        )
        self.assertIsNotNone(repaired)
        self.assertEqual(repaired.makespan, source.makespan)
        self.assertLess(repaired.objective_key, source.objective_key)
        self.assertEqual(repaired.split_bay_count, 0)

    def test_construction_only_source_stops_before_steps_7_and_8(self):
        work = [2, 0, 2, 0, 2]
        result = cwp_solver.solve_cwp(
            work, 2, [1, 5], restarts=4, time_limit=0.2, seed=11,
            critical_mode="both", skip_general_repair=True,
            stop_before_critical=True,
        )
        cwp_solver.verify_solution(work, 2, [1, 5], result)
        self.assertGreater(result.operator_calls["construction"], 0)
        self.assertEqual(result.operator_calls["trajectory"], 0)
        self.assertEqual(result.operator_calls["critical_beam"], 0)
        self.assertEqual(result.operator_calls["mcts"], 0)
        self.assertEqual(result.operator_calls["layered"], 0)
        self.assertEqual(result.operator_calls["exact"], 0)
        self.assertEqual(
            result.method, "dp_dispatch_priority_construction_only_no_solver"
        )

    def test_critical_mode_ablation_switches_only_calls(self):
        kwargs = dict(restarts=2, time_limit=0.2, seed=7)
        off = cwp_solver.solve_cwp([2, 0, 2, 0, 2], 2, [1, 5],
                                   critical_mode="off_reallocate", **kwargs)
        beam = cwp_solver.solve_cwp([2, 0, 2, 0, 2], 2, [1, 5],
                                    critical_mode="beam", **kwargs)
        self.assertEqual(off.operator_calls.get("critical_beam", 0), 0)
        self.assertGreaterEqual(beam.operator_calls.get("critical_beam", 0), 0)

    def test_four_objective_priority_order(self):
        def candidate(h=10, split=0, deviation=0, moves=0, assignments=100, reversals=100):
            return cwp_solver._CandidateSchedule(
                slots=[], makespan=h, assignment_count=assignments,
                split_bay_count=split, load_deviation=deviation,
                reversal_count=reversals, movement_count=moves,
                loads=[], owners=[])

        self.assertLess(candidate(h=9, split=5, deviation=100, moves=100).objective_key,
                        candidate().objective_key)
        self.assertLess(candidate(deviation=100, moves=100).objective_key,
                        candidate(split=1).objective_key)
        self.assertLess(candidate(moves=100).objective_key,
                        candidate(deviation=1).objective_key)
        self.assertLess(candidate(moves=1).objective_key,
                        candidate(moves=2).objective_key)
        self.assertEqual(candidate().objective_key,
                         candidate(assignments=0, reversals=0).objective_key)

    def test_central_load_preference(self):
        weights = [1, 2, 3, 3, 2, 1]
        central = cwp_solver._load_deviation([1, 2, 3, 3, 2, 1], weights, 12)
        edge = cwp_solver._load_deviation([3, 2, 1, 1, 2, 3], weights, 12)
        self.assertEqual(central, 0)
        self.assertLess(central, edge)

    def test_initial_dp_matches_complete_domain(self):
        self.assertIsNone(cwp_solver._weighted_initial_positions([1]*5, 3, [2]))
        rng = random.Random(73)
        for n in range(3, 10):
            for m in range(1, (n + 1) // 2 + 1):
                configs = cwp_solver._legal_configurations(n, m)
                for _ in range(8):
                    starts = list(rng.choice(configs)[::2])
                    weights = [rng.randrange(-4, 6) for _ in range(n)]
                    valid = [c for c in configs if set(starts) <= set(c)]
                    expected = max(sum(weights[b-1] for b in c) for c in valid)
                    result = cwp_solver._weighted_initial_positions(weights, m, starts)
                    self.assertEqual(result[0], expected)
                    self.assertIn(result[1], valid)
                    work = [rng.randrange(5) for _ in range(n)]
                    eligibility = cwp_solver._bay_eligibility(n, m, configs)
                    self.assertEqual(
                        cwp_solver._makespan_lower_bound(work, m, valid, eligibility),
                        cwp_solver._makespan_lower_bound(work, m, valid[:1], eligibility, starts))

    def test_large_initial_pool_is_bounded_and_safe(self):
        work = [1] * 60
        _, starts, configs = cwp_solver._validate_input(work, 12, [1, 9, 21, 45])
        self.assertLessEqual(len(configs), 128)
        self.assertTrue(configs)
        for config in configs:
            self.assertTrue(set(starts) <= set(config))
            self.assertTrue(all(b-a >= 2 for a, b in zip(config, config[1:])))
        # Coverage must account for all legal initial positions, including
        # ones absent from a one-element heuristic pool.
        n, m = 27, 6
        configs = cwp_solver._legal_configurations(n, m)
        starts = [1, 9]
        valid = [c for c in configs if set(starts) <= set(c)]
        work = [i % 4 for i in range(n)]
        eligible = cwp_solver._bay_eligibility(n, m, [])
        self.assertEqual(
            cwp_solver._makespan_lower_bound(work, m, valid, eligible),
            cwp_solver._makespan_lower_bound(work, m, valid[:1], eligible, starts))

    def test_dispatch_dp_matches_exhaustive_scores_and_progress(self):
        rng = random.Random(91)
        for n in range(3, 9):
            for m in range(1, (n + 1) // 2 + 1):
                for _ in range(10):
                    cells = [[None if rng.random() < 0.15 else
                              (rng.uniform(-8, 8), rng.randrange(3))
                              for _ in range(n)] for _ in range(m)]
                    expected = [None, None]
                    for config in cwp_solver._legal_configurations(n, m):
                        chosen = [cells[q][b - 1] for q, b in enumerate(config)]
                        if any(x is None for x in chosen):
                            continue
                        score = sum(x[0] for x in chosen)
                        flag = max(x[1] for x in chosen)
                        for k, allowed in enumerate((flag > 0, flag == 2)):
                            if allowed and (expected[k] is None or score > expected[k]):
                                expected[k] = score
                    actual = cwp_solver._best_safe_dispatch(cells)
                    for want, got in zip(expected, actual):
                        if want is None:
                            self.assertIsNone(got)
                        else:
                            self.assertAlmostEqual(want, got[0])

    def test_trajectory_repair_preserves_first_slot_and_work(self):
        incumbent = cwp_solver._candidate_from_history(
            [1, 0, 1], 1, [(1,), (1,), (1,), (3,), (3,), (3,)])
        repaired, _ = cwp_solver._trajectory_repair(
            [1, 0, 1], 1, [1], incumbent, time.perf_counter() + 1, 42)
        self.assertIsNotNone(repaired)
        self.assertLess(repaired.makespan, incumbent.makespan)
        self.assertEqual(repaired.slots[0].work_bay, 1)
        self.assertEqual(sum(slot.state == 'work' for slot in repaired.slots), 2)
        for a, b in zip(repaired.slots, repaired.slots[1:]):
            self.assertEqual(a.end_bay, b.start_bay)

    def test_critical_window_repair_finds_known_improvement(self):
        incumbent = cwp_solver._candidate_from_history(
            [1, 0, 1], 1, [(1,), (1,), (1,), (3,), (3,), (3,)]
        )
        repaired, _ = cwp_solver._critical_window_beam_repair(
            [1, 0, 1], 1, [1], incumbent, (0,), time.perf_counter() + 0.5, 17,
            window=(1, 3),
        )
        self.assertIsNotNone(repaired)
        self.assertEqual(repaired.makespan, 4)
        self.assertLess(repaired.makespan, incumbent.makespan)
        cwp_solver.verify_solution(
            [1, 0, 1], 1, [1],
            cwp_solver.Solution(
                status="HEURISTIC_FEASIBLE", method="test",
                makespan=repaired.makespan, makespan_lower_bound=1,
                lower_bound_components={}, makespan_proven_optimal=False,
                proven_lexicographic_optimal=False,
                assignment_count=repaired.assignment_count,
                split_bay_count=repaired.split_bay_count,
                load_deviation=repaired.load_deviation,
                reversal_count=repaired.reversal_count,
                movement_count=repaired.movement_count,
                crane_loads=repaired.loads, target_weights=[1],
                bay_cranes={1: [1], 3: [1]}, slots=repaired.slots,
                restarts_completed=0, strategy_evaluations=[],
                layered_search_states=0, layered_search_improvements=0,
                mcts_iterations=0, mcts_improvements=0,
                exact_search_nodes=0, exact_search_improvements=0,
                exact_search_proved_optimal=False, search_seconds=0,
                max_steps=10,
            ),
        )

    def test_critical_windows_are_in_bounds_for_small_fleets(self):
        for m in (1, 2, 3):
            n = 2 * m + 1
            work = [2] + [0] * (n - 2) + [2]
            initial = tuple(1 + 2 * q for q in range(m))
            final = tuple(3 + 2 * q for q in range(m))
            history = [initial] * 3 + [final] * 3
            incumbent = cwp_solver._candidate_from_history(work, m, history)
            windows = cwp_solver._critical_repair_windows(incumbent, m)
            self.assertTrue(windows)
            for chain, (start, end) in windows:
                self.assertTrue(chain)
                self.assertTrue(all(0 <= q < m for q in chain))
                self.assertGreaterEqual(start, 1)
                self.assertGreater(end, start)
                self.assertLessEqual(end - start, max(3, (incumbent.makespan - 1) // 4))
            horizon = incumbent.makespan - 1
            self.assertTrue(any(end == horizon for _, (_, end) in windows))

    def test_trajectory_repair_can_resume_after_deadline(self):
        work = [5, 0, 5]
        incumbent = cwp_solver._candidate_from_history(
            work, 1, [(1,)] * 6 + [(3,)] * 6
        )
        first, first_count, state = cwp_solver._trajectory_repair(
            work, 1, [1], incumbent, time.perf_counter() + 0.001, 9,
            return_state=True,
        )
        self.assertIsNone(first)
        self.assertIsNotNone(state)
        self.assertGreater(first_count, 0)
        second, second_count, state_again = cwp_solver._trajectory_repair(
            work, 1, [1], incumbent, time.perf_counter() + 0.01, 9,
            repair_state=state, return_state=True,
        )
        self.assertIsNone(second)
        self.assertGreaterEqual(second_count, first_count)
        self.assertIsNotNone(state_again)

    def test_process_timeout_returns_validated_checkpoint(self):
        start = time.perf_counter()
        with patch.object(cwp_solver, '_solve_worker', slow_worker):
            solution = cwp_solver.solve_cwp_bounded([1, 0, 1], 1, [1], time_limit=1)
        self.assertLess(time.perf_counter() - start, 3)
        cwp_solver.verify_solution([1, 0, 1], 1, [1], solution)

    def test_nonfinite_budget_is_rejected(self):
        for budget in (float('nan'), float('inf'), 0, -1):
            with self.assertRaises(ValueError):
                cwp_solver.solve_cwp([1], 1, [], time_limit=budget)

    def test_small_results_against_independent_breadth_first_search(self):
        rng = random.Random(2026)
        for _ in range(16):
            n = rng.randrange(3, 7)
            m = rng.randrange(1, min(2, (n + 1) // 2) + 1)
            work = [rng.randrange(3) for _ in range(n)]
            configs = [p for p in itertools.combinations(range(1, n + 1), m)
                       if all(b - a >= 2 for a, b in zip(p, p[1:]))]
            reachable = {b for p in configs for b in p}
            work = [w if i + 1 in reachable else 0 for i, w in enumerate(work)]
            starts = [next(i + 1 for i, w in enumerate(work) if w)] if any(work) else []
            queue = deque((tuple(work), p, 0) for p in configs if set(starts) <= set(p))
            seen = set()
            optimum = None
            while queue:
                remaining, positions, depth = queue.popleft()
                if not any(remaining):
                    optimum = depth
                    break
                for nxt in configs:
                    if depth == 0 and any(nxt[positions.index(b)] != b for b in starts):
                        continue
                    rem = list(remaining)
                    for a, b in zip(positions, nxt):
                        if a == b and rem[a - 1]:
                            rem[a - 1] -= 1
                    state = (tuple(rem), nxt)
                    if state not in seen:
                        seen.add(state)
                        queue.append((*state, depth + 1))
            result = cwp_solver.solve_cwp(work, m, starts, time_limit=0.1, restarts=20)
            cwp_solver.verify_solution(work, m, starts, result)
            self.assertLessEqual(result.makespan_lower_bound, optimum)
            self.assertGreaterEqual(result.makespan, optimum)
            if result.makespan_proven_optimal:
                self.assertEqual(result.makespan, optimum)

    def test_direct_safe_configuration_generation_is_complete(self) -> None:
        for bay_count in range(1, 11):
            for crane_count in range(1, (bay_count + 1) // 2 + 1):
                reference = [
                    positions
                    for positions in itertools.combinations(
                        range(1, bay_count + 1), crane_count
                    )
                    if all(
                        right - left >= 2
                        for left, right in zip(positions, positions[1:])
                    )
                ]
                self.assertEqual(
                    cwp_solver._legal_configurations(bay_count, crane_count),
                    reference,
                )

    def test_direct_eligibility_matches_configuration_enumeration(self) -> None:
        for bay_count in range(3, 11):
            for crane_count in range(1, (bay_count + 1) // 2 + 1):
                configurations = cwp_solver._legal_configurations(
                    bay_count, crane_count
                )
                expected = [
                    {
                        q
                        for q in range(crane_count)
                        if any(config[q] == bay for config in configurations)
                    }
                    for bay in range(1, bay_count + 1)
                ]
                self.assertEqual(
                    cwp_solver._bay_eligibility(
                        bay_count, crane_count, configurations
                    ),
                    expected,
                )

    def test_small_schedule_remains_safe_and_optimal(self) -> None:
        work = [2, 0, 2, 0, 2]
        solution = cwp_solver.solve_cwp(
            work, 2, [1], restarts=100, time_limit=1, seed=7
        )
        cwp_solver.verify_solution(work, 2, [1], solution)
        self.assertTrue(solution.makespan_proven_optimal)
        self.assertEqual(solution.makespan, solution.makespan_lower_bound)

    def test_partition_boundary_relaxation_opens_only_nearby_work(self) -> None:
        owner = [0, 0, 0, 1, 1, 1]
        work = [3, 0, 4, 5, 0, 2]
        relaxed = cwp_solver._relax_partition_boundaries(owner, work, radius=1)
        self.assertEqual(relaxed, [0, 0, None, None, 1, 1])

    def test_partition_boundary_is_found_across_empty_bays(self) -> None:
        relaxed = cwp_solver._relax_partition_boundaries(
            [0, 0, None, 1, 1], [2, 2, 0, 2, 2], radius=1
        )
        self.assertEqual(relaxed, [0, None, None, None, 1])

    def test_validator_rejects_position_teleportation(self) -> None:
        solution = cwp_solver.Solution(
            status="HEURISTIC_FEASIBLE",
            method="test",
            makespan=2,
            makespan_lower_bound=1,
            lower_bound_components={},
            makespan_proven_optimal=False,
            proven_lexicographic_optimal=False,
            assignment_count=2,
            split_bay_count=0,
            load_deviation=0,
            reversal_count=0,
            movement_count=0,
            crane_loads=[2],
            target_weights=[1],
            bay_cranes={1: [1], 3: [1]},
            slots=[
                cwp_solver.Slot(0, 1, "work", 1, 1, 1),
                cwp_solver.Slot(1, 1, "work", 3, 3, 3),
            ],
            restarts_completed=0,
            strategy_evaluations=[],
            layered_search_states=0,
            layered_search_improvements=0,
            mcts_iterations=0,
            mcts_improvements=0,
            exact_search_nodes=0,
            exact_search_improvements=0,
            exact_search_proved_optimal=False,
            search_seconds=0.0,
            max_steps=2,
        )
        with self.assertRaisesRegex(AssertionError, "位置不连续"):
            cwp_solver.verify_solution([1, 0, 1], 1, [], solution)

    def test_many_crane_neighbor_rotation_stays_safe(self) -> None:
        bay_count, crane_count = 15, 5
        configurations = cwp_solver._legal_configurations(bay_count, crane_count)
        eligibility = cwp_solver._bay_eligibility(
            bay_count, crane_count, configurations
        )
        remaining = [1] * bay_count
        criticality = [float(i % 4) for i in range(bay_count)]
        positions = (1, 4, 7, 10, 13)
        seen = set()
        for attempt in range(12):
            neighbors = cwp_solver._focused_exact_neighbors(
                positions,
                remaining,
                eligibility,
                criticality,
                attempt,
                1,
                (2, 5, 8, 11, 14),
            )
            self.assertTrue(neighbors)
            for config in neighbors:
                self.assertTrue(
                    all(b - a >= 2 for a, b in zip(config, config[1:]))
                )
                seen.add(config)
        self.assertGreater(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
