import itertools
import unittest

import cwp_solver


class SolverImprovementTests(unittest.TestCase):
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
