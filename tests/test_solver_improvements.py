import itertools
import json
import random
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import cwp_solver
from tools import evaluate_critical_step
from tools import render_reference_schedule


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

    def test_trajectory_short_excursion_diagnostics(self):
        rows = [
            (2, 9, 13),
            (2, 7, 9),
            (4, 9, 13),
            (4, 9, 13),
            (4, 9, 12),
            (4, 9, 12),
            (4, 9, 13),
            (4, 9, 11),
            (4, 9, 11),
            (4, 9, 13),
        ]
        slots = [
            cwp_solver.Slot(
                t, q + 1, "work", bay, rows[t + 1][q], bay
            )
            for t, row in enumerate(rows[:-1])
            for q, bay in enumerate(row)
        ]
        candidate = cwp_solver._CandidateSchedule(
            slots=slots, makespan=len(rows) - 1,
            assignment_count=3, split_bay_count=0,
            load_deviation=0, reversal_count=0, movement_count=8,
            loads=[3, 3, 3], owners=[{0}, {1}, {2}], move_time=0,
        )
        self.assertEqual(
            cwp_solver._short_excursion_details(candidate, 3),
            [
                (1, 1, 1, 9, 7, 1),
                (2, 1, 1, 13, 9, 1),
                (2, 4, 5, 13, 12, 2),
                (2, 7, 8, 13, 11, 2),
            ],
        )
        self.assertEqual(cwp_solver._trajectory_smoothness(candidate, 3), (4, 6, 11))

    def test_trajectory_long_visit_is_not_a_short_excursion(self):
        rows = [(5,), (7,), (7,), (7,), (5,), (5,)]
        slots = [
            cwp_solver.Slot(t, 1, "work", row[0], rows[t + 1][0], row[0])
            for t, row in enumerate(rows[:-1])
        ]
        candidate = cwp_solver._CandidateSchedule(
            slots=slots, makespan=len(rows) - 1,
            assignment_count=1, split_bay_count=0,
            load_deviation=0, reversal_count=0, movement_count=2,
            loads=[4], owners=[set() for _ in range(9)], move_time=0,
        )
        self.assertEqual(cwp_solver._short_excursion_details(candidate, 1), [])

    def test_continuity_detects_same_crane_bay_fragmentation(self):
        candidate = cwp_solver._candidate_from_history(
            [0, 3, 0], 1, [(2,), (2,), (3,), (2,), (2,)],
            move_time=0, preserve_horizon=True,
        )
        report = cwp_solver._continuity_diagnostics(candidate, 1)
        self.assertEqual(candidate.makespan, 4)
        self.assertEqual(report["bay_fragmentation"], 1)
        self.assertEqual(
            [(item["start"], item["end_exclusive"])
             for item in report["work_blocks_by_bay"]["2"]],
            [(0, 2), (3, 4)],
        )
        self.assertEqual(report["work_revisit_count"], 1)
        self.assertEqual(report["crane_work_revisits"][0]["bay"], 2)

    def test_continuity_reports_crane_bay_segments_and_aba_excursion(self):
        rows = [(2, 9), (2, 7), (2, 9), (2, 9)]
        work = [0] * 9
        work[6] = 1
        work[8] = 2
        plan = {(t, q): None for t in range(3) for q in range(2)}
        plan[(0, 1)] = 9
        plan[(1, 1)] = 7
        plan[(2, 1)] = 9
        candidate = cwp_solver._candidate_from_rows_and_work_plan(
            work, 2, rows, plan,
        )
        report = cwp_solver._continuity_diagnostics(candidate, 2)
        q2_blocks = report["work_blocks_by_crane_bay"]["2"]
        self.assertEqual(
            [(item["bay"], item["start"], item["end_exclusive"])
             for item in q2_blocks],
            [(7, 1, 2), (9, 0, 1), (9, 2, 3)],
        )
        self.assertEqual(report["short_excursion_count"], 1)
        self.assertEqual(report["movement_count_by_crane"]["2"], 2)

    def test_h208_q3_bay13_reports_multislot_forced_start_revisit(self):
        """The real H=208 source must expose Q3's long return explicitly."""
        root = Path(__file__).resolve().parents[1]
        source_path = root / "experiments" / "step8_q2_bay9_consolidation_20260927" / "h208_20s" / "schedule_artifacts" / "fixed" / "seed_0" / "budget_20s" / "trajectory" / "execution_best.json"
        input_path = root / "experiments" / "step8_q2_bay9_consolidation_20260927" / "smoke_h208_8s_v3" / "backup" / "fixed_input.json"
        if not source_path.exists() or not input_path.exists():
            self.skipTest("最新 H=208 Q3 回归文件不在精简测试环境中")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        instance = json.loads(input_path.read_text(encoding="utf-8"))
        slots = [cwp_solver.Slot(**item) for item in source["slots"]]
        owners = [set() for _ in instance["W"]]
        loads = [0] * instance["M"]
        for slot in slots:
            if slot.state == "work":
                owners[slot.work_bay - 1].add(slot.crane - 1)
                loads[slot.crane - 1] += 1
        candidate = cwp_solver._CandidateSchedule(
            slots=slots,
            makespan=source["makespan"],
            assignment_count=source["assignment_count"],
            split_bay_count=source["split_bay_count"],
            load_deviation=source["load_deviation"],
            reversal_count=source["reversal_count"],
            movement_count=source["movement_count"],
            loads=loads,
            owners=owners,
            move_time=0,
        )
        report = cwp_solver._continuity_diagnostics(candidate, instance["M"])
        revisit = next(
            item for item in report["crane_work_revisits"]
            if item["crane"] == 3 and item["bay"] == 13
        )
        self.assertEqual(
            (revisit["primary_block"]["start"], revisit["primary_block"]["end_exclusive"]),
            (0, 52),
        )
        self.assertEqual(
            (revisit["residual_block"]["start"], revisit["residual_block"]["work_count"]),
            (182, 24),
        )
        self.assertEqual(revisit["gap"], 130)
        self.assertEqual(revisit["total_work_on_crane_bay"], 76)
        self.assertEqual(revisit["prefix_completion_end"], 76)
        self.assertTrue(any(
            item["crane"] == 3 and item["bay"] == 13
            for item in report["multi_slot_revisits"]
        ))
        self.assertTrue(any(
            item["crane"] == 3 and item["bay"] == 13
            for item in report["forced_start_revisits"]
        ))
        self.assertEqual(report["return_move_count_by_crane"]["3"], 1)
        self.assertTrue(any(
            item["crane"] == 3 and 4 in item["interfering_cranes"]
            for item in report["crane_work_revisits"]
            if item["bay"] == 13
        ))

    def test_h208_forced_prefix_operator_consolidates_q2_bay9(self):
        """The real Q2 interruption must be repaired within the bounded operator."""
        root = Path(__file__).resolve().parents[1]
        artifact_dir = (
            root / "experiments" / "step8_q3_bay13_phase_resequence_20260927"
            / "smoke_h208_3s_v6" / "schedule_artifacts" / "fixed" / "seed_0"
            / "budget_3s" / "trajectory"
        )
        source_path = artifact_dir / "execution_best.json"
        input_path = (
            root / "experiments" / "step8_q3_bay13_phase_resequence_20260927"
            / "smoke_h208_3s_v6" / "backup" / "fixed_input.json"
        )
        if not source_path.exists() or not input_path.exists():
            self.skipTest("当前固定 H=208 Q2 回归文件不在精简测试环境中")
        source_data = json.loads(source_path.read_text(encoding="utf-8"))
        instance = json.loads(input_path.read_text(encoding="utf-8"))
        slots = [cwp_solver.Slot(**item) for item in source_data["slots"]]
        owners = [set() for _ in instance["W"]]
        loads = [0] * instance["M"]
        for slot in slots:
            if slot.state == "work":
                owners[slot.work_bay - 1].add(slot.crane - 1)
                loads[slot.crane - 1] += 1
        source = cwp_solver._CandidateSchedule(
            slots=slots,
            makespan=source_data["makespan"],
            assignment_count=source_data["assignment_count"],
            split_bay_count=source_data["split_bay_count"],
            load_deviation=source_data["load_deviation"],
            reversal_count=source_data["reversal_count"],
            movement_count=source_data["movement_count"],
            loads=loads,
            owners=owners,
            move_time=0,
        )
        before = cwp_solver._continuity_diagnostics(
            source, instance["M"], instance["S"]
        )
        interruption = before["forced_prefix_interruptions"][0]
        self.assertEqual(interruption["crane"], 2)
        self.assertEqual(interruption["forced_bay"], 9)
        self.assertEqual(interruption["interrupt_bays"], [7])
        self.assertEqual(interruption["interrupt_work"], 3)

        proposals, stats = cwp_solver._forced_prefix_consolidation_exchange(
            instance["W"], instance["M"], instance["S"], source,
            state_limit=4096,
            max_candidates=32,
            deadline=time.perf_counter() + 10,
            source_hash="h208-q2-forced-prefix-test",
        )
        self.assertGreaterEqual(stats["focus_interruptions"], 1)
        self.assertGreaterEqual(stats["neighbor_bands_generated"], 1)
        self.assertGreaterEqual(stats["ledger_closed"], 1)
        self.assertEqual(stats["status"], "CANDIDATE_FOUND")
        self.assertLess(stats["states_expanded"], 4096)
        self.assertTrue(proposals)

        repaired = cwp_solver._candidate_from_explicit_phase_transaction(
            instance["W"], instance["M"], instance["S"],
            proposals[0]["history"], source,
            proposals[0]["work_plan"], proposals[0]["active_cranes"],
            source_hash="h208-q2-forced-prefix-test",
            transaction_source_hash=proposals[0]["source_hash"],
        )
        after = cwp_solver._continuity_diagnostics(
            repaired, instance["M"], instance["S"]
        )
        self.assertEqual(repaired.objective_key, (206, 12))
        self.assertEqual(after["work_block_count_by_crane_bay"]["2"]["9"], 1)
        self.assertEqual(after["forced_prefix_interruption_count"], 0)
        self.assertEqual(repaired.movement_count, 12)
        self.assertEqual(repaired.split_bay_count, source.split_bay_count)
        self.assertEqual(
            cwp_solver._candidate_position_rows(repaired, instance["M"])[0],
            cwp_solver._candidate_position_rows(source, instance["M"])[0],
        )
        for t in range(source.makespan):
            for q in (2, 3, 4):
                before_slot = next(
                    slot for slot in source.slots
                    if slot.time == t and slot.crane == q + 1
                )
                after_slot = next(
                    slot for slot in repaired.slots
                    if slot.time == t and slot.crane == q + 1
                )
                self.assertEqual(
                    (before_slot.start_bay, before_slot.work_bay),
                    (after_slot.start_bay, after_slot.work_bay),
                )
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            instance["W"], instance["M"], instance["S"], repaired,
        ))

    def test_phase_block_resequence_merges_multislot_revisit(self):
        """A bounded phase permutation removes a synthetic long return."""
        work = [0, 8, 0, 1, 0, 5, 0, 8]
        rows = [
            (2, 6, 8), (2, 6, 8), (2, 4, 8), (2, 6, 8),
            (2, 6, 8), (2, 6, 8), (2, 6, 8), (2, 6, 8),
            (2, 6, 8),
        ]
        plan = {(t, q): None for t in range(8) for q in range(3)}
        for t in range(8):
            plan[(t, 0)] = 2
            plan[(t, 2)] = 8
        for t in (0, 1, 3, 4, 5):
            plan[(t, 1)] = 6
        plan[(2, 1)] = 4
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 3, rows, plan,
        )
        before = cwp_solver._continuity_diagnostics(source, 3)
        self.assertEqual(before["multi_slot_revisits"][0]["residual_length"], 3)
        proposals, stats = cwp_solver._phase_block_resequence_exchange(
            work, 3, [2, 6, 8], source,
            max_active_cranes=3,
            state_limit=5000,
            max_candidates=4,
            source_hash="synthetic-phase-source",
        )
        self.assertEqual(stats["status"], "SEARCH_COMPLETE")
        self.assertGreaterEqual(stats["multi_slot_revisits"], 1)
        self.assertGreaterEqual(stats["complete_phase_plans"], 1)
        self.assertTrue(proposals)
        repaired = cwp_solver._candidate_from_explicit_phase_transaction(
            work, 3, [2, 6, 8], proposals[0]["history"], source,
            proposals[0]["work_plan"], proposals[0]["active_cranes"],
            source_hash="synthetic-phase-source",
            transaction_source_hash=proposals[0]["source_hash"],
        )
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            work, 3, [2, 6, 8], repaired,
        ))
        after = cwp_solver._continuity_diagnostics(repaired, 3)
        self.assertFalse(any(
            item["crane"] == 2 and item["bay"] == 6
            for item in after["crane_work_revisits"]
        ))
        self.assertEqual(after["extra_work_blocks_total"], 0)

    def test_phase_closure_event_search_merges_revisit_and_reduces_moves(self):
        work = [0, 2, 0, 1, 0, 1]
        starts = [2, 6]
        rows = [(2, 6), (4, 6), (2, 6), (2, 6)]
        plan = {(t, q): None for t in range(3) for q in range(2)}
        plan[(0, 0)] = 2
        plan[(0, 1)] = 6
        plan[(1, 0)] = 4
        plan[(2, 0)] = 2
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 2, rows, plan,
        )
        self.assertEqual(source.objective_key, (3, 2))
        proposals, stats = cwp_solver._phase_closure_relay_search(
            work, 2, starts, source, state_limit=5_000, max_candidates=8,
            source_hash="synthetic-phase-closure-source",
        )
        self.assertEqual(stats["status"], "SEARCH_COMPLETE")
        self.assertEqual(stats["focus_revisits"], 1)
        self.assertGreater(stats["ledger_closed"], 0)
        self.assertGreater(stats["event_schedules_generated"], 0)
        self.assertTrue(proposals)
        proposal = proposals[0]
        repaired = cwp_solver._candidate_from_explicit_phase_transaction(
            work, 2, starts, proposal["history"], source,
            proposal["work_plan"], proposal["active_cranes"],
            source_hash="synthetic-phase-closure-source",
            transaction_source_hash=proposal["source_hash"],
        )
        self.assertEqual(repaired.objective_key, (3, 1))
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            work, 2, starts, repaired,
        ))
        after = cwp_solver._continuity_diagnostics(repaired, 2, starts)
        self.assertFalse(any(
            item["crane"] == 1 and item["bay"] == 2
            for item in after["crane_work_revisits"]
        ))

    def test_phase_closure_reports_unsupported_nonzero_move_time(self):
        source = cwp_solver._candidate_from_rows_and_work_plan(
            [0, 2, 0, 1, 0, 1], 2,
            [(2, 6), (4, 6), (2, 6), (2, 6)],
            {(0, 0): 2, (0, 1): 6, (1, 0): 4, (1, 1): None,
             (2, 0): 2, (2, 1): None},
        )
        source.move_time = 1
        proposals, stats = cwp_solver._phase_closure_relay_search(
            [0, 2, 0, 1, 0, 1], 2, [2, 6], source,
        )
        self.assertEqual(proposals, [])
        self.assertEqual(stats["status"], "UNSUPPORTED")
        self.assertEqual(stats["reason"], "nonzero_move_time")

    def test_cross_crane_phase_relay_changes_owner_in_one_verified_transaction(self):
        """A focus residual can be handed to an adjacent idle crane.

        The source has Q2 revisiting bay 5.  The new operator is required to
        emit a complete ledger candidate whose residual bay-5 phase is owned by
        another crane; the old owner-preserving closure cannot express this
        change.
        """
        work = [0, 1, 0, 0, 2, 0, 0, 1, 0]
        starts = [2, 5, 8]
        rows = [(2, 5, 8), (2, 4, 8), (2, 5, 8), (2, 5, 8)]
        plan = {(t, q): None for t in range(3) for q in range(3)}
        plan[(0, 0)] = 2
        plan[(0, 1)] = 5
        plan[(2, 1)] = 5
        plan[(0, 2)] = 8
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 3, rows, plan,
        )
        proposals, stats = cwp_solver._cross_crane_phase_relay_search(
            work, 3, starts, source,
            state_limit=5_000, max_candidates=8,
            source_hash="synthetic-cross-crane-source",
        )
        self.assertEqual(stats["status"], "SEARCH_COMPLETE")
        self.assertGreaterEqual(stats["owner_change_branches"], 1)
        self.assertGreaterEqual(stats["ledger_closed"], 1)
        self.assertTrue(proposals)
        proposal = next(
            item for item in proposals
            if item["details"]["owner_changes"]
        )
        repaired = cwp_solver._candidate_from_explicit_cross_crane_phase_transaction(
            work, 3, starts, proposal["history"], source,
            proposal["work_plan"], proposal["active_cranes"],
            source_hash="synthetic-cross-crane-source",
            transaction_source_hash=proposal["source_hash"],
        )
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            work, 3, starts, repaired,
        ))
        self.assertEqual(
            proposal["details"]["work_ledger"]["source_by_bay"],
            proposal["details"]["work_ledger"]["transaction_by_bay"],
        )
        self.assertTrue(any(
            item["from_crane"] == 2 and item["to_crane"] in (1, 3)
            for item in proposal["details"]["owner_changes"]
        ))

    def test_phase_closure_transactions_and_stats_reach_iterative_report(self):
        work = [0, 2, 0, 1, 0, 1]
        starts = [2, 6]
        rows = [(2, 6), (4, 6), (2, 6), (2, 6)]
        plan = {(t, q): None for t in range(3) for q in range(2)}
        plan[(0, 0)] = 2
        plan[(0, 1)] = 6
        plan[(1, 0)] = 4
        plan[(2, 0)] = 2
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 2, rows, plan,
        )
        output = {}
        cwp_solver._cumulative_local_trajectory_repair_iterative(
            work, 2, starts, source, time.perf_counter() + 2.0, 7,
            move_time=0, continuity_output=output,
            enable_descent=False, enable_operational_repairs=True,
            preserve_horizon=True, enable_work_transfer=False,
            enable_fragmentation_repair=False, enable_cyclic_exchange=False,
            enable_phase_resequence=False, enable_phase_closure=True,
            enable_forced_prefix_consolidation=False,
            strict_local_transactions=True, local_state_limit=1,
            source_hash="synthetic-phase-closure-source",
        )
        self.assertEqual(output["formal_best"].objective_key, (3, 1))
        self.assertTrue(output["phase_closure_transactions"])
        closure = output["stats"]["phase_closure_relay"]
        self.assertGreater(closure["verified"], 0)
        self.assertGreater(closure["accepted"], 0)

    def test_phase_closure_real_q3_revisit_expands_through_five_cranes(self):
        root = Path(__file__).resolve().parents[1]
        input_path = root / "experiments" / "step8_q3_bay13_phase_resequence_20260927" / "smoke_h208_3s_v6" / "backup" / "fixed_input.json"
        source_path = root / "experiments" / "step8_completion_moves_20260928" / "latest_source_300s" / "global_best.json"
        instance = json.loads(input_path.read_text(encoding="utf-8"))
        source = evaluate_critical_step.source_candidate(
            instance["W"], instance["M"], instance["S"], source_path,
            move_time=0,
        )
        diagnostics = cwp_solver._continuity_diagnostics(
            source, instance["M"], instance["S"]
        )
        focus = next(
            item for item in diagnostics["crane_work_revisits"]
            if item["crane"] == 3 and item["bay"] == 13
        )
        self.assertEqual(source.objective_key, (206, 12))
        self.assertEqual(focus["primary_block"]["start"], 0)
        self.assertEqual(focus["primary_block"]["end_exclusive"], 52)
        self.assertEqual(focus["residual_block"]["start"], 182)
        self.assertEqual(focus["residual_block"]["end_exclusive"], 206)
        self.assertEqual(focus["total_work_on_crane_bay"], 76)
        proposals, stats = cwp_solver._phase_closure_relay_search(
            instance["W"], instance["M"], instance["S"], source,
            state_limit=50_000, max_candidates=8,
        )
        self.assertEqual(stats["status"], "SEARCH_COMPLETE")
        self.assertEqual(stats["focus_revisits"], 1)
        self.assertGreater(stats["activity_bands_tested"], 0)
        self.assertEqual(stats["max_activity_width"], instance["M"])
        self.assertEqual(proposals, [])

    def test_cross_crane_phase_relay_real_q3_improves_completion_and_moves(self):
        """The unified relay resolves the real Q3@bay13 residual phase."""
        root = Path(__file__).resolve().parents[1]
        input_path = root / "experiments" / "step8_q3_bay13_phase_resequence_20260927" / "smoke_h208_3s_v6" / "backup" / "fixed_input.json"
        source_path = root / "experiments" / "step8_completion_moves_20260928" / "latest_source_300s" / "global_best.json"
        instance = json.loads(input_path.read_text(encoding="utf-8"))
        source = evaluate_critical_step.source_candidate(
            instance["W"], instance["M"], instance["S"], source_path,
            move_time=0,
        )
        proposals, stats = cwp_solver._cross_crane_phase_relay_search(
            instance["W"], instance["M"], instance["S"], source,
            state_limit=50_000, max_candidates=8,
            max_assignment_variants=256,
            deadline=time.perf_counter() + 2.0,
            source_hash="real-cross-crane-source",
        )
        self.assertIn(stats["status"], {"CANDIDATE_FOUND", "SEARCH_COMPLETE"})
        self.assertGreaterEqual(stats["owner_change_branches"], 1)
        self.assertTrue(proposals)
        proposal = min(
            proposals, key=lambda item: tuple(item["details"]["candidate_objective"])
        )
        repaired = cwp_solver._candidate_from_explicit_cross_crane_phase_transaction(
            instance["W"], instance["M"], instance["S"],
            proposal["history"], source, proposal["work_plan"],
            proposal["active_cranes"],
            source_hash="real-cross-crane-source",
            transaction_source_hash=proposal["source_hash"],
        )
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            instance["W"], instance["M"], instance["S"], repaired,
        ))
        self.assertLess(repaired.objective_key, source.objective_key)
        self.assertEqual(repaired.objective_key, (204, 10))
        self.assertTrue(any(
            item["bay"] == 13
            and item["from_crane"] == 3
            and item["to_crane"] == 4
            for item in proposal["details"]["owner_changes"]
        ))

    def test_global_rebalance_reaches_h280_workload_lower_bound(self):
        root = Path(__file__).resolve().parents[1]
        input_path = root / "experiments" / "mentor_step8_20260916_h280" / "input" / "instance.json"
        source_path = root / "experiments" / "mentor_step8_20260916_h280" / "input" / "source_schedule.json"
        instance = json.loads(input_path.read_text(encoding="utf-8"))
        source = evaluate_critical_step.source_candidate(
            instance["W"], instance["M"], instance["S"], source_path,
            move_time=0,
        )

        candidate, stats = cwp_solver._global_balanced_assignment_search(
            instance["W"], instance["M"], instance["S"], source,
            time.perf_counter() + 5.0,
        )

        self.assertEqual(stats["workload_lower_bound"], 221)
        self.assertEqual(stats["status"], "FOUND")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.objective_key, (221, 12))
        self.assertEqual(candidate.loads, [221, 221, 221, 221])
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            instance["W"], instance["M"], instance["S"], candidate,
        ))

    def test_global_rebalance_runs_without_source_revisits(self):
        root = Path(__file__).resolve().parents[1]
        input_path = root / "experiments" / "mentor_step8_20260916_h280" / "input" / "instance.json"
        source_path = root / "experiments" / "mentor_step8_20260916_h280" / "input" / "source_schedule.json"
        instance = json.loads(input_path.read_text(encoding="utf-8"))
        source = evaluate_critical_step.source_candidate(
            instance["W"], instance["M"], instance["S"], source_path,
            move_time=0,
        )
        self.assertEqual(cwp_solver._continuity_diagnostics(
            source, instance["M"], instance["S"]
        )["work_revisit_count"], 0)
        output = {}

        candidate, _evaluated, _prepared, first = (
            cwp_solver._cumulative_local_trajectory_repair_iterative(
                instance["W"], instance["M"], instance["S"], source,
                time.perf_counter() + 5.0, 19, move_time=0,
                continuity_output=output, enable_descent=False,
                enable_operational_repairs=False,
            )
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.objective_key, (221, 12))
        self.assertEqual(first.objective_key, (221, 12))
        self.assertEqual(output["global_rebalance"]["status"], "FOUND")

    def test_global_rebalance_supports_two_cranes(self):
        work = [0, 2, 0, 2, 0]
        rows = [(2, 4)] * 5
        work_plan = {
            (time_index, crane): (
                2 if crane == 0 and time_index < 2
                else 4 if crane == 1 and time_index >= 2
                else None
            )
            for time_index in range(4)
            for crane in range(2)
        }
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 2, rows, work_plan,
        )

        candidate, stats = cwp_solver._global_balanced_assignment_search(
            work, 2, [], source, time.perf_counter() + 2.0,
        )

        self.assertEqual(stats["status"], "FOUND")
        self.assertEqual(candidate.objective_key, (2, 0))
        self.assertEqual(candidate.loads, [2, 2])
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            work, 2, [], candidate,
        ))

    def test_global_rebalance_reserves_mandatory_start_work_before_rebalancing(self):
        work = [0, 4, 0, 4]
        rows = [(2, 4)] * 6
        work_plan = {
            (time_index, crane): (
                2 if crane == 0 and time_index in {0, 1, 3, 4}
                else 4 if crane == 1 and time_index < 4
                else None
            )
            for time_index in range(5)
            for crane in range(2)
        }
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 2, rows, work_plan,
        )

        candidate, stats = cwp_solver._global_balanced_assignment_search(
            work, 2, [2, 4], source, time.perf_counter() + 2.0,
        )

        self.assertEqual(source.completion_time, 5)
        self.assertEqual(stats["status"], "FOUND")
        self.assertEqual(candidate.objective_key, (4, 0))
        self.assertEqual(candidate.loads, [4, 4])
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            work, 2, [2, 4], candidate,
        ))

    def test_continuity_detects_long_and_abc_revisits(self):
        candidate = cwp_solver._candidate_from_history(
            [0, 0, 0, 0, 2, 0, 3, 0, 1], 1,
            [(5,), (7,), (7,), (7,), (9,), (5,), (5,)],
            move_time=0, preserve_horizon=True,
        )
        report = cwp_solver._continuity_diagnostics(candidate, 1)
        blocks = report["crane_position_blocks"]
        seven_block = next(item for item in blocks if item["position"] == 7)
        self.assertEqual(seven_block["length"], 3)
        self.assertEqual(report["work_revisit_count"], 1)
        self.assertEqual(report["crane_work_revisits"][0]["bay"], 5)
        self.assertEqual(cwp_solver._short_excursion_details(candidate, 1), [])

    def test_h208_continuity_regression_sees_known_fragmentation(self):
        root = Path(__file__).resolve().parents[1]
        source_path = root / "experiments" / "mentor_step8_trajectory_smoothing_20260917" / "seed0_300s_segment_operators" / "window_plots" / "fixed" / "seed_0" / "budget_300s" / "trajectory" / "polished_best.json"
        input_path = root / "experiments" / "mentor_step8_baseline_20260917_h209" / "input" / "instance.json"
        if not source_path.exists() or not input_path.exists():
            self.skipTest("历史 H=208 实验文件不在精简测试环境中")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        instance = json.loads(input_path.read_text(encoding="utf-8"))
        slots = [cwp_solver.Slot(**item) for item in source["slots"]]
        M = instance["M"]
        owners = [set() for _ in instance["W"]]
        loads = [0] * M
        for slot in slots:
            if slot.state == "work":
                owners[slot.work_bay - 1].add(slot.crane - 1)
                loads[slot.crane - 1] += 1
        candidate = cwp_solver._CandidateSchedule(
            slots=slots, makespan=source["makespan"],
            assignment_count=source["assignment_count"],
            split_bay_count=source["split_bay_count"],
            load_deviation=source["load_deviation"],
            reversal_count=source["reversal_count"],
            movement_count=source["movement_count"], loads=loads,
            owners=owners, move_time=0,
        )
        report = cwp_solver._continuity_diagnostics(candidate, M)
        self.assertEqual(
            [(item["start"], item["end_exclusive"])
             for item in report["work_blocks_by_bay"]["2"]],
            [(0, 1), (207, 208)],
        )
        q3_bay11 = [
            item for item in report["crane_work_revisits"]
            if item["crane"] == 3 and item["bay"] == 11
        ]
        self.assertTrue(q3_bay11)
        self.assertEqual(q3_bay11[0]["previous_block"]["length"], 7)

    def test_continuity_does_not_count_pure_yielding_as_work_revisit(self):
        candidate = cwp_solver._candidate_from_history(
            [0, 0, 1], 1, [(1,), (3,), (1,), (1,)],
            move_time=0, preserve_horizon=True,
        )
        report = cwp_solver._continuity_diagnostics(candidate, 1)
        self.assertEqual(report["work_revisit_count"], 0)
        self.assertEqual(report["position_revisit_count"], 1)
        self.assertEqual(len(report["pure_yielding_revisits"]), 1)

    def test_fixed_horizon_decoder_keeps_global_idle_rows(self):
        compressed = cwp_solver._candidate_from_history(
            [0, 0, 1], 1, [(1,), (1,), (3,), (3,)], move_time=0
        )
        fixed = cwp_solver._candidate_from_history(
            [0, 0, 1], 1, [(1,), (1,), (3,), (3,)],
            move_time=0, preserve_horizon=True,
        )
        self.assertEqual(compressed.makespan, 1)
        self.assertEqual(fixed.makespan, 3)
        self.assertEqual(len(fixed.slots), 3)
        cwp_solver.verify_solution(
            [0, 0, 1], 1, [],
            type("Check", (), {
                "slots": fixed.slots,
                "makespan": fixed.makespan,
                "crane_loads": fixed.loads,
                "reversal_count": fixed.reversal_count,
                "movement_count": fixed.movement_count,
                "move_time": 0,
            })(),
        )

    def test_revisit_exchange_is_paired_and_verified(self):
        source_history = [(5,), (7,), (7,), (7,), (5,), (5,), (5,)] + [(7,)] * 8
        candidate = cwp_solver._candidate_from_history(
            [0, 0, 0, 0, 4, 0, 10, 0, 0], 1,
            source_history, move_time=0, preserve_horizon=True,
        )
        proposals = cwp_solver._continuity_block_neighbors(
            [0, 0, 0, 0, 4, 0, 10, 0, 0], 1, candidate,
        )
        exchange = next(
            proposal_history for name, proposal_history in proposals
            if name == "revisit_work_exchange"
        )
        changed = cwp_solver._history_diff_regions(
            [tuple(row) for row in source_history], exchange
        )
        self.assertLessEqual(len(changed), 2)
        self.assertTrue(all(item["length"] <= 8 for item in changed))
        repaired = cwp_solver._candidate_from_history(
            [0, 0, 0, 0, 4, 0, 10, 0, 0], 1, exchange,
            move_time=0, preserve_horizon=True,
        )
        self.assertEqual(repaired.makespan, candidate.makespan)
        cwp_solver.verify_solution(
            [0, 0, 0, 0, 4, 0, 10, 0, 0], 1, [5],
            type("Check", (), {
                "slots": repaired.slots,
                "makespan": repaired.makespan,
                "crane_loads": repaired.loads,
                "reversal_count": repaired.reversal_count,
                "movement_count": repaired.movement_count,
                "move_time": 0,
            })(),
        )

    def test_explicit_work_transfer_is_a_real_local_transaction(self):
        """A local relay moves work with its crane and freezes the outside."""
        work = [0, 4, 0, 0, 0]
        source = cwp_solver._candidate_from_history(
            work, 2, [(2, 5)] * 5, move_time=0, preserve_horizon=True,
        )
        proposals, stats = cwp_solver._work_transfer_transactions(
            work, 2, source, max_candidates=32,
        )
        self.assertGreater(stats["unique"], 0)
        proposal = next(
            item for item in proposals
            if item["operator"] == "tail_relay"
            and item["details"]["source_interval"] == [1, 4]
        )
        candidate = cwp_solver._candidate_from_explicit_work_transaction(
            work, 2, proposal["history"], source,
            proposal["work_plan"], proposal["regions"],
        )
        self.assertEqual(len({item["crane"] for item in proposal["regions"]}), 2)
        self.assertTrue(all(item["length"] <= 8 for item in proposal["regions"]))
        check = type("Check", (), {
            "slots": candidate.slots, "makespan": candidate.makespan,
            "crane_loads": candidate.loads,
            "reversal_count": candidate.reversal_count,
            "movement_count": candidate.movement_count,
            "move_time": 0,
        })()
        cwp_solver.verify_solution(work, 2, [], check)
        # The first slot is outside the [1,4) work transfer and remains the
        # original Q1 work / Q2 idle pair.
        self.assertEqual(
            [(slot.crane, slot.state, slot.work_bay)
             for slot in candidate.slots if slot.time == 0],
            [(1, "work", 2), (2, "idle", None)],
        )

    def test_residual_exchange_closes_a_real_same_bay_gap(self):
        work = [0, 4, 0, 0, 0, 0]
        rows = [(2,)] * 6
        source_plan = {
            (t, 0): (2 if t in (0, 1, 3, 4) else None)
            for t in range(5)
        }
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 1, rows, source_plan,
        )
        before = cwp_solver._continuity_diagnostics(source, 1)
        proposals, stats = cwp_solver._work_transfer_transactions(
            work, 1, source, max_candidates=16,
        )
        proposal = next(
            item for item in proposals
            if item["operator"] == "paired_residual_exchange"
        )
        candidate = cwp_solver._candidate_from_explicit_work_transaction(
            work, 1, proposal["history"], source,
            proposal["work_plan"], proposal["regions"],
        )
        after = cwp_solver._continuity_diagnostics(candidate, 1)
        self.assertEqual(before["bay_fragmentation"], 1)
        self.assertEqual(after["bay_fragmentation"], 0)
        self.assertEqual(after["work_revisit_count"], 0)
        self.assertEqual(len(proposal["regions"]), 2)
        self.assertTrue(all(item["length"] <= 8 for item in proposal["regions"]))
        self.assertEqual(stats["operators"]["paired_residual_exchange"], 1)

    def test_paired_window_cyclic_exchange_closes_long_residual_and_freezes_outside(self):
        """A two-window relay absorbs a one-slot return without changing H."""
        work = [0, 2, 0, 10, 0, 10, 0, 1]
        horizon = 12
        rows = [(2, 6, 8)]
        rows.append((4, 6, 8))
        rows.extend([(4, 6, 8)] * 9)
        rows.extend([(2, 6, 8), (2, 6, 8)])
        self.assertEqual(len(rows), horizon + 1)
        work_plan = {
            (t, q): None
            for t in range(horizon)
            for q in range(3)
        }
        work_plan[(0, 0)] = 2
        work_plan[(horizon - 1, 0)] = 2
        for t in range(1, 11):
            work_plan[(t, 0)] = 4
            work_plan[(t, 1)] = 6
        work_plan[(1, 2)] = 8
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 3, rows, work_plan,
        )
        before = cwp_solver._continuity_diagnostics(source, 3)
        self.assertEqual(before["max_work_revisit_gap"], 10)
        proposals, stats = cwp_solver._paired_window_cyclic_work_exchange(
            work, 3, source, max_candidates=16, state_limit=16,
            source_hash="frozen-source",
        )
        proposal = next(
            item for item in proposals
            if item["details"]["crane"] == 1
            and item["details"]["bay"] == 2
            and item["details"]["chain"] == [1, 2]
        )
        candidate = cwp_solver._candidate_from_explicit_work_transaction(
            work, 3, proposal["history"], source,
            proposal["work_plan"], proposal["regions"],
            source_hash="frozen-source",
            transaction_source_hash=proposal["source_hash"],
        )
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            work, 3, [], candidate,
        ))
        after = cwp_solver._continuity_diagnostics(candidate, 3)
        q1_bay2_blocks = [
            block for block in after["crane_position_blocks"]
            if block["crane"] == 1
            and block["position"] == 2
            and block["work_count"]
        ]
        self.assertEqual(len(q1_bay2_blocks), 1)
        self.assertEqual(q1_bay2_blocks[0]["start"], 0)
        self.assertEqual(q1_bay2_blocks[0]["end_exclusive"], 2)
        self.assertEqual(after["max_work_revisit_gap"], 0)
        self.assertEqual(proposal["details"]["ledger_delta"]["closed"], {})
        self.assertEqual(stats["status"], "SEARCH_COMPLETE")
        self.assertEqual(len({item["segment"] for item in proposal["regions"]}), 2)
        self.assertTrue(all(item["length"] <= 8 for item in proposal["regions"]))
        self.assertEqual(candidate.makespan, source.makespan)

        editable = {
            (t, q)
            for region in proposal["regions"]
            for t in range(region["start"], region["end_exclusive"])
            for q in (region["crane"] - 1,)
        }
        source_slots = {(s.time, s.crane - 1): s for s in source.slots}
        candidate_slots = {(s.time, s.crane - 1): s for s in candidate.slots}
        for key, original in source_slots.items():
            if key in editable:
                continue
            changed = candidate_slots[key]
            self.assertEqual(
                (changed.state, changed.start_bay, changed.end_bay, changed.work_bay),
                (original.state, original.start_bay, original.end_bay, original.work_bay),
            )

    def test_h208_one_slot_residual_at_q1_bay2_has_exact_200_slot_gap(self):
        """Lock the H=208 shape and exercise the real cyclic proposal path."""
        horizon = 208
        work = [0, 2, 0, 200, 0, 1, 0, 1]
        rows = [(2, 6, 8)]
        rows.extend([(4, 6, 8)] * 200)
        rows.extend([(2, 6, 8)] * 8)
        self.assertEqual(len(rows), horizon + 1)
        work_plan = {
            (t, q): None for t in range(horizon) for q in range(3)
        }
        work_plan[(0, 0)] = 2
        work_plan[(201, 0)] = 2
        for t in range(1, 201):
            work_plan[(t, 0)] = 4
        work_plan[(1, 1)] = 6
        work_plan[(1, 2)] = 8
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 3, rows, work_plan,
        )
        before = cwp_solver._continuity_diagnostics(source, 3)
        revisit = next(
            item for item in before["crane_work_revisits"]
            if item["crane"] == 1 and item["bay"] == 2
        )
        self.assertEqual(revisit["primary_block"]["start"], 0)
        self.assertEqual(revisit["residual_block"]["start"], 201)
        self.assertEqual(revisit["gap"], 200)

        proposals, stats = cwp_solver._paired_window_cyclic_work_exchange(
            work, 3, source, max_candidates=16, state_limit=16,
            source_hash="h208-regression-source",
        )
        proposal = next(
            item for item in proposals
            if item["details"]["crane"] == 1
            and item["details"]["bay"] == 2
            and item["details"]["chain"] == [1, 2]
        )
        repaired = cwp_solver._candidate_from_explicit_work_transaction(
            work, 3, proposal["history"], source,
            proposal["work_plan"], proposal["regions"],
            source_hash="h208-regression-source",
            transaction_source_hash=proposal["source_hash"],
        )
        after = cwp_solver._continuity_diagnostics(repaired, 3)
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            work, 3, [], repaired,
        ))
        self.assertFalse(any(
            item["crane"] == 1 and item["bay"] == 2
            for item in after["crane_work_revisits"]
        ))
        self.assertEqual(
            [
                (block["start"], block["end_exclusive"])
                for block in after["crane_position_blocks"]
                if block["crane"] == 1
                and block["position"] == 2
                and block["work_count"]
            ],
            [(0, 2)],
        )
        self.assertEqual(proposal["details"]["ledger_delta"]["closed"], {})
        self.assertEqual(stats["status"], "SEARCH_COMPLETE")

    def test_paired_window_cyclic_exchange_reports_bounded_search_as_unknown(self):
        work = [0, 2, 0, 10, 0, 10, 0, 1]
        horizon = 12
        rows = [(2, 6, 8), (4, 6, 8)] + [(4, 6, 8)] * 9 + [
            (2, 6, 8), (2, 6, 8),
        ]
        work_plan = {
            (t, q): None for t in range(horizon) for q in range(3)
        }
        work_plan[(0, 0)] = work_plan[(horizon - 1, 0)] = 2
        for t in range(1, 11):
            work_plan[(t, 0)] = 4
            work_plan[(t, 1)] = 6
        work_plan[(1, 2)] = 8
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 3, rows, work_plan,
        )
        proposals, stats = cwp_solver._paired_window_cyclic_work_exchange(
            work, 3, source, max_candidates=16, state_limit=1,
        )
        self.assertEqual(stats["expanded_states"], 1)
        self.assertGreaterEqual(stats["state_limit"], 1)
        self.assertEqual(stats["status"], "UNKNOWN")
        # A bounded local miss is explicitly not a global infeasibility claim.
        self.assertNotEqual(stats["status"], "INFEASIBLE")

    def test_execution_rank_prioritizes_completion_and_moves_before_diagnostics(self):
        work = [1, 0, 1]
        horizon = 4
        continuous_rows = [(1,), (3,), (3,), (3,), (3,)]
        continuous_work = {
            (t, 0): ({0: 1, 1: 3}.get(t))
            for t in range(horizon)
        }
        continuous = cwp_solver._candidate_from_rows_and_work_plan(
            work, 1, continuous_rows, continuous_work,
        )
        revisit_rows = [(1,), (3,), (1,), (1,), (1,)]
        revisit_work = {
            (t, 0): ({0: 1, 1: 3}.get(t))
            for t in range(horizon)
        }
        revisit = cwp_solver._candidate_from_rows_and_work_plan(
            work, 1, revisit_rows, revisit_work,
        )
        self.assertEqual(continuous.completion_time, revisit.completion_time)
        self.assertEqual(continuous.objective_key, revisit.objective_key)
        self.assertEqual(continuous.split_bay_count, revisit.split_bay_count)

        longer_rows = [(1,)] * 210
        longer_work = {
            (t, 0): (1 if t in (0, 208) else None)
            for t in range(209)
        }
        longer_continuous = cwp_solver._candidate_from_rows_and_work_plan(
            [2, 0, 0], 1, longer_rows, longer_work,
        )
        short_with_revisit_rows = [(1,)] + [(3,)] * 206 + [(1,), (1,)]
        short_with_revisit_work = {
            (t, 0): (1 if t in (0, 207) else None)
            for t in range(208)
        }
        short_with_revisit = cwp_solver._candidate_from_rows_and_work_plan(
            [2, 0, 0], 1, short_with_revisit_rows, short_with_revisit_work,
        )
        self.assertEqual(short_with_revisit.completion_time, 208)
        self.assertEqual(longer_continuous.completion_time, 209)
        self.assertLess(
            cwp_solver._execution_rank(short_with_revisit, 1),
            cwp_solver._execution_rank(longer_continuous, 1),
        )

        # At equal C, K wins even if diagnostic fields are much worse.
        higher_move_slots = [
            cwp_solver.Slot(0, 1, "work", 1, 1, 1),
            cwp_solver.Slot(0, 2, "work", 5, 5, 5),
            cwp_solver.Slot(1, 1, "work", 3, 3, 3),
            cwp_solver.Slot(1, 2, "work", 7, 7, 7),
        ]
        lower_move_slots = [
            cwp_solver.Slot(0, 1, "work", 1, 1, 1),
            cwp_solver.Slot(0, 2, "work", 5, 5, 5),
            cwp_solver.Slot(1, 1, "work", 1, 1, 1),
            cwp_solver.Slot(1, 2, "work", 7, 7, 7),
        ]
        higher_moves = cwp_solver._CandidateSchedule(
            slots=higher_move_slots, makespan=2, assignment_count=4,
            split_bay_count=0, load_deviation=0, reversal_count=0,
            movement_count=2, loads=[2, 2],
            owners=[{0}, set(), {0}, set(), {1}, set(), {1}], move_time=0,
        )
        lower_moves_more_splits = cwp_solver._CandidateSchedule(
            slots=lower_move_slots, makespan=2, assignment_count=4,
            split_bay_count=99, load_deviation=10**9, reversal_count=50,
            movement_count=1, loads=[2, 2],
            owners=[{0, 1}, set(), {0, 1}, set(), {1}, set(), {1}], move_time=0,
        )
        self.assertEqual(higher_moves.completion_time, lower_moves_more_splits.completion_time)
        self.assertLess(lower_moves_more_splits.objective_key, higher_moves.objective_key)
        self.assertLess(
            cwp_solver._execution_rank(lower_moves_more_splits, 2),
            cwp_solver._execution_rank(higher_moves, 2),
        )

    def test_work_transfer_rejects_outside_work_and_oversized_contract(self):
        work = [0, 5, 0, 0, 0]
        source = cwp_solver._candidate_from_history(
            work, 2, [(2, 5)] * 6, move_time=0, preserve_horizon=True,
        )
        proposals, _ = cwp_solver._work_transfer_transactions(
            work, 2, source, max_candidates=32,
        )
        proposal = next(
            item for item in proposals
            if item["details"]["source_interval"] == [1, 4]
        )
        tampered = dict(proposal["work_plan"])
        tampered[(4, 0)] = None
        with self.assertRaisesRegex(RuntimeError, "窗口外作业"):
            cwp_solver._candidate_from_explicit_work_transaction(
                work, 2, proposal["history"], source,
                tampered, proposal["regions"],
            )
        with self.assertRaisesRegex(RuntimeError, "不能超过8"):
            cwp_solver._normalize_work_transfer_regions(
                [{"crane": 1, "start": 0, "end_exclusive": 9}],
                10, 2,
            )
        with self.assertRaisesRegex(RuntimeError, "三台相邻"):
            cwp_solver._normalize_work_transfer_regions(
                [
                    {"crane": 1, "start": 0, "end_exclusive": 1},
                    {"crane": 2, "start": 0, "end_exclusive": 1},
                    {"crane": 3, "start": 0, "end_exclusive": 1},
                    {"crane": 4, "start": 0, "end_exclusive": 1},
                ],
                source.makespan, 4,
            )

    def test_work_transfer_rejects_source_hash_mismatch(self):
        work = [0, 4, 0, 0, 0]
        source = cwp_solver._candidate_from_history(
            work, 2, [(2, 5)] * 5, move_time=0, preserve_horizon=True,
        )
        proposals, _ = cwp_solver._work_transfer_transactions(
            work, 2, source, max_candidates=32, source_hash="source-a",
        )
        proposal = next(
            item for item in proposals if item["operator"] == "tail_relay"
        )
        with self.assertRaisesRegex(RuntimeError, "source_hash"):
            cwp_solver._candidate_from_explicit_work_transaction(
                work, 2, proposal["history"], source,
                proposal["work_plan"], proposal["regions"],
                source_hash="source-b",
                transaction_source_hash=proposal["source_hash"],
            )

    def test_strict_polish_accepts_only_explicit_continuity_transaction(self):
        work = [0, 4, 0, 0, 0, 0]
        rows = [(2,)] * 6
        source_plan = {
            (t, 0): (2 if t in (0, 1, 3, 4) else None)
            for t in range(5)
        }
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 1, rows, source_plan,
        )
        details = {}
        refined, _ = cwp_solver._refine_same_horizon_trajectory(
            work, 1, [], source, time.perf_counter() + 0.2, 4,
            continuity=True, result_box=details,
            enable_work_transfer=True,
            protect_source_continuity=True,
            strict_local_transactions=True,
        )
        self.assertEqual(
            cwp_solver._continuity_diagnostics(refined, 1)["bay_fragmentation"],
            0,
        )
        self.assertGreaterEqual(
            details["stats"]["work_transfer"]["accepted"], 1,
        )

    def test_continuity_can_accept_more_than_six_successive_improvements(self):
        source_history = [
            (1,), (3,), (1,), (3,), (1,), (3,), (1,), (3,),
            (1,), (3,), (1,), (3,), (1,), (3,), (1,),
        ]
        candidate = cwp_solver._candidate_from_history(
            [1, 0, 0], 1, source_history, move_time=0,
            preserve_horizon=True,
        )
        proposal_histories = []
        working = list(source_history)
        for index in range(1, 8):
            working[2 * index - 1] = (1,)
            proposal_histories.append(("test_progress", [tuple(row) for row in working]))
        with patch.object(cwp_solver, "_trajectory_segment_neighbors", return_value=[]), \
             patch.object(
                 cwp_solver, "_continuity_block_neighbors",
                 side_effect=[[item] for item in proposal_histories] + [[]],
             ), patch.object(cwp_solver, "_critical_repair_windows", return_value=[]):
            refined, evaluated = cwp_solver._refine_same_horizon_trajectory(
                [1, 0, 0], 1, [1], candidate,
                time.perf_counter() + 0.5, 91, move_time=0,
                continuity=True,
            )
        self.assertGreaterEqual(evaluated, 7)
        self.assertLessEqual(refined.movement_count, 1)

    def test_trajectory_segment_neighbors_are_block_level(self):
        rows = [
            (2, 9, 13),
            (2, 7, 9),
            (4, 9, 13),
            (4, 9, 13),
            (4, 9, 12),
            (4, 9, 12),
            (4, 9, 13),
            (4, 9, 11),
            (4, 9, 11),
            (4, 9, 13),
        ]
        slots = [
            cwp_solver.Slot(
                t, q + 1, "work", bay, rows[t + 1][q], bay
            )
            for t, row in enumerate(rows[:-1])
            for q, bay in enumerate(row)
        ]
        candidate = cwp_solver._CandidateSchedule(
            slots=slots, makespan=len(rows) - 1,
            assignment_count=3, split_bay_count=1,
            load_deviation=0, reversal_count=0, movement_count=8,
            loads=[3, 3, 3], owners=[{0}, {1}, {2}], move_time=0,
        )
        neighbors = cwp_solver._trajectory_segment_neighbors(
            [0] * 13, 3, [], candidate
        )
        names = {name for name, _history in neighbors}
        self.assertIn("short_visit_eliminate", names)
        self.assertIn("continuous_handoff_batch", names)
        self.assertIn("visit_boundary_slide_delay", names)
        self.assertTrue(all(len(history) == candidate.makespan + 1
                            for _name, history in neighbors))
        self.assertTrue(all(history[0] == tuple(rows[0])
                            for _name, history in neighbors))

    def test_polish_cannot_trade_earlier_completion_for_smoothness(self):
        def make_candidate(rows, work_time):
            slots = [
                cwp_solver.Slot(
                    t, 1,
                    "work" if t == work_time else "idle",
                    row[0], row[0], row[0] if t == work_time else None,
                )
                for t, row in enumerate(rows[:-1])
            ]
            return cwp_solver._CandidateSchedule(
                slots=slots, makespan=len(rows) - 1,
                assignment_count=1, split_bay_count=0,
                load_deviation=0,
                reversal_count=sum(
                    a != b
                    for a, b in zip(
                        [1 if rows[i + 1][0] > rows[i][0] else -1
                         for i in range(len(rows) - 1)
                         if rows[i + 1][0] != rows[i][0]],
                        [1 if rows[i + 1][0] > rows[i][0] else -1
                         for i in range(len(rows) - 1)
                         if rows[i + 1][0] != rows[i][0]][1:],
                    )
                ),
                movement_count=2,
                loads=[1], owners=[set(), set(), {0}], move_time=0,
            )

        rough = make_candidate([(1,), (3,), (1,), (1,), (1,)], 1)
        smooth = make_candidate([(1,), (2,), (3,), (3,), (3,)], 2)
        with patch.object(
            cwp_solver, "_critical_repair_windows", return_value=[((0,), (1, 3))]
        ), patch.object(
            cwp_solver, "_trajectory_segment_neighbors", return_value=[]
        ), patch.object(
            cwp_solver, "_trajectory_repair",
            side_effect=[(smooth, 1), (None, 1), (None, 1)],
        ):
            refined, evaluated = cwp_solver._refine_same_horizon_trajectory(
                [0, 0, 1], 1, [], rough, time.perf_counter() + 0.2, 5,
                move_time=0,
            )
        self.assertEqual(refined.slots, rough.slots)
        self.assertLess(rough.objective_key, smooth.objective_key)
        self.assertEqual(evaluated, 2)

    def test_same_horizon_smoothing_returns_valid_input_at_deadline(self):
        candidate = cwp_solver._candidate_from_history(
            [1, 0, 1], 1, [(1,), (1,), (3,), (3,)], move_time=0
        )
        refined, evaluated = cwp_solver._refine_same_horizon_trajectory(
            [1, 0, 1], 1, [1], candidate,
            time.perf_counter() - 1.0, 13, move_time=0,
        )
        self.assertEqual(refined.slots, candidate.slots)
        self.assertEqual(refined.makespan, candidate.makespan)
        self.assertEqual(evaluated, 0)
        cwp_solver.verify_solution(
            [1, 0, 1], 1, [1],
            type("Check", (), {
                "slots": refined.slots,
                "makespan": refined.makespan,
                "crane_loads": refined.loads,
                "reversal_count": refined.reversal_count,
                "movement_count": refined.movement_count,
                "move_time": 0,
            })(),
        )

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

    def test_formal_objective_is_completion_then_moves_only(self):
        def candidate(h=10, moves=0, split=0, deviation=0):
            positions = [1] * h
            position = 1
            for t in range(1, min(moves, h - 1) + 1):
                position = 4 - position
                positions[t] = position
            for t in range(min(moves, h - 1) + 1, h):
                positions[t] = position
            slots = [
                cwp_solver.Slot(
                    t, 1, "work", positions[t], positions[t], positions[t],
                )
                for t in range(h)
            ]
            return cwp_solver._CandidateSchedule(
                slots=slots, makespan=h, assignment_count=h,
                split_bay_count=split, load_deviation=deviation,
                reversal_count=0, movement_count=moves,
                loads=[h], owners=[{0}], move_time=0)

        self.assertLess(
            candidate(h=9, split=5, deviation=100, moves=8).objective_key,
            candidate().objective_key,
        )
        self.assertLess(
            candidate(h=10, split=99, deviation=10**9, moves=0).objective_key,
            candidate(h=10, split=0, deviation=0, moves=1).objective_key,
        )
        self.assertLess(
            candidate(h=205, moves=20).objective_key,
            candidate(h=206, moves=11).objective_key,
        )
        self.assertLess(
            candidate(h=206, moves=11, split=99, deviation=10**9).objective_key,
            candidate(h=206, moves=12, split=0, deviation=0).objective_key,
        )

    def test_fixed_reference_sources_normalize_to_real_completion(self):
        input_path = Path(
            "experiments/step8_q3_bay13_phase_resequence_20260927/"
            "smoke_h208_3s_v6/backup/fixed_input.json"
        )
        source_paths = [
            Path(
                "experiments/step8_q3_bay13_phase_resequence_20260927/"
                "smoke_h208_3s_v6/schedule_artifacts/fixed/seed_0/"
                "budget_3s/trajectory/execution_best.json"
            ),
            Path(
                "experiments/step8_q2_bay9_forced_prefix_20260927/"
                "smoke_h208_5s/schedule_artifacts/fixed/seed_0/"
                "budget_5s/trajectory/execution_best.json"
            ),
        ]
        data = json.loads(input_path.read_text(encoding="utf-8"))
        W, M, S = data["W"], data["M"], data.get("S", [])
        expected = [(206, 13), (206, 12)]
        for source_path, objective in zip(source_paths, expected):
            source = evaluate_critical_step.source_candidate(
                W, M, S, source_path, data.get("move_time", 0),
            )
            self.assertEqual(source.objective_key, objective)
            self.assertEqual(source.completion_time, 206)
            self.assertEqual(source.schedule_horizon, 206)
            self.assertEqual(source.normalization_source_horizon, 208)
            self.assertEqual(source.normalization_trimmed_slots, 2)
            evaluate_critical_step.verify_candidate(
                W, M, S, source, data.get("move_time", 0),
            )
            chart = render_reference_schedule.chart_data(
                json.loads(source_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(chart[3], 206)
            self.assertEqual(chart[4], 208)
            self.assertEqual(chart[5], objective[1])

    def test_normalization_trims_tail_without_changing_work_or_source(self):
        slots = [
            cwp_solver.Slot(0, 1, "work", 1, 1, 1),
            cwp_solver.Slot(1, 1, "work", 1, 1, 1),
            cwp_solver.Slot(2, 1, "work", 3, 3, 3),
            cwp_solver.Slot(3, 1, "idle", 3, 3, None),
            cwp_solver.Slot(4, 1, "idle", 5, 5, None),
            cwp_solver.Slot(5, 1, "idle", 5, 5, None),
        ]
        padded = cwp_solver._CandidateSchedule(
            slots=slots, makespan=6, assignment_count=2,
            split_bay_count=0, load_deviation=0, reversal_count=0,
            movement_count=2, loads=[3],
            owners=[{0}, set(), {0}, set(), set()], move_time=0,
        )
        self.assertEqual(padded.objective_key, (3, 1))
        self.assertEqual(padded.schedule_horizon, 6)
        evaluate_critical_step.verify_candidate([2, 0, 1, 0, 0], 1, [1], padded, 0)
        normalized = cwp_solver._normalize_completed_candidate(padded)
        again = cwp_solver._normalize_completed_candidate(normalized)
        self.assertEqual(normalized.objective_key, (3, 1))
        self.assertEqual(normalized.schedule_horizon, 3)
        self.assertEqual(len(normalized.slots), 3)
        self.assertEqual(normalized.slots, slots[:3])
        self.assertEqual(padded.schedule_horizon, 6)
        self.assertEqual(len(padded.slots), 6)
        self.assertEqual(again.slots, normalized.slots)
        self.assertEqual(again.objective_key, normalized.objective_key)
        evaluate_critical_step.verify_candidate([2, 0, 1, 0, 0], 1, [1], normalized, 0)

    def test_incomplete_short_prefix_cannot_pass_formal_candidate_verifier(self):
        incomplete = cwp_solver._CandidateSchedule(
            slots=[cwp_solver.Slot(0, 1, "work", 1, 1, 1)],
            makespan=8, assignment_count=1, split_bay_count=0,
            load_deviation=0, reversal_count=0, movement_count=0,
            loads=[1], owners=[{0}], move_time=0,
        )
        self.assertEqual(incomplete.objective_key, (1, 0))
        self.assertFalse(cwp_solver._candidate_passes_independent_verifier(
            [2], 1, [1], incomplete,
        ))

    def test_multislot_move_crossing_completion_is_counted_once(self):
        slots = [
            cwp_solver.Slot(0, 1, "work", 1, 1, 1),
            cwp_solver.Slot(0, 2, "move", 4, 5, None, 1, 1, 2),
            cwp_solver.Slot(1, 1, "idle", 1, 1, None),
            cwp_solver.Slot(1, 2, "move", 5, 6, None, 1, 2, 2),
            cwp_solver.Slot(2, 1, "idle", 1, 1, None),
            cwp_solver.Slot(2, 2, "move", 6, 7, None, 2, 1, 2),
            cwp_solver.Slot(3, 1, "idle", 1, 1, None),
            cwp_solver.Slot(3, 2, "move", 7, 8, None, 2, 2, 2),
        ]
        check = type("Check", (), {
            "slots": slots, "makespan": 1, "schedule_horizon": 4,
            "crane_loads": [1, 0], "reversal_count": 0,
            "movement_count": 1, "schedule_movement_count": 2,
            "move_time": 2,
        })()
        cwp_solver.verify_solution([1, 0, 0, 0, 0, 0, 0, 0], 2, [1], check)

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
                makespan=repaired.completion_time, makespan_lower_bound=1,
                lower_bound_components={}, makespan_proven_optimal=False,
                proven_lexicographic_optimal=False,
                assignment_count=repaired.assignment_count,
                split_bay_count=repaired.split_bay_count,
                load_deviation=repaired.load_deviation,
                reversal_count=repaired.reversal_count,
                movement_count=repaired.completion_movement_count,
                crane_loads=repaired.loads, target_weights=[1],
                schedule_horizon=repaired.schedule_horizon,
                schedule_movement_count=repaired.movement_count,
                move_time=repaired.move_time,
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
            if m > 1:
                self.assertTrue(any(chain[-1] == m - 1 for chain, _ in windows))

    def test_zero_time_shortening_potential_decodes_prepared_schedule(self):
        slots = [
            cwp_solver.Slot(0, 1, "work", 1, 1, 1),
            cwp_solver.Slot(0, 2, "idle", 3, 3, None),
            cwp_solver.Slot(1, 1, "idle", 1, 1, None),
            cwp_solver.Slot(1, 2, "work", 3, 3, 3),
        ]
        candidate = cwp_solver._CandidateSchedule(
            slots=slots, makespan=2, assignment_count=2,
            split_bay_count=0, load_deviation=0, reversal_count=0,
            movement_count=0, loads=[1, 1], owners=[{0}, set(), {1}],
            move_time=0,
        )
        potential, remove_at, deficits = cwp_solver._shortening_potential(
            [1, 0, 1], 2, candidate
        )
        self.assertEqual(potential[:3], (0, 0, 0))
        self.assertIsNotNone(remove_at)
        self.assertFalse(any(deficits))
        shortened = cwp_solver._decode_best_shortening([1, 0, 1], 2, candidate)
        self.assertIsNotNone(shortened)
        self.assertEqual(shortened.makespan, 1)

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

    def test_idle_diagnostics_separates_internal_and_completed_suffix(self) -> None:
        candidate = cwp_solver._candidate_from_history(
            [1, 0, 1], 1, [(1,), (1,), (3,), (3,)],
            move_time=0, preserve_horizon=True,
        )
        report = cwp_solver._idle_diagnostics(candidate, 1)
        self.assertEqual(report["total_internal_idle"], 1)
        self.assertEqual(report["max_internal_idle"], 1)
        self.assertEqual(report["per_crane"][0]["leading_idle"], 0)
        self.assertEqual(report["per_crane"][0]["trailing_idle"], 0)

    def test_terminal_alignment_preserves_route_and_closes_finish_gap(self) -> None:
        work = [4, 0, 0, 1, 0, 1]
        starts = [1, 4]
        rows = [(1, 4), (1, 6), (1, 6), (1, 6), (1, 6)]
        plan = {(t, q): None for t in range(4) for q in range(2)}
        for t in range(4):
            plan[(t, 0)] = 1
        plan[(0, 1)] = 4
        plan[(1, 1)] = 6
        source = cwp_solver._candidate_from_rows_and_work_plan(
            work, 2, rows, plan,
        )

        aligned = cwp_solver._right_shift_terminal_work_blocks(
            work, 2, starts, source,
        )

        self.assertEqual(aligned.objective_key, source.objective_key)
        self.assertEqual(aligned.movement_count, source.movement_count)
        self.assertEqual(aligned.reversal_count, source.reversal_count)
        self.assertEqual(
            cwp_solver._balanced_schedule_metrics(aligned, 2)["finish_times"],
            [4, 4],
        )
        self.assertEqual(
            cwp_solver._continuity_diagnostics(
                aligned, 2, starts
            )["work_revisit_count"],
            0,
        )
        self.assertTrue(cwp_solver._candidate_passes_independent_verifier(
            work, 2, starts, aligned,
        ))

    def test_fixed_window_work_decoder_freezes_external_work(self) -> None:
        work = [3, 0, 2]
        source = cwp_solver._candidate_from_history(
            work, 1, [(1,), (1,), (1,), (3,), (3,), (3,)],
            move_time=0, preserve_horizon=True,
        )
        proposed = cwp_solver._candidate_from_history_with_frozen_work(
            work, 1, [(1,), (1,), (3,), (1,), (3,), (3,)], source,
            [{"crane": 1, "start": 2, "end_exclusive": 4}],
        )
        cwp_solver.verify_solution(
            work, 1, [1], type("Check", (), {
                "slots": proposed.slots, "makespan": proposed.makespan,
                "crane_loads": proposed.loads,
                "reversal_count": proposed.reversal_count,
                "movement_count": proposed.movement_count, "move_time": 0,
            })(),
        )
        source_outside = [
            (slot.time, slot.work_bay)
            for slot in source.slots if slot.state == "work" and slot.time not in (2, 3)
        ]
        proposed_outside = [
            (slot.time, slot.work_bay)
            for slot in proposed.slots if slot.state == "work" and slot.time not in (2, 3)
        ]
        self.assertEqual(proposed_outside, source_outside)

    def test_iterative_descent_continues_after_first_shortening(self) -> None:
        work = [8]
        source = cwp_solver._candidate_from_history(
            work, 1,
            [(1,), (1,), (1,), (1,), (3,), (3,), (1,), (1,), (1,), (1,), (1,)],
            move_time=0, preserve_horizon=True,
        )
        h9 = cwp_solver._candidate_from_history(
            work, 1,
            [(1,), (1,), (1,), (1,), (3,), (1,), (1,), (1,), (1,), (1,)],
            move_time=0, preserve_horizon=True,
        )
        h8 = cwp_solver._candidate_from_history(
            work, 1, [(1,)] * 9, move_time=0, preserve_horizon=True,
        )
        seen_sources = []

        def targeted(_W, _M, candidate, _chain, _window):
            seen_sources.append(candidate.makespan)
            if candidate.completion_time == 10:
                return h9, 1
            if candidate.completion_time == 9:
                return h8, 1
            return None, 1

        with patch.object(
            cwp_solver, "_critical_repair_windows",
            return_value=[((0,), (1, 2))],
        ), patch.object(
            cwp_solver, "_targeted_local_preparation",
            side_effect=targeted,
        ):
            result, evaluated, prepared, first = (
                cwp_solver._cumulative_local_trajectory_repair_iterative(
                    work, 1, [1], source,
                    time.perf_counter() + 0.4, 123,
                    move_time=0, enable_operational_repairs=False,
                )
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.makespan, 8)
        self.assertEqual(prepared.makespan, 8)
        self.assertEqual(first.makespan, 9)
        self.assertEqual(seen_sources, [10, 9])
        self.assertGreaterEqual(evaluated, 2)


if __name__ == "__main__":
    unittest.main()
