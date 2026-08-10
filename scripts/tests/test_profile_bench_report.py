import argparse
import contextlib
import io
import json
import pathlib
import tempfile
import unittest


class ProfileBenchReportTests(unittest.TestCase):
    def test_profile_bench_does_not_persist_setup_cache(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        workflow = (repo / ".github/workflows/profile-bench.yml").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("disk-persistence", workflow)
        self.assertNotIn("LOCALAPPDATA", workflow)

    def test_profile_bench_records_workflow_shard_identity(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        workflow = (repo / ".github/workflows/profile-bench.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn('--benchmark-shard "${{ matrix.group.name }}"', workflow)

    def test_merge_base_policy_reads_narrow_profile_mode_registry(self) -> None:
        from scripts.profile_bench_merge_base_policy import profile_modes_from_modes_rs

        modes_rs = """
const PROFILE_SELECTED_MODES: &[ProfileMode] = &[
    ProfileMode { name: "dense_fp128", run: run_dense },
    ProfileMode { name: "onehot_fp128", run: run_onehot },
];
const PROFILE_ALL_MODES: &[ProfileMode] = &[
    ProfileMode { name: "unrelated", run: run_unrelated },
];
"""

        self.assertEqual(
            profile_modes_from_modes_rs(modes_rs, profile_ci=True),
            {"dense_fp128", "onehot_fp128"},
        )

    def test_plan_case_runs_orders_warmups_then_measured(self) -> None:
        from scripts.profile_bench_report import BenchmarkCaseSpec, ScheduledRun, plan_case_runs

        case = BenchmarkCaseSpec(mode="onehot_fp128", num_vars=24, num_polys=1)
        summary_dir = pathlib.Path("/tmp/bench-root")
        schedule = plan_case_runs("/bin/profile", summary_dir, case, runs=2, warmups=1)

        self.assertEqual(len(schedule), 3)
        self.assertEqual(schedule[0].kind, "warmup")
        self.assertEqual(schedule[1].kind, "measured")
        self.assertEqual(schedule[2].kind, "measured")
        self.assertEqual(schedule[1].run_index, 1)
        self.assertEqual(schedule[2].run_index, 2)
        self.assertEqual(schedule[0].run_dir, summary_dir / case.case_id / "warmup-1")
        self.assertEqual(schedule[1].run_dir, summary_dir / case.case_id / "run-1")
        self.assertEqual(schedule[2].run_dir, summary_dir / case.case_id / "run-2")

    def test_interleaved_schedule_alternates_binaries(self) -> None:
        from scripts.profile_bench_report import BenchmarkCaseSpec, plan_case_runs

        case = BenchmarkCaseSpec(mode="onehot_fp128", num_vars=24, num_polys=1)
        binaries = [
            ("/bin/pr", pathlib.Path("/tmp/pr")),
            ("/bin/base", pathlib.Path("/tmp/base")),
        ]
        plans = [
            plan_case_runs(binary, summary_dir, case, runs=2, warmups=1)
            for binary, summary_dir in binaries
        ]
        self.assertEqual(len({len(plan) for plan in plans}), 1)
        schedule = [run for slot in zip(*plans) for run in slot]

        self.assertEqual(
            [run.binary for run in schedule],
            [
                "/bin/pr",
                "/bin/base",
                "/bin/pr",
                "/bin/base",
                "/bin/pr",
                "/bin/base",
            ],
        )

    def test_configured_cases_rejects_duplicate_case_ids(self) -> None:
        from scripts.profile_bench_report import configured_cases

        args = type(
            "Args",
            (),
            {
                "case": ["onehot_fp128:24:1", "onehot_fp128:24:1"],
                "mode": "onehot_fp128",
                "num_vars": 24,
                "num_polys": 1,
            },
        )()
        with self.assertRaisesRegex(ValueError, "duplicate benchmark case ids"):
            configured_cases(args)

    def test_ingest_tail_summary_fields_parses_wire_and_cap_low_bits(self) -> None:
        from scripts.profile_bench_report import ingest_tail_summary_fields

        summary: dict[str, object] = {}
        ingest_tail_summary_fields(
            summary,
            {
                "final_w_encoding": "terminal_response",
                "tail_log_basis_inner": "6",
                "z_witness_linf_cap": "4096",
                "z_rice_low_bits_wire": "10",
                "z_rice_low_bits_cap": "12",
                "z_bits_per_coord_golomb": "12.50",
            },
        )
        self.assertEqual(summary["z_rice_low_bits_wire"], 10)
        self.assertEqual(summary["z_rice_low_bits_cap"], 12)
        self.assertAlmostEqual(summary["z_bits_per_coord_golomb"], 12.50)
        self.assertEqual(summary["terminal_log_basis"], 6)

    def test_terminal_response_encoding_renders_component_breakdown(self) -> None:
        from scripts.profile_bench_report import render_tail_encoding

        summary = {
            "tail_encoding": "terminal_response",
            "tail_policy": "non_zk_default",
            "tail_num_elems": 96,
            "tail_log_basis_inner": 6,
            "tail_z_prefix_bytes": 8,
            "tail_z_golomb_bytes": 12,
            "tail_z_bytes": 20,
            "tail_z_field_elems": 32,
            "tail_z_ring_elems": 1,
            "tail_e_bytes": 64,
            "tail_e_field_elems": 32,
            "tail_e_ring_elems": 1,
            "tail_t_bytes": 64,
            "tail_t_field_elems": 32,
            "tail_t_ring_elems": 1,
        }

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_tail_encoding(summary)
        report = output.getvalue()

        self.assertIn("quotient-free terminal response", report)
        self.assertIn("inner gadget basis width `6` bits", report)
        self.assertIn("Folded-witness (`z`) segment", report)
        self.assertIn("Opening-digit (`e`) segment", report)
        self.assertIn("Inner-commitment (`t`) segment", report)

    def test_compact_report_renders_terminal_response_component_table(self) -> None:
        from scripts.profile_bench_report import render_terminal_response_components

        case = {
            "mode": "onehot_fp128",
            "num_vars": 32,
            "num_polys": 1,
            "exit_code": 0,
            "tail_encoding": "terminal_response",
            "tail_z_bytes": 20,
            "tail_e_bytes": 64,
            "tail_t_bytes": 96,
            "tail_bytes": 180,
        }

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_terminal_response_components([case])
        report = output.getvalue()

        self.assertIn("Terminal response component breakdown", report)
        self.assertIn("20 bytes", report)
        self.assertIn("64 bytes", report)
        self.assertIn("96 bytes", report)
        self.assertIn("180 bytes", report)
        self.assertIn("sum exactly", report)

    def test_z_fold_encoding_stats_prefers_wire_low_bits(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            'INFO z fold encoding stats label=onehot_fp128 '
            'z_coords=100 witness_linf_cap=4096 rice_low_bits_wire=10 rice_low_bits_cap=12 '
            'bits_per_coord_at_wire=12.5 bits_per_coord_packed=15.0 z_payload_bytes=200\n'
        )
        summary = extract_summary(log, mode="onehot_fp128", num_vars=24, num_polys=1)
        self.assertEqual(summary["z_rice_low_bits_wire"], 10)
        self.assertEqual(summary["z_rice_low_bits_cap"], 12)
        self.assertAlmostEqual(summary["z_bits_per_coord_golomb"], 12.5)

    def test_setup_size_parses_flat_field_count(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            " INFO setup sizes label=onehot_fp128 "
            "num_setup_field_elements=4096 setup_vector_bytes=65536 "
            "setup_ntt_cache_bytes=131072\n"
        )

        summary = extract_summary(log, "onehot_fp128", 24, 1)

        self.assertEqual(summary["num_setup_field_elements"], 4096)

    def test_onehot_commit_schedule_is_recorded(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            " INFO one hot commit schedule sweep=merge block_tile=64 hot_terms=512 "
            "source_count=2 total_blocks=128 workers=8 n_a=5 active_a_cols=64 "
            "ring_dimension=256 estimated_matrix_passes=8 "
            "scratch_budget_per_worker=8388608\n"
        )

        summary = extract_summary(log, "onehot_fp128", 32, 2)

        self.assertEqual(
            summary["onehot_commit_schedules"],
            [
                {
                    "sweep": "merge",
                    "block_tile": 64,
                    "hot_terms": 512,
                    "source_count": 2,
                    "total_blocks": 128,
                    "workers": 8,
                    "n_a": 5,
                    "active_a_cols": 64,
                    "ring_dimension": 256,
                    "estimated_matrix_passes": 8,
                }
            ],
        )

    def test_verify_timings_keep_multi_and_single_thread_modes_separate(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = "\n".join(
            [
                "INFO profile thread pools prove_threads=16 verify_multi_threads=16 "
                "verify_single_threads=1",
                "INFO profile verification start label=onehot_fp128 "
                'verify_mode="multi threaded"',
                "INFO akita batched verify complete elapsed_s=0.007",
                "INFO verify multi threaded OK label=onehot_fp128 elapsed_s=0.008",
                "INFO profile verification start label=onehot_fp128 "
                'verify_mode="single threaded"',
                "INFO akita batched verify complete elapsed_s=0.012",
                "INFO verify single threaded OK label=onehot_fp128 elapsed_s=0.013",
            ]
        )

        summary = extract_summary(log, "onehot_fp128", 32, 1)

        self.assertEqual(summary["prove_threads"], 16)
        self.assertEqual(summary["verify_multi_threads"], 16)
        self.assertEqual(summary["verify_single_threads"], 1)
        self.assertEqual(summary["verification_modes"], "multi_and_single")
        self.assertEqual(summary["verify_total_s"], 0.008)
        self.assertEqual(summary["verify_single_total_s"], 0.013)
        self.assertEqual(summary["verify_akita_s"], 0.007)
        self.assertEqual(summary["verify_single_akita_s"], 0.012)

    def test_legacy_verify_timing_is_the_multi_thread_baseline(self) -> None:
        from scripts.profile_bench_report import extract_summary, missing_required_run_metrics

        summary = extract_summary(
            "INFO verify OK label=onehot_fp128 elapsed_s=0.008\n",
            "onehot_fp128",
            32,
            1,
        )

        self.assertEqual(summary["verify_total_s"], 0.008)
        self.assertNotIn("verify_single_total_s", summary)
        self.assertNotIn("verify_single_total_s", missing_required_run_metrics(summary))

        summary["verification_modes"] = "multi_and_single"
        self.assertIn("verify_single_total_s", missing_required_run_metrics(summary))

    def test_setup_size_converts_merge_base_ring_count_to_flat_fields(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            " INFO setup sizes label=onehot_fp128 "
            "setup_ring_elements=64 setup_vector_bytes=65536 "
            "setup_ntt_cache_bytes=131072\n"
        )

        summary = extract_summary(log, "onehot_fp128", 24, 1)

        self.assertEqual(summary["num_setup_field_elements"], 4096)

    def test_planned_fold_level_parses_physical_geometry(self) -> None:
        from scripts.profile_bench_report import (
            extract_summary,
            planned_group_planner_value,
        )

        log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=2 n_b=3 n_d=4 b_slice_count=4 physical_b_input_width=192 '
            'logical_b_rows=12 complete_b_compression_bytes=3072 '
            'challenge_l1_mass=8 log_basis=5 position_index_bits=7 block_index_bits=3 '
            'num_live_ring_elements_per_claim=768 num_live_blocks=6 block_index_domain_size=8 '
            'num_positions_per_block=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )

        summary = extract_summary(log, mode="onehot_fp128", num_vars=24, num_polys=1)

        self.assertEqual(
            summary["planned_levels"][0],
            {
                "level": 0,
                "d_a": 64,
                "d_b": 32,
                "d_d": 16,
                "n_a": 2,
                "n_b": 3,
                "n_d": 4,
                "b_slice_count": 4,
                "physical_b_input_width": 192,
                "logical_b_rows": 12,
                "complete_b_compression_bytes": 3072,
                "challenge_l1_mass": 8,
                "log_basis_inner": 5,
                "log_basis_outer": 5,
                "log_basis_open": 5,
                "position_index_bits": 7,
                "block_index_bits": 3,
                "num_positions_per_block": 128,
                "num_live_blocks": 6,
                "num_live_ring_elements_per_claim": 768,
                "block_index_domain_size": 8,
                "num_digits_inner": 4,
                "num_digits_outer": 5,
                "num_digits_open": 5,
                "delta_fold": 6,
                "input_witness_len": 1024,
                # Legacy scalar `current_w_len` is not a group breakdown.
                "current_w_len": [],
                "next_w_len": 2048,
                "setup_prefix_natural_field_elements": 0,
                "setup_prefix_padded_field_elements": 0,
                "level_bytes": 4096,
            },
        )
        rendered = planned_group_planner_value(
            {
                **summary["planned_levels"][0],
                "group": "final",
                "group_role": "final",
                "consumer_level": 0,
                "witness_field_elements": 1024,
                "num_digits_fold": 6,
            }
        )
        self.assertIn("B slices: 4", rendered)
        self.assertIn("Physical B cols: 192", rendered)
        self.assertIn("Logical B rows: 12", rendered)
        self.assertIn("Complete B compression: 3,072", rendered)

        legacy_log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 '
            'n_a=2 n_b=3 n_d=4 challenge_l1_mass=8 log_basis=5 '
            'position_index_bits=7 block_index_bits=3 '
            'num_live_ring_elements_per_claim=768 num_live_blocks=6 '
            'block_index_domain_size=8 num_positions_per_block=128 '
            'delta_commit=4 delta_open=5 delta_fold=6 '
            'current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )
        legacy = extract_summary(
            legacy_log, mode="onehot_fp128", num_vars=24, num_polys=1
        )["planned_levels"][0]
        self.assertIsNone(legacy["physical_b_input_width"])
        self.assertIsNone(legacy["complete_b_compression_bytes"])
        legacy_rendered = planned_group_planner_value(
            {
                **legacy,
                "group": "final",
                "group_role": "final",
                "consumer_level": 0,
                "witness_field_elements": 1024,
                "num_digits_fold": 6,
            }
        )
        self.assertIn("Physical B cols: not reported", legacy_rendered)
        self.assertIn("Complete B compression: not reported", legacy_rendered)
        self.assertNotIn("Physical B cols: 0", legacy_rendered)

    def test_planned_fold_level_parses_typed_schedule_field_names(self) -> None:
        from scripts.profile_bench_report import extract_summary

        # The typed-schedule cutover renamed scalar lengths to
        # `input_witness_len`/`output_witness_len`, dropped `level_bytes`, and
        # now emits `current_w_len` as a group breakdown plus setup-prefix sizes.
        log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=2 n_b=3 n_d=4 '
            'challenge_l1_mass=8 log_basis=5 position_index_bits=7 block_index_bits=3 '
            'num_live_ring_elements_per_claim=768 num_live_blocks=6 block_index_domain_size=8 '
            'num_positions_per_block=128 num_digits_inner=4 num_digits_outer=5 num_digits_open=5 '
            'delta_fold=6 input_witness_len=1024 output_witness_len=2048 '
            'current_w_len=pre0=512;final=512 next_w_len=2048 '
            'setup_prefix_natural_field_elements=100 setup_prefix_padded_field_elements=128\n'
        )

        summary = extract_summary(log, mode="onehot_fp128", num_vars=24, num_polys=1)
        level = summary["planned_levels"][0]

        self.assertEqual(
            level["current_w_len"],
            [
                {"group": "pre0", "field_elements": 512},
                {"group": "final", "field_elements": 512},
            ],
        )
        self.assertEqual(level["next_w_len"], 2048)
        self.assertEqual(level["setup_prefix_natural_field_elements"], 100)
        self.assertEqual(level["setup_prefix_padded_field_elements"], 128)
        self.assertEqual(level["num_live_ring_elements_per_claim"], 768)
        self.assertNotIn("level_bytes", level)

    def test_unmatched_planned_group_is_reported(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            "INFO planned fold group label=onehot_fp128 level=3 group=orphan "
            "group_role=setup_offload consumer_level=4 witness_field_elements=64 "
            "d_a=64 d_b=64 d_d=64 n_a=1 n_b=1 n_d=1 "
            "log_basis_inner=3 log_basis_outer=3 log_basis_open=3 "
            "num_digits_inner=1 num_digits_outer=1 num_digits_open=1 "
            "num_digits_fold=1 challenge_l1_mass=8 "
            "num_live_ring_elements_per_claim=1 num_live_blocks=1 "
            "num_positions_per_block=1 block_index_domain_size=1 "
            "setup_prefix_natural_field_elements=64 "
            "setup_prefix_padded_field_elements=64\n"
        )

        summary = extract_summary(log, "onehot_fp128", 32, 1)

        self.assertEqual(
            summary["warnings"],
            ["planned fold groups for L3 have no matching planned fold level"],
        )

    def test_planned_fold_level_normalizes_merge_base_geometry(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 n_a=2 n_b=3 n_d=4 '
            'challenge_l1_mass=8 log_basis=5 m_vars=7 r_vars=3 '
            'num_blocks=8 block_len=2 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )

        summary = extract_summary(log, mode="onehot_fp128", num_vars=24, num_polys=1)
        level = summary["planned_levels"][0]

        self.assertEqual(level["position_index_bits"], 7)
        self.assertEqual(level["block_index_bits"], 3)
        self.assertEqual(level["num_positions_per_block"], 128)
        self.assertEqual(level["num_live_blocks"], 1)
        self.assertEqual(level["num_live_ring_elements_per_claim"], 16)
        self.assertEqual(level["block_index_domain_size"], 8)
        self.assertEqual((level["d_a"], level["d_b"], level["d_d"]), (64, 64, 64))

    def test_planned_fold_level_normalizes_position_bits_merge_base_geometry(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 n_a=2 n_b=3 n_d=4 '
            'challenge_l1_mass=8 log_basis=5 position_bits=7 block_bits=3 '
            'num_blocks=8 block_len=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )

        summary = extract_summary(log, mode="onehot_fp128", num_vars=24, num_polys=1)
        level = summary["planned_levels"][0]

        self.assertEqual(level["position_index_bits"], 7)
        self.assertEqual(level["block_index_bits"], 3)
        self.assertEqual(level["num_positions_per_block"], 128)
        self.assertEqual(level["num_live_blocks"], 1)

    def test_rendered_schedule_uses_names_and_merge_base_deltas(self) -> None:
        from scripts.profile_bench_report import extract_summary, render_fold_details

        current_log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=4 n_b=6 n_d=8 challenge_l1_mass=16 log_basis=6 position_index_bits=7 '
            'block_index_bits=3 num_live_ring_elements_per_claim=768 num_live_blocks=6 '
            'block_index_domain_size=8 num_positions_per_block=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )
        baseline_log = current_log.replace("n_a=4", "n_a=2").replace(
            "level_bytes=4096", "level_bytes=2048"
        )
        current = extract_summary(current_log, "onehot_fp128", 24, 1)["planned_levels"]
        baseline = extract_summary(baseline_log, "onehot_fp128", 24, 1)["planned_levels"]
        proof_log = (
            'INFO proof fold level label=onehot_fp128 level=0 d=64 total_bytes=20 '
            'fold_grind_nonce_bytes=4 stage1_range_image_evaluation_bytes=16 '
            'root_variant=terminal\n'
        )
        proof = extract_summary(proof_log, "onehot_fp128", 24, 1)["proof_levels"]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(current, proof, baseline, proof)
        report = output.getvalue()

        self.assertIn("Fold schedule and proof cost", report)
        self.assertIn("Rings A/B/D: 64 / 32 / 16", report)
        self.assertIn("Rows A/B/D: 4 / 6 / 8", report)
        self.assertIn("<em>Matrix geometry</em>", report)
        self.assertIn(
            "<sub>Merge base</sub><br><strong>Final group</strong><br>"
            "<em>Matrix geometry</em><br>Rings A/B/D: 64 / 32 / 16<br>"
            "Rows A/B/D: 2 / 6 / 8",
            report,
        )
        self.assertNotIn("+100.0% vs base", report)
        self.assertIn("Proof bytes", report)
        self.assertNotIn("Planned fold-level proof bytes", report)
        self.assertNotIn("| M |", report)
        self.assertNotIn("r_pos", report)

    def test_multi_group_root_and_setup_offload_keep_group_parameters(self) -> None:
        from scripts.profile_bench_report import extract_summary, render_fold_details

        group_fields = (
            "consumer_level={consumer} witness_field_elements={witness} "
            "d_a={d_a} d_b=64 d_d=64 n_a={n_a} n_b=1 n_d=1 "
            "log_basis_inner={basis} log_basis_outer={basis} log_basis_open={basis} "
            "num_digits_inner={inner_digits} num_digits_outer={outer_digits} "
            "num_digits_open={open_digits} "
            "num_digits_fold={fold_digits} challenge_l1_mass={l1} "
            "num_live_ring_elements_per_claim={live} num_live_blocks={blocks} "
            "num_positions_per_block={positions} block_index_domain_size={domain} "
            "setup_prefix_natural_field_elements={natural} "
            "setup_prefix_padded_field_elements={padded}"
        )
        log = "\n".join(
            [
                "INFO planned fold group label=onehot_fp128_multi_group_recursive level=0 "
                "group=pre0 group_role=precommitted "
                + group_fields.format(
                    consumer=0,
                    witness=65536,
                    d_a=64,
                    n_a=3,
                    basis=3,
                    inner_digits=1,
                    outer_digits=43,
                    open_digits=43,
                    fold_digits=2,
                    l1=51,
                    live=1024,
                    blocks=4,
                    positions=256,
                    domain=4,
                    natural=0,
                    padded=0,
                ),
                "INFO planned fold group label=onehot_fp128_multi_group_recursive level=0 "
                "group=final group_role=final "
                + group_fields.format(
                    consumer=0,
                    witness=8589934592,
                    d_a=256,
                    n_a=1,
                    basis=3,
                    inner_digits=1,
                    outer_digits=43,
                    open_digits=43,
                    fold_digits=3,
                    l1=23,
                    live=16777216,
                    blocks=512,
                    positions=32768,
                    domain=512,
                    natural=0,
                    padded=0,
                ),
                "INFO planned fold group label=onehot_fp128_multi_group_recursive level=0 "
                "group=setup_to_L1 group_role=setup_offload "
                + group_fields.format(
                    consumer=1,
                    witness=11294208,
                    d_a=256,
                    n_a=2,
                    basis=4,
                    inner_digits=32,
                    outer_digits=32,
                    open_digits=32,
                    fold_digits=3,
                    l1=23,
                    live=65536,
                    blocks=256,
                    positions=256,
                    domain=256,
                    natural=11294208,
                    padded=16777216,
                ),
                "INFO planned fold level label=onehot_fp128_multi_group_recursive level=0 "
                "d=256 d_a=256 d_b=64 d_d=64 n_a=1 n_b=1 n_d=1 "
                "challenge_l1_mass=23 log_basis=3 position_index_bits=15 "
                "block_index_bits=9 num_live_ring_elements_per_claim=16777216 "
                "num_live_blocks=512 block_index_domain_size=512 "
                "num_positions_per_block=32768 delta_commit=1 delta_open=43 delta_fold=3 "
                "input_witness_len=8590000128 output_witness_len=47963968 "
                "current_w_len=pre0:65536;final:8589934592 next_w_len=47963968",
                "INFO proof fold level label=onehot_fp128_multi_group_recursive level=0 "
                "d=256 total_bytes=804 stage3_sumcheck_bytes=800 "
                "fold_grind_nonce_bytes=4 root_variant=terminal",
            ]
        )
        summary = extract_summary(log, "onehot_fp128_multi_group_recursive", 32, 4)
        planned = summary["planned_levels"]
        proof = summary["proof_levels"]
        groups = planned[0]["groups"]

        self.assertEqual([group["group_role"] for group in groups], [
            "precommitted",
            "final",
            "setup_offload",
        ])
        self.assertEqual(groups[0]["d_a"], 64)
        self.assertEqual(groups[1]["d_a"], 256)
        self.assertEqual(groups[2]["consumer_level"], 1)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(planned, proof, planned, proof)
        report = output.getvalue()

        self.assertIn(
            "<strong>Precommit 1</strong><br><em>Matrix geometry</em><br>"
            "Rings A/B/D: 64 / 64 / 64",
            report,
        )
        self.assertIn(
            "<strong>Final group</strong><br><em>Matrix geometry</em><br>"
            "Rings A/B/D: 256 / 64 / 64",
            report,
        )
        self.assertIn(
            "<strong>Setup offload → L1</strong><br><em>Matrix geometry</em><br>"
            "Rings A/B/D: 256 / 64 / 64",
            report,
        )
        self.assertIn(
            "<em>Setup prefix</em><br>Natural → padded: "
            "11,294,208 → 16,777,216",
            report,
        )
        self.assertIn(
            "<strong>Folded output</strong><br>Field elements: 47,963,968",
            report,
        )
        self.assertNotIn("setup fields; relation", report)

    def test_proof_breakdown_omits_zero_components(self) -> None:
        from scripts.profile_bench_report import (
            extract_summary,
            proof_level_component_bytes,
            render_fold_details,
        )

        log = (
            'INFO proof fold level label=onehot_fp128 level=0 d=64 total_bytes=20 '
            'fold_grind_nonce_bytes=4 grind_nonce=3 grind_attempts=4 '
            'stage1_range_image_evaluation_bytes=16 '
            'root_variant=terminal\n'
        )
        levels = extract_summary(log, "onehot_fp128", 24, 1)["proof_levels"]
        self.assertEqual(proof_level_component_bytes(levels[0]), 20)
        planned_log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=4 n_b=6 n_d=8 challenge_l1_mass=16 log_basis=6 position_index_bits=7 '
            'block_index_bits=3 num_live_ring_elements_per_claim=768 num_live_blocks=6 '
            'block_index_domain_size=8 num_positions_per_block=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048\n'
        )
        planned = extract_summary(planned_log, "onehot_fp128", 24, 1)["planned_levels"]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(planned, levels, planned, levels)
        report = output.getvalue()

        self.assertIn("Fold by fold", report)
        self.assertIn("<strong>Stage 1</strong><br>16 bytes", report)
        self.assertIn("<sub>range image 16</sub>", report)
        self.assertNotIn("<strong>Opening</strong>", report)
        self.assertNotIn("<strong>Stage 2</strong>", report)
        self.assertIn("+0.0% vs merge base", report)
        self.assertIn("terminal response", report)
        self.assertIn("Grinding retries", report)
        proof_table_lines = [
            line
            for line in report.splitlines()
            if line.startswith("| Fold | Step |") or line.startswith("| L0 | terminal root |")
        ]
        self.assertEqual(len({line.count("|") for line in proof_table_lines}), 1)

    def test_matrix_splits_metrics_and_embeds_merge_base_deltas(self) -> None:
        from scripts.profile_bench_report import normalize_case_summary, render_matrix_summary

        current = normalize_case_summary(
            {
                "mode": "onehot_fp128",
                "num_vars": 32,
                "num_polys": 1,
                "exit_code": 0,
                "setup_s": 2.0,
                "setup_vector_bytes": 4 * 1024 * 1024,
                "setup_ntt_cache_bytes": 8 * 1024 * 1024,
                "commit_s": 4.0,
                "prove_total_s": 6.0,
                "verify_total_s": 0.008,
                "verify_single_total_s": 0.012,
                "max_rss_kib": 2048,
                "proof_size_bytes": 4096,
                "planned_levels": [{"level": 0, "d_a": 64, "d_b": 64, "d_d": 64}],
            }
        )
        baseline = dict(current)
        for key in (
            "setup_s",
            "setup_vector_bytes",
            "setup_ntt_cache_bytes",
            "commit_s",
            "prove_total_s",
            "verify_total_s",
            "verify_single_total_s",
            "max_rss_kib",
            "proof_size_bytes",
        ):
            baseline[key] = float(current[key]) / 2.0

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_matrix_summary([current], {str(current["case_id"]): baseline})
        report = output.getvalue()

        self.assertEqual(report.count("+100.0%"), 9)
        self.assertNotIn("vs base</sub>", report)
        self.assertNotIn("vs merge base</sub>", report)
        self.assertIn("### Phase time", report)
        self.assertIn("### Memory and setup size", report)
        self.assertIn("### Proof size and protocol shape", report)
        self.assertNotIn("### Protocol shape", report)
        self.assertNotIn("| Status |", report)
        self.assertIn("Setup vector", report)
        self.assertIn("Prepared NTT cache", report)
        self.assertIn("Verify, multi-threaded", report)
        self.assertIn("Verify, single-threaded", report)
        self.assertIn("Fold A/B/D schedule", report)
        self.assertIn("64/64/64", report)
        self.assertIn("4.0 MiB", report)
        self.assertIn("8.0 MiB", report)
        self.assertIn("4,096 bytes", report)
        self.assertIn("Fp128 one\\-hot nv32", report)
        self.assertNotIn("D=64", report)
        self.assertNotIn("Proof B", report)
        self.assertNotIn("Setup Mode", report)
        table_lines = [line for line in report.splitlines() if line.startswith("|")]
        self.assertLessEqual(max(line.count("|") for line in table_lines), 8)
    def test_fold_dimension_schedule_collapses_uniform_suffix(self) -> None:
        from scripts.profile_bench_report import fold_dimension_schedule

        summary = {
            "planned_levels": [
                {"d_a": 256, "d_b": 64, "d_d": 64},
                {"d_a": 64, "d_b": 64, "d_d": 64},
                {"d_a": 64, "d_b": 64, "d_d": 64},
            ]
        }
        self.assertEqual(fold_dimension_schedule(summary), "256/64/64 → 64/64/64")

    def test_adaptive_case_label_omits_ring_dimensions_and_mixed_dimension_config(self) -> None:
        from scripts.profile_bench_report import human_case_label, normalize_case_summary

        summary = normalize_case_summary(
            {
                "mode": "onehot_fp128",
                "num_vars": 32,
                "num_polys": 1,
                "exit_code": 0,
                "planned_levels": [
                    {"level": 0, "d_a": 256, "d_b": 64, "d_d": 64}
                ],
            }
        )

        self.assertEqual(
            human_case_label(summary), "Fp128 one-hot nv32, direct setup check"
        )

    def test_adaptive_multi_group_case_label_omits_ring_dimensions(self) -> None:
        from scripts.profile_bench_report import human_case_label, normalize_case_summary

        summary = normalize_case_summary(
            {
                "mode": "onehot_fp128_multi_group",
                "num_vars": 32,
                "num_polys": 4,
                "exit_code": 0,
                "planned_levels": [
                    {"level": 0, "d_a": 256, "d_b": 64, "d_d": 64}
                ],
            }
        )

        self.assertEqual(
            human_case_label(summary),
            "Fp128 multi-group, direct setup check",
        )

    def test_recursive_singleton_case_label_matches_direct_workload(self) -> None:
        from scripts.profile_bench_report import human_case_label, normalize_case_summary

        summary = normalize_case_summary(
            {
                "mode": "onehot_fp128",
                "num_vars": 36,
                "num_polys": 1,
                "setup_contribution_mode": "recursive",
                "exit_code": 0,
            }
        )

        self.assertEqual(
            human_case_label(summary),
            "Fp128 one-hot nv36, recursive setup check",
        )

    def test_case_label_keeps_non_dimension_topology_variant(self) -> None:
        from scripts.profile_bench_report import human_case_label, normalize_case_summary

        summary = normalize_case_summary(
            {
                "mode": "onehot_fp128_multi_chunk_w4r2",
                "num_vars": 32,
                "num_polys": 1,
                "exit_code": 0,
                "planned_levels": [
                    {"level": 0, "d_a": 256, "d_b": 64, "d_d": 64}
                ],
            }
        )

        self.assertEqual(
            human_case_label(summary),
            "Fp128 one-hot nv32 W4R2, direct setup check",
        )

    def test_multi_group_statement_uses_three_points_and_mixed_arities(self) -> None:
        from scripts.profile_bench_report import (
            benchmark_name,
            normalize_case_summary,
            public_opening_statement,
        )

        summary = normalize_case_summary(
            {
                "mode": "onehot_fp128_multi_group_recursive",
                "num_vars": 32,
                "num_polys": 4,
                "setup_contribution_mode": "recursive",
                "exit_code": 0,
                "planned_levels": [
                    {
                        "level": 0,
                        "groups": [
                            {
                                "group_role": "precommitted",
                                "public_num_vars": 16,
                                "public_num_polynomials": 1,
                            },
                            {
                                "group_role": "precommitted",
                                "public_num_vars": 16,
                                "public_num_polynomials": 1,
                            },
                            {
                                "group_role": "final",
                                "public_num_vars": 32,
                                "public_num_polynomials": 2,
                            },
                        ],
                    }
                ],
            }
        )

        statement = public_opening_statement(summary)
        name = benchmark_name(
            "onehot_fp128_multi_group_recursive", 32, 4, "recursive"
        )
        self.assertIn("one 16 variable polynomial", statement)
        self.assertIn("at its own point", statement)
        self.assertIn("2 32 variable polynomials", statement)
        self.assertEqual(
            name,
            "fp128 multi-group opening with 4 polynomials (recursive setup contribution)",
        )
        self.assertNotIn("same-point", name)

    def test_profile_definitions_separate_ci_shards_from_public_statements(self) -> None:
        from scripts.profile_bench_report import normalize_case_summary, render_profile_definitions

        cases = [
            normalize_case_summary(
                {
                    "mode": "dense_fp32",
                    "num_vars": 26,
                    "num_polys": 1,
                    "benchmark_shard": "1-fp32-base",
                }
            ),
            normalize_case_summary(
                {
                    "mode": "onehot_fp32",
                    "num_vars": 30,
                    "num_polys": 1,
                    "benchmark_shard": "1-fp32-base",
                }
            ),
            normalize_case_summary(
                {
                    "mode": "dense_fp64",
                    "num_vars": 26,
                    "num_polys": 1,
                    "benchmark_shard": "2-fp64-base",
                }
            ),
        ]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_profile_definitions(cases)
        report = output.getvalue()

        shard_section, statement_section = report.split("### Public opening statements")
        self.assertIn("### Benchmark shards", shard_section)
        self.assertIn(
            "| <code>1-fp32-base</code> | Fp32 dense nv26, direct setup check<br>Fp32 one\\-hot nv30, direct setup check |",
            shard_section,
        )
        self.assertIn(
            "| <code>2-fp64-base</code> | Fp64 dense nv26, direct setup check |",
            shard_section,
        )
        self.assertIn("Over Fp32", statement_section)
        self.assertIn("Fp32 dense nv26, direct setup check", statement_section)
        self.assertIn("Over Fp64", statement_section)
        self.assertIn("Fp64 dense nv26, direct setup check", statement_section)

    def test_partial_merge_base_coverage_is_explicit(self) -> None:
        from scripts.profile_bench_report import render_report

        case = {
            "mode": "dense_fp32",
            "num_vars": 26,
            "num_polys": 1,
            "benchmark_shard": "1-fp32-base",
            "exit_code": 1,
            "failure_phase": "prove",
            "error": "fixture failure",
            "setup_s": 1.0,
            "runs": 1,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            current_path = root / "current.json"
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            current_path.write_text(
                json.dumps({"warmups": 0, "cases": [case]}), encoding="utf-8"
            )
            (baseline_dir / "summary.json").write_text(
                json.dumps({"warmups": 0, "cases": []}), encoding="utf-8"
            )
            args = argparse.Namespace(
                summary=str(current_path),
                main_baseline_dir=str(baseline_dir),
                previous_baseline_dir="",
                compact=True,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(render_report(args), 0)
            report = output.getvalue()

        self.assertIn("comparisons are available for `0` of `1` profiles", report)
        self.assertIn("no matching merge-base case", report)
        self.assertNotIn("Each delta below compares", report)

    def test_incomplete_public_opening_groups_fall_back(self) -> None:
        from scripts.profile_bench_report import (
            normalize_case_summary,
            public_opening_groups,
            public_opening_statement,
        )

        summary = normalize_case_summary(
            {
                "mode": "onehot_fp128_multi_group_recursive",
                "num_vars": 32,
                "num_polys": 4,
                "exit_code": 0,
                "planned_levels": [
                    {
                        "level": 0,
                        "groups": [
                            {
                                "group_role": "precommitted",
                                "public_num_vars": 16,
                                "public_num_polynomials": 1,
                            },
                            {
                                "group_role": "final",
                                "public_num_vars": 32,
                                "public_num_polynomials": 1,
                            },
                        ],
                    }
                ],
            }
        )

        self.assertEqual(public_opening_groups(summary), [])
        self.assertEqual(
            summary["warnings"],
            [
                "public opening groups describe 2 of 4 polynomials; "
                "using the generic opening statement"
            ],
        )
        self.assertEqual(
            public_opening_statement(summary),
            "Over Fp128, 4 polynomials are split across independent opening groups.",
        )

    def test_full_report_renders_overhauled_tables(self) -> None:
        from scripts.profile_bench_report import render_report

        level = {
            "level": 0,
            "d_a": 64,
            "d_b": 32,
            "d_d": 16,
            "n_a": 2,
            "n_b": 3,
            "n_d": 4,
            "challenge_l1_mass": 8,
            "log_basis": 5,
            "position_index_bits": 7,
            "block_index_bits": 3,
            "num_positions_per_block": 128,
            "num_live_blocks": 6,
            "num_live_ring_elements_per_claim": 768,
            "block_index_domain_size": 8,
            "delta_commit": 4,
            "delta_open": 5,
            "delta_fold": 6,
            "current_w_len": 1024,
            "next_w_len": 2048,
            "level_bytes": 12,
        }
        proof_level = {
            "level": 0,
            "d": 64,
            "total_bytes": 4,
            "present_byte_fields": ["fold_grind_nonce_bytes"],
            "extension_opening_partials_bytes": 0,
            "extension_opening_sumcheck_bytes": 0,
            "fold_grind_nonce_bytes": 4,
            "opening_payload_bytes": 0,
            "stage1_sumcheck_bytes": 0,
            "stage1_interstage_claims_bytes": 0,
            "stage1_range_image_evaluation_bytes": 0,
            "stage2_sumcheck_bytes": 0,
            "stage3_sumcheck_bytes": 0,
            "next_w_payload_bytes": 0,
            "next_w_eval_bytes": 0,
            "root_variant": "terminal",
        }
        case = {
            "mode": "onehot_fp128",
            "num_vars": 32,
            "num_polys": 1,
            "setup_contribution_mode": "direct",
            "exit_code": 0,
            "setup_s": 2.0,
            "setup_vector_bytes": 4 * 1024 * 1024,
            "setup_ntt_cache_bytes": 8 * 1024 * 1024,
            "commit_s": 3.0,
            "prove_total_s": 4.0,
            "verify_total_s": 0.005,
            "max_rss_kib": 2048,
            "proof_size_bytes": 12,
            "accounted_bytes": 12,
            "akita_fold_bytes": 12,
            "tail_bytes": 0,
            "akita_levels": 1,
            "planned_levels": [level],
            "proof_levels": [proof_level],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            current_path = root / "current.json"
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            payload = {"warmups": 0, "cases": [case]}
            current_path.write_text(json.dumps(payload), encoding="utf-8")
            (baseline_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
            args = argparse.Namespace(
                summary=str(current_path),
                main_baseline_dir=str(baseline_dir),
                previous_baseline_dir="",
                compact=False,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(render_report(args), 0)
            report = output.getvalue()

        self.assertIn("Delta versus merge base", report)
        self.assertIn("unchanged", report)
        self.assertIn("4.0<br><sub>4,194,304 bytes</sub>", report)
        self.assertIn("8.0<br><sub>8,388,608 bytes</sub>", report)
        self.assertIn("Measured result", report)
        self.assertIn("Execution parameters", report)
        self.assertIn("Fold schedule and proof cost", report)
        self.assertIn("Fold by fold", report)
        self.assertIn("Rings A/B/D", report)
        self.assertIn("Proof bytes", report)
        self.assertNotIn("Proof byte components", report)
        self.assertNotIn("Planned fold-level proof bytes", report)
        self.assertNotIn("merge base: Witness:", report)
        self.assertNotIn("merge base: Relation:", report)
        self.assertNotIn("Proof framing", report)

    def test_failed_report_does_not_claim_successful_timing_samples(self) -> None:
        from scripts.profile_bench_report import render_report

        case = {
            "mode": "onehot_fp128",
            "num_vars": 32,
            "num_polys": 1,
            "exit_code": 1,
            "failure_phase": "prove",
            "error": "benchmark process failed",
            "runs": 1,
            "planned_levels": [
                {
                    "level": 0,
                    "d_a": 64,
                    "d_b": 64,
                    "d_d": 64,
                    "n_a": 1,
                    "n_b": 1,
                    "n_d": 1,
                    "challenge_l1_mass": 8,
                    "log_basis_inner": 3,
                    "log_basis_outer": 3,
                    "log_basis_open": 3,
                    "num_digits_inner": 1,
                    "num_digits_outer": 1,
                    "num_digits_open": 1,
                    "delta_fold": 1,
                    "input_witness_len": 64,
                    "current_w_len": [],
                    "next_w_len": 32,
                    "num_live_ring_elements_per_claim": 1,
                    "num_live_blocks": 1,
                    "num_positions_per_block": 1,
                    "block_index_domain_size": 1,
                    "setup_prefix_natural_field_elements": 0,
                    "setup_prefix_padded_field_elements": 0,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            summary_path = pathlib.Path(tmp) / "summary.json"
            summary_path.write_text(
                json.dumps({"warmups": 0, "cases": [case]}), encoding="utf-8"
            )
            args = argparse.Namespace(
                summary=str(summary_path),
                main_baseline_dir="",
                previous_baseline_dir="",
                compact=False,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(render_report(args), 0)
            report = output.getvalue()

        self.assertIn("0 of 1 profiles passed", report)
        self.assertNotIn("Times are medians", report)
        self.assertIn("Fold schedule and proof cost", report)
        self.assertIn("| n/a |", report)
        self.assertIn("Grinding was not measured", report)

    def test_configured_cases_treats_setup_mode_as_case_dimension(self) -> None:
        from scripts.profile_bench_report import configured_cases

        args = type(
            "Args",
            (),
            {
                "case": [
                    "onehot_fp128:36:1:direct",
                    "onehot_fp128:36:1:recursive",
                ],
                "mode": "onehot_fp128",
                "num_vars": 36,
                "num_polys": 1,
            },
        )()

        cases = configured_cases(args)

        self.assertEqual([case.setup_mode for case in cases], ["direct", "recursive"])
        self.assertEqual(
            [case.mode for case in cases],
            ["onehot_fp128", "onehot_fp128"],
        )
        self.assertNotEqual(cases[0].case_id, cases[1].case_id)
        self.assertTrue(cases[1].case_id.endswith("-setup-recursive"))

    def test_nv36_direct_renders_immediately_before_recursive(self) -> None:
        from scripts.profile_bench_report import (
            human_case_label,
            normalize_case_summary,
            report_case_sort_key,
        )

        cases = [
            normalize_case_summary(
                {
                    "mode": "onehot_fp128",
                    "num_vars": 36,
                    "num_polys": 1,
                    "setup_contribution_mode": "direct",
                    "benchmark_shard": "3-fp128-base",
                }
            ),
            normalize_case_summary(
                {
                    "mode": "onehot_fp128",
                    "num_vars": 36,
                    "num_polys": 1,
                    "setup_contribution_mode": "recursive",
                    "benchmark_shard": "3-fp128-base",
                }
            ),
        ]

        ordered = sorted(cases, key=report_case_sort_key)
        self.assertEqual(
            [human_case_label(case) for case in ordered],
            [
                "Fp128 one-hot nv36, direct setup check",
                "Fp128 one-hot nv36, recursive setup check",
            ],
        )

    def test_write_aggregate_summaries_propagates_sibling_failure(self) -> None:
        from scripts.profile_bench_report import (
            BenchmarkCaseSpec,
            ScheduledRun,
            case_status,
            write_aggregate_summaries,
        )

        case = BenchmarkCaseSpec(mode="onehot_fp128", num_vars=24, num_polys=1)
        pr_dir = pathlib.Path("pr-root")
        base_dir = pathlib.Path("base-root")
        ok_summary = {
            "case_id": case.case_id,
            "exit_code": 0,
            "run_index": 1,
            "setup_s": 1.0,
            "commit_s": 2.0,
            "prove_total_s": 3.0,
            "verify_total_s": 4.0,
            "max_rss_kib": 100,
            "proof_size_bytes": 10,
        }
        failed_summary = {
            "case_id": case.case_id,
            "exit_code": 1,
            "run_index": 1,
            "failure_phase": "prove",
            "error": "boom",
            "setup_s": 1.0,
            "commit_s": 2.0,
            "prove_total_s": 3.0,
            "verify_total_s": 4.0,
            "max_rss_kib": 100,
            "proof_size_bytes": 10,
        }
        results = [
            (
                ScheduledRun(
                    "/bin/pr",
                    pr_dir,
                    pr_dir / case.case_id / "run-1",
                    case,
                    "measured",
                    1,
                ),
                ok_summary,
            ),
            (
                ScheduledRun(
                    "/bin/base",
                    base_dir,
                    base_dir / case.case_id / "run-1",
                    case,
                    "measured",
                    1,
                ),
                failed_summary,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            pr_path = pathlib.Path(tmp) / "pr"
            base_path = pathlib.Path(tmp) / "base"
            remapped = []
            for run, summary in results:
                summary_dir = pr_path if run.summary_dir == pr_dir else base_path
                run_dir = summary_dir / run.run_dir.relative_to(run.summary_dir)
                remapped.append(
                    (
                        ScheduledRun(
                            run.binary, summary_dir, run_dir, run.case, run.kind, run.run_index
                        ),
                        summary,
                    )
                )
            write_aggregate_summaries([pr_path, base_path], [case], remapped, warmups=1)

            pr_summary = json.loads((pr_path / "summary.json").read_text(encoding="utf-8"))
            base_summary = json.loads((base_path / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(pr_summary["cases"]), 1)
            self.assertEqual(len(base_summary["cases"]), 1)
            self.assertEqual(case_status(pr_summary["cases"][0]), "fail")
            self.assertEqual(case_status(base_summary["cases"][0]), "fail")
            self.assertIn("paired binary failed", pr_summary["cases"][0]["error"])

    def test_write_aggregate_summaries_preserves_benchmark_shard(self) -> None:
        from scripts.profile_bench_report import (
            BenchmarkCaseSpec,
            ScheduledRun,
            write_aggregate_summaries,
        )

        case = BenchmarkCaseSpec(mode="dense_fp32", num_vars=26, num_polys=1)
        summary = {
            "case_id": case.case_id,
            "exit_code": 0,
            "run_index": 1,
            "setup_s": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp)
            run = ScheduledRun(
                "/bin/profile",
                output_dir,
                output_dir / case.case_id,
                case,
                "measured",
                1,
            )
            write_aggregate_summaries(
                [output_dir],
                [case],
                [(run, summary)],
                warmups=0,
                benchmark_shard="1-fp32-base",
            )
            payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            csv_text = (output_dir / "summary.csv").read_text(encoding="utf-8")

        self.assertEqual(payload["cases"][0]["benchmark_shard"], "1-fp32-base")
        self.assertIn("benchmark_shard", csv_text.splitlines()[0])
        self.assertIn("1-fp32-base", csv_text)

    def test_validate_case_consistency_tolerates_terminal_proof_level(self) -> None:
        from scripts.profile_bench_report import (
            PROOF_LEVEL_BYTE_FIELDS,
            validate_case_consistency,
        )

        def level(index: int) -> dict:
            return {
                "level": index,
                "d_a": 64,
                "d": 64,
                "total_bytes": 0,
                **{field: 0 for field in PROOF_LEVEL_BYTE_FIELDS},
            }

        planned = [level(i) for i in range(5)]
        # The proof carries the planned non-terminal folds plus one trailing
        # terminal level the planner reports separately; that is allowed.
        proof_with_terminal = [level(i) for i in range(6)]
        validate_case_consistency(
            {"planned_levels": planned, "proof_levels": proof_with_terminal}
        )
        # Equal counts (degenerate single/terminal-only proofs) are also allowed.
        validate_case_consistency(
            {"planned_levels": planned, "proof_levels": [level(i) for i in range(5)]}
        )
        # Two extra proof levels is a genuine mismatch and must fail closed.
        with self.assertRaises(ValueError):
            validate_case_consistency(
                {"planned_levels": planned, "proof_levels": [level(i) for i in range(7)]}
            )
        # Fewer proof levels than planned must also fail closed.
        with self.assertRaises(ValueError):
            validate_case_consistency(
                {"planned_levels": planned, "proof_levels": [level(i) for i in range(4)]}
            )


if __name__ == "__main__":
    unittest.main()
