import argparse
import contextlib
import io
import json
import pathlib
import subprocess
import tempfile
import unittest


class ProfileBenchReportTests(unittest.TestCase):
    def test_report_direct_entrypoint_loads_split_modules(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["python3", "scripts/profile_bench_report.py", "--help"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout)

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
            "tail_z_budget_bytes": 16,
            "z_witness_linf_cap": "4096",
            "z_rice_low_bits_wire": 10,
            "z_bits_per_coord_golomb": 3.0,
            "z_bits_per_coord_packed": 13.0,
            "z_packed_hypothetical_bytes": 52,
        }

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_tail_encoding(summary)
        report = output.getvalue()

        self.assertIn(
            "Clear response: `96` coefficients across `z`, `e`, and `t`", report
        )
        self.assertIn("incoming witness uses a basis width of `6` bits", report)
        self.assertIn("| Folded response (`z`) | 20 bytes | 32 | 1 |", report)
        self.assertIn("| Opening values (`e`) | 64 bytes | 32 | 1 |", report)
        self.assertIn("| Inner-commitment values (`t`) | 64 bytes | 32 | 1 |", report)
        self.assertIn("Golomb parameter `10`", report)
        self.assertIn("coefficient limit `4096`", report)
        self.assertIn("`3.00` bits per coefficient", report)
        self.assertNotIn("Wire total", report)
        self.assertNotIn("savings", report)

    def test_proof_table_embeds_terminal_response_components(self) -> None:
        from scripts.profile_bench_report import normalize_case_summary, render_matrix_summary

        case = normalize_case_summary(
            {
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
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_matrix_summary([case], None)
        report = output.getvalue()

        self.assertIn("180 bytes", report)
        self.assertIn("<sub>z 20 · e 64 · t 96</sub>", report)
        self.assertNotIn("Terminal response component breakdown", report)

    def test_z_fold_encoding_stats_prefers_wire_low_bits(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            'INFO z fold encoding stats label=onehot_fp128 '
            'z_coords=100 witness_linf_cap=4096 rice_low_bits_wire=10 rice_low_bits_cap=12 '
            'bits_per_coord_at_wire=12.5 bits_per_coord_packed=15.0 z_payload_bytes=200\n'
        )
        summary = extract_summary(
            log,
            mode="onehot_fp128",
            num_vars=24,
            num_polys=1,
            allow_legacy_relation_mode=True,
        )
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

    def test_timing_and_size_events_parse_inside_profile_root_span(self) -> None:
        from scripts.profile_bench_report import SUMMARY_CSV_COLUMNS, extract_summary

        log = "\n".join(
            [
                " INFO akita_profile_run: statement_prepare label=dense_fp128 elapsed_s=0.125",
                " INFO akita_profile_run: setup_expand label=dense_fp128 elapsed_s=0.25",
                " INFO akita_profile_run: backend_prepare label=dense_fp128 elapsed_s=0.75",
                " INFO akita_profile_run: setup label=dense_fp128 elapsed_s=1.0",
                " INFO akita_profile_run: setup sizes label=dense_fp128 num_setup_field_elements=4096 setup_vector_bytes=65536 setup_ntt_cache_bytes=131072",
                " INFO akita_profile_run: commit label=dense_fp128 elapsed_s=2.0",
                " INFO akita_profile_run: prove label=dense_fp128 elapsed_s=3.0",
                " INFO akita_profile_run: verifier NTT cache size label=dense_fp128 verifier_ntt_cache_bytes=8192",
            ]
        )

        summary = extract_summary(log, "dense_fp128", 28, 1)

        self.assertNotIn("statement_prepare_s", summary)
        self.assertNotIn("statement_prepare_s", SUMMARY_CSV_COLUMNS)
        self.assertEqual(summary["setup_expand_s"], 0.25)
        self.assertEqual(summary["backend_prepare_s"], 0.75)
        self.assertEqual(summary["setup_s"], 1.0)
        self.assertEqual(summary["setup_vector_bytes"], 65536)
        self.assertEqual(summary["commit_s"], 2.0)
        self.assertEqual(summary["prove_total_s"], 3.0)
        self.assertEqual(summary["verifier_ntt_cache_bytes"], 8192)

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

    def test_tracing_optional_integer_accepts_debug_and_structured_values(self) -> None:
        from scripts.profile_bench_report import parse_tracing_optional_int

        self.assertIsNone(parse_tracing_optional_int(None))
        self.assertIsNone(parse_tracing_optional_int("None"))
        self.assertEqual(parse_tracing_optional_int("Some(4096)"), 4096)
        self.assertEqual(parse_tracing_optional_int("4096"), 4096)

    def test_coefficient_packing_report_names_method_and_geometry(self) -> None:
        from scripts.profile_bench_fold_details import challenge_line, opening_method_lines

        params = {
            "opening_method": "subring_coefficient_packing",
            "source_encoding": "canonical_coefficients",
            "extension_degree": 2,
            "d_a": 256,
            "challenge_subring_dimension": 64,
            "packing_factor": 2,
            "packing_partial_width": 128,
            "packing_quotient_width": 128,
            "challenge_count_pm1": 23,
            "challenge_count_pm2": 0,
        }

        self.assertEqual(
            opening_method_lines(params),
            [
                "Subring coefficient packing",
                "Extension degree: k2",
                "Challenge subring: S64",
                "Packing factor: h2",
                "Packed partial width: 128 base-field coefficients",
                "Q_pack width: 128 base-field coefficients",
                "Committed source: canonical coefficient table",
            ],
        )
        self.assertIn("Subring S64 embedded in A ring D256", challenge_line(params))
        tensor = {
            "opening_method": "evaluation_trace",
            "source_encoding": "tensor_subfield_projection",
            "extension_degree": 4,
            "d_a": 256,
        }
        self.assertIn(
            "Committed source: tensor subfield projection (k4)",
            opening_method_lines(tensor),
        )

    def test_planned_packing_log_round_trips_method_source_chunks_and_eor(self) -> None:
        from scripts.profile_bench_fold_details import (
            fold_path_value,
            opening_method_lines,
            render_group_choices,
        )
        from scripts.profile_bench_report import extract_summary

        log = (
            "INFO planned fold group label=onehot_fp128 level=1 group=folded "
            "group_role=folded consumer_level=1 witness_field_elements=1024 "
            "d_a=256 d_b=128 d_d=64 source_encoding=canonical_coefficients "
            "extension_degree=1 "
            "opening_method=subring_coefficient_packing "
            "challenge_subring_dimension=Some(64) packing_factor=Some(4) "
            "packing_partial_width=Some(64) packing_quotient_width=Some(64) "
            "n_a=2 n_b=3 n_d=4 log_basis_inner=5 log_basis_outer=5 "
            "log_basis_open=5 num_digits_inner=4 num_digits_outer=5 "
            "num_digits_open=5 num_digits_fold=6 relation_mode=quotient_lift "
            "num_digits_quotient=26 challenge_l1_mass=23 "
            "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
            "num_positions_per_block=128 block_index_domain_size=8 "
            "setup_prefix_natural_field_elements=0 "
            "setup_prefix_padded_field_elements=0\n"
            "INFO planned fold group label=onehot_fp128 level=1 group=pre0 "
            "group_role=precommitted consumer_level=1 witness_field_elements=512 "
            "d_a=256 d_b=128 d_d=64 source_encoding=canonical_coefficients "
            "extension_degree=1 opening_method=subring_coefficient_packing "
            "challenge_subring_dimension=Some(128) packing_factor=Some(2) "
            "packing_partial_width=Some(128) packing_quotient_width=Some(128) "
            "n_a=2 n_b=3 n_d=4 log_basis_inner=5 log_basis_outer=5 "
            "log_basis_open=5 num_digits_inner=4 num_digits_outer=5 "
            "num_digits_open=5 num_digits_fold=6 relation_mode=quotient_lift "
            "num_digits_quotient=26 challenge_l1_mass=23 "
            "num_live_ring_elements_per_claim=4 num_live_blocks=1 "
            "num_positions_per_block=4 block_index_domain_size=1 "
            "setup_prefix_natural_field_elements=0 "
            "setup_prefix_padded_field_elements=0\n"
            "INFO planned fold level label=onehot_fp128 level=1 d=256 d_a=256 "
            "d_b=128 d_d=64 source_encoding=canonical_coefficients "
            "witness_chunk_count=8 witness_chunk_active=true "
            "witness_chunk_activated_levels=2 "
            "opening_method=subring_coefficient_packing "
            "challenge_subring_dimension=Some(64) packing_factor=Some(4) "
            "packing_partial_width=Some(64) packing_quotient_width=Some(64) "
            "extension_opening_reduction_present=false "
            "extension_opening_reduction_bytes=0 n_a=2 n_b=3 n_d=4 "
            "challenge_l1_mass=23 log_basis=5 position_index_bits=7 block_index_bits=3 "
            "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
            "block_index_domain_size=8 num_positions_per_block=128 delta_commit=4 "
            "delta_open=5 delta_fold=6 relation_mode=quotient_lift "
            "num_digits_quotient=26 input_witness_len=1024 output_witness_len=2048 "
            "current_w_len=folded:1024 next_w_len=2048\n"
        )

        level = extract_summary(log, "onehot_fp128", 24, 1)["planned_levels"][0]
        group, precommit = level["groups"]

        self.assertEqual(level["opening_method"], "subring_coefficient_packing")
        self.assertEqual(level["source_encoding"], "canonical_coefficients")
        self.assertEqual(level["challenge_subring_dimension"], 64)
        self.assertEqual(level["packing_factor"], 4)
        self.assertEqual(level["packing_partial_width"], 64)
        self.assertEqual(level["packing_quotient_width"], 64)
        self.assertEqual(level["witness_chunk_count"], 8)
        self.assertEqual(level["witness_chunk_activated_levels"], 2)
        self.assertTrue(level["witness_chunk_active"])
        self.assertFalse(level["extension_opening_reduction_present"])
        self.assertEqual(level["extension_opening_reduction_bytes"], 0)
        self.assertIn("Witness chunks: 8 · activated levels: 2", fold_path_value(level))
        self.assertEqual(group["opening_method"], "subring_coefficient_packing")
        self.assertEqual(group["source_encoding"], "canonical_coefficients")
        self.assertEqual(group["challenge_subring_dimension"], 64)
        self.assertEqual(group["packing_factor"], 4)
        self.assertEqual(precommit["challenge_subring_dimension"], 128)
        self.assertEqual(precommit["packing_factor"], 2)

        rendered = render_group_choices(
            level["groups"], [], lambda item: "<br>".join(opening_method_lines(item))
        )
        self.assertIn("Challenge subring: S64", rendered)
        self.assertIn("Challenge subring: S128", rendered)

        evaluation_trace_baseline = [dict(item) for item in level["groups"]]
        evaluation_trace_baseline[0]["opening_method"] = "evaluation_trace"
        changed = render_group_choices(
            level["groups"],
            evaluation_trace_baseline,
            lambda item: "<br>".join(opening_method_lines(item)),
        )
        self.assertIn("<sub>Merge base</sub>", changed)
        self.assertIn("Evaluation trace", changed)

    def test_planned_fold_level_parses_physical_geometry(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=2 n_b=3 n_d=4 '
            'challenge_l1_mass=8 log_basis=5 position_index_bits=7 block_index_bits=3 '
            'num_live_ring_elements_per_claim=768 num_live_blocks=6 block_index_domain_size=8 '
            'num_positions_per_block=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )

        summary = extract_summary(
            log,
            mode="onehot_fp128",
            num_vars=24,
            num_polys=1,
            allow_legacy_relation_mode=True,
        )

        self.assertEqual(
            summary["planned_levels"][0],
            {
                "level": 0,
                "d_a": 64,
                "d_b": 32,
                "d_d": 16,
                "a_width": 0,
                "b_width": 0,
                "d_width": 0,
                "n_a": 2,
                "n_b": 3,
                "n_d": 4,
                "security_route": "L-infinity",
                "response_l2_sq_cap": None,
                "b_slice_count": 1,
                "physical_b_input_width": None,
                "logical_b_rows": None,
                "complete_b_compression_bytes": None,
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
                "relation_mode": "quotient_lift",
                "num_digits_quotient": 26,
                "input_witness_len": 1024,
                # Legacy scalar `current_w_len` is not a group breakdown.
                "current_w_len": [],
                "next_w_len": 2048,
                "setup_prefix_natural_field_elements": 0,
                "setup_prefix_padded_field_elements": 0,
                "level_bytes": 4096,
            },
        )

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
            'delta_fold=6 relation_mode=quotient_lift num_digits_quotient=26 '
            'input_witness_len=1024 output_witness_len=2048 '
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

        summary = extract_summary(
            log, "onehot_fp128", 32, 1, allow_legacy_relation_mode=True
        )

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

        summary = extract_summary(
            log,
            mode="onehot_fp128",
            num_vars=24,
            num_polys=1,
            allow_legacy_relation_mode=True,
        )
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

        summary = extract_summary(
            log,
            mode="onehot_fp128",
            num_vars=24,
            num_polys=1,
            allow_legacy_relation_mode=True,
        )
        level = summary["planned_levels"][0]

        self.assertEqual(level["position_index_bits"], 7)
        self.assertEqual(level["block_index_bits"], 3)
        self.assertEqual(level["num_positions_per_block"], 128)
        self.assertEqual(level["num_live_blocks"], 1)

    def test_rendered_schedule_uses_names_and_merge_base_deltas(self) -> None:
        from scripts.profile_bench_report import extract_summary, render_fold_details

        current_log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=4 n_b=6 n_d=8 response_l2_sq_cap=Some(100) challenge_l1_mass=53 '
            'challenge_count_pm1=31 challenge_count_pm2=11 '
            'challenge_operator_norm_threshold=Some(18) log_basis=6 position_index_bits=7 '
            'block_index_bits=3 num_live_ring_elements_per_claim=768 num_live_blocks=6 '
            'block_index_domain_size=8 num_positions_per_block=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )
        baseline_log = current_log.replace("n_a=4", "n_a=2").replace(
            "level_bytes=4096", "level_bytes=2048"
        )
        current = extract_summary(
            current_log,
            "onehot_fp128",
            24,
            1,
            allow_legacy_relation_mode=True,
        )["planned_levels"]
        baseline = extract_summary(
            baseline_log,
            "onehot_fp128",
            24,
            1,
            allow_legacy_relation_mode=True,
        )["planned_levels"]
        proof_log = (
            'INFO proof fold level label=onehot_fp128 level=0 d=64 total_bytes=20 '
            'fold_grind_nonce_bytes=4 stage1_range_image_evaluation_bytes=16 '
            'root_variant=terminal\n'
        )
        proof = extract_summary(proof_log, "onehot_fp128", 24, 1)["proof_levels"]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(current, proof, None, baseline, proof, None)
        report = output.getvalue()

        self.assertIn("Fold schedule and proof cost", report)
        self.assertIn("A commitment: ring D64 · module rank 4", report)
        self.assertIn("B commitment: ring D32 · module rank 6", report)
        self.assertIn("D opening: ring D16 · module rank 8", report)
        self.assertIn("Sum of squared coefficients (L2)", report)
        self.assertIn("Sum of squared coefficients (L2): ≤ 100", report)
        self.assertIn(
            "Ring D64 · shell 31 at ±1 and 11 at ±2 · operator norm threshold 18",
            report,
        )
        self.assertIn("<em>Commitment matrices used at this fold</em>", report)
        self.assertIn("z response: 4 input digits", report)
        self.assertIn("e opening: 5 digits", report)
        self.assertIn("t matrix image: 5 digits", report)
        self.assertIn("r shared quotient: 22 digits", report)
        self.assertIn(
            "<sub>Merge base</sub><br><strong>Final group</strong><br>"
            "<em>Opening method</em><br>Evaluation trace<br>"
            "Opening width: full A ring D64 base-field coefficients<br>"
            "<em>Commitment matrices used at this fold</em><br>"
            "A commitment: ring D64 · module rank 2",
            report,
        )
        self.assertNotIn("+100.0% vs base", report)
        self.assertIn("Proof bytes", report)
        self.assertNotIn("Planned fold-level proof bytes", report)
        self.assertNotIn("| M |", report)
        self.assertNotIn("r_pos", report)

    def test_new_display_fields_do_not_create_false_plan_deltas(self) -> None:
        from scripts.profile_bench_report import extract_summary, render_fold_details

        common = (
            "INFO planned fold level label=onehot_fp128 level=0 d=64 "
            "d_a=64 d_b=64 d_d=64 n_a=3 n_b=2 n_d=2 "
            "challenge_l1_mass=53 log_basis_inner=5 log_basis_outer=6 "
            "log_basis_open=6 position_index_bits=7 block_index_bits=3 "
            "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
            "block_index_domain_size=8 num_positions_per_block=128 "
            "num_digits_inner=1 num_digits_outer=22 num_digits_open=22 "
            "delta_fold=2 relation_mode=quotient_lift num_digits_quotient=22 "
            "input_witness_len=4096 output_witness_len=2048 "
            "current_w_len=final=4096 next_w_len=2048"
        )
        current_log = common + " a_width=128 b_width=264 d_width=132\n"
        baseline_log = common + "\n"
        proof_log = (
            "INFO proof fold level label=onehot_fp128 level=0 d=64 "
            "total_bytes=4 fold_grind_nonce_bytes=4\n"
        )
        current = extract_summary(current_log, "onehot_fp128", 24, 1)["planned_levels"]
        baseline = extract_summary(baseline_log, "onehot_fp128", 24, 1)["planned_levels"]
        proof = extract_summary(proof_log, "onehot_fp128", 24, 1)["proof_levels"]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(current, proof, None, baseline, proof, None)
        report = output.getvalue()

        self.assertIn("input width 128", report)
        self.assertIn("r shared quotient: 22 digits", report)
        self.assertNotIn("<sub>Merge base</sub>", report)

    def test_mixed_relation_schedule_reports_cutover_without_reduced_quotient_rows(self) -> None:
        from scripts.profile_bench_report import extract_summary, render_fold_details

        def level(index: int, relation_mode: str, quotient_digits: int) -> str:
            return (
                "INFO planned fold level label=onehot_fp128 "
                f"level={index} d=64 d_a=64 d_b=64 d_d=64 "
                "n_a=3 n_b=2 n_d=2 challenge_l1_mass=53 "
                "log_basis_inner=5 log_basis_outer=6 log_basis_open=6 "
                "position_index_bits=7 block_index_bits=3 "
                "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
                "block_index_domain_size=8 num_positions_per_block=128 "
                "num_digits_inner=1 num_digits_outer=22 num_digits_open=22 "
                f"delta_fold=2 relation_mode={relation_mode} "
                f"num_digits_quotient={quotient_digits} "
                f"input_witness_len={4096 >> index} output_witness_len={2048 >> index} "
                f"current_w_len=folded={4096 >> index} next_w_len={2048 >> index}"
            )

        summary = extract_summary(
            "\n".join(
                [
                    level(0, "quotient_lift", 22),
                    level(1, "quotient_lift", 22),
                    level(2, "reduced_evaluation", 0),
                ]
            ),
            "onehot_fp128",
            24,
            1,
        )
        planned = summary["planned_levels"]
        self.assertEqual(
            [entry["relation_mode"] for entry in planned],
            ["quotient_lift", "quotient_lift", "reduced_evaluation"],
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(planned, [], None, None, None, None)
        report = output.getvalue()

        self.assertEqual(report.count("r shared quotient:"), 2)
        self.assertIn("Ring relation: Quotient lift (shared R rows)", report)
        self.assertIn("Ring relation: Reduced evaluation (no R rows)", report)

    def test_current_relation_mode_parser_rejects_missing_and_unknown_values(self) -> None:
        from scripts.profile_bench_report import extract_summary

        common = (
            "INFO planned fold level label=onehot_fp128 level=0 d=64 "
            "d_a=64 d_b=64 d_d=64 n_a=3 n_b=2 n_d=2 "
            "challenge_l1_mass=53 log_basis_inner=5 log_basis_outer=6 "
            "log_basis_open=6 position_index_bits=7 block_index_bits=3 "
            "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
            "block_index_domain_size=8 num_positions_per_block=128 "
            "num_digits_inner=1 num_digits_outer=22 num_digits_open=22 "
            "delta_fold=2 num_digits_quotient=22 input_witness_len=4096 "
            "output_witness_len=2048 current_w_len=final=4096 next_w_len=2048"
        )
        with self.assertRaisesRegex(ValueError, "missing relation_mode"):
            extract_summary(common, "onehot_fp128", 24, 1)
        with self.assertRaisesRegex(ValueError, "invalid relation_mode"):
            extract_summary(
                common + " relation_mode=quotient_typo", "onehot_fp128", 24, 1
            )

        legacy = extract_summary(
            common,
            "onehot_fp128",
            24,
            1,
            allow_legacy_relation_mode=True,
        )
        self.assertEqual(legacy["planned_levels"][0]["relation_mode"], "quotient_lift")

    def test_relation_mode_parser_rejects_nonzero_reduced_rows_and_group_mismatch(self) -> None:
        from scripts.profile_bench_report import extract_summary

        reduced_level = (
            "INFO planned fold level label=onehot_fp128 level=0 d=64 "
            "d_a=64 d_b=64 d_d=64 n_a=3 n_b=2 n_d=2 "
            "challenge_l1_mass=53 log_basis_inner=5 log_basis_outer=6 "
            "log_basis_open=6 position_index_bits=7 block_index_bits=3 "
            "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
            "block_index_domain_size=8 num_positions_per_block=128 "
            "num_digits_inner=1 num_digits_outer=22 num_digits_open=22 "
            "delta_fold=2 relation_mode=reduced_evaluation num_digits_quotient=22 "
            "input_witness_len=4096 output_witness_len=2048 "
            "current_w_len=final=4096 next_w_len=2048"
        )
        with self.assertRaisesRegex(ValueError, "num_digits_quotient=22"):
            extract_summary(reduced_level, "onehot_fp128", 24, 1)

        group = (
            "INFO planned fold group label=onehot_fp128 level=0 group=final "
            "group_role=final consumer_level=0 witness_field_elements=4096 "
            "d_a=64 d_b=64 d_d=64 n_a=3 n_b=2 n_d=2 "
            "log_basis_inner=5 log_basis_outer=6 log_basis_open=6 "
            "num_digits_inner=1 num_digits_outer=22 num_digits_open=22 "
            "num_digits_fold=2 relation_mode=reduced_evaluation "
            "num_digits_quotient=0 challenge_l1_mass=53 "
            "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
            "num_positions_per_block=128 block_index_domain_size=8 "
            "setup_prefix_natural_field_elements=0 "
            "setup_prefix_padded_field_elements=0\n"
        )
        quotient_level = reduced_level.replace(
            "relation_mode=reduced_evaluation num_digits_quotient=22",
            "relation_mode=quotient_lift num_digits_quotient=22",
        )
        with self.assertRaisesRegex(ValueError, "expected 'quotient_lift'"):
            extract_summary(group + quotient_level, "onehot_fp128", 24, 1)

    def test_verifier_relation_phase_timings_are_parsed_and_rendered(self) -> None:
        from scripts.profile_bench_report import (
            extract_summary,
            render_relation_phase_timings,
        )

        log = (
            "INFO verifier relation phase timing label=onehot_fp128 "
            "verify_mode=single_threaded relation_mode=reduced "
            "phase=structured_groups calls=3 mean_elapsed_nanos=2000000 "
            "total_elapsed_nanos=6000000\n"
        )
        summary = extract_summary(
            log, "onehot_fp128", 24, 1, allow_legacy_relation_mode=True
        )
        summary["verify_single_total_s"] = 0.012
        timing = summary["relation_phase_timings"][0]
        self.assertEqual(timing["verify_mode"], "single threaded")
        self.assertEqual(timing["relation_mode"], "reduced")
        self.assertEqual(timing["total_elapsed_nanos"], 6_000_000)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_relation_phase_timings(summary)
        report = output.getvalue()
        self.assertIn("Reduced evaluation", report)
        self.assertIn("Structured groups", report)
        self.assertIn("single threaded `12.0ms`", report)

    def test_terminal_fold_reports_its_full_geometry(self) -> None:
        from scripts.profile_bench_report import (
            extract_summary,
            planned_terminal_planner_value,
            render_fold_details,
        )

        log = "\n".join(
            [
                "INFO planned fold level label=onehot_fp128 level=0 d=64 "
                "d_a=64 d_b=64 d_d=64 a_width=128 b_width=64 d_width=64 "
                "n_a=4 n_b=2 n_d=2 challenge_l1_mass=53 "
                "challenge_count_pm1=31 challenge_count_pm2=11 "
                "log_basis_inner=5 log_basis_outer=6 log_basis_open=6 "
                "position_index_bits=7 block_index_bits=3 "
                "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
                "block_index_domain_size=8 num_positions_per_block=128 "
                "num_digits_inner=1 num_digits_outer=22 num_digits_open=22 "
                "delta_fold=2 num_digits_quotient=22 input_witness_len=4096 "
                "output_witness_len=2048 current_w_len=final=4096 next_w_len=2048",
                "INFO planned terminal state label=onehot_fp128 level=1 "
                "input_witness_len=2048 d_a=64 n_a=3 inner_width=128 "
                "log_basis_inner=5 num_digits_inner=1 fold_log_basis=6 "
                "fold_digit_count=2 challenge_l1_mass=53 "
                "challenge_count_pm1=31 challenge_count_pm2=11 "
                "challenge_operator_norm_threshold=Some(18) "
                "response_l2_sq_cap=Some(633237013) "
                "z_linf_cap=None "
                "num_live_ring_elements_per_claim=1908 "
                "num_positions_per_block=256 num_live_blocks=8 "
                "block_index_domain_size=8",
                "INFO proof fold level label=onehot_fp128 level=0 d=64 "
                "total_bytes=4 fold_grind_nonce_bytes=4",
                "INFO proof fold level label=onehot_fp128 level=1 d=64 "
                "total_bytes=564 extension_opening_partials_bytes=64 "
                "extension_opening_sumcheck_bytes=480 "
                "extension_opening_final_claims_bytes=16 fold_grind_nonce_bytes=4 "
                "root_variant=terminal",
            ]
        )
        summary = extract_summary(
            log, "onehot_fp128", 24, 1, allow_legacy_relation_mode=True
        )
        terminal = summary["terminal_plan"]
        self.assertEqual(terminal["level"], 1)
        self.assertEqual(terminal["d_a"], 64)
        self.assertEqual(terminal["n_a"], 3)
        self.assertEqual(terminal["inner_width"], 128)
        terminal_block = planned_terminal_planner_value(terminal)
        self.assertNotIn("B commitment", terminal_block)
        self.assertNotIn("D opening", terminal_block)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(
                summary["planned_levels"],
                summary["proof_levels"],
                terminal,
                None,
                None,
                None,
            )
        report = output.getvalue()

        self.assertIn(
            "A commitment: ring D64 · input width 128 · module rank 3", report
        )
        self.assertIn(
            "z response: 1 input digit (basis bits 5) × "
            "2 response digits (basis bits 6)",
            report,
        )
        self.assertIn(
            "Ring D64 · shell 31 at ±1 and 11 at ±2 · operator norm threshold 18",
            report,
        )
        self.assertEqual(report.count("Maximum coefficient magnitude (Linf)"), 1)
        self.assertIn("Sum of squared coefficients (L2): ≤ 633,237,013", report)
        self.assertIn("<em>Input from L0</em><br>Field elements: 2,048", report)
        self.assertIn("Clear z, e, and t terminal response", report)
        self.assertIn("final claims 16", report)
        self.assertNotIn("| L1 | terminal fold | — | — |", report)

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
        summary = extract_summary(
            log,
            "onehot_fp128_multi_group_recursive",
            32,
            4,
            allow_legacy_relation_mode=True,
        )
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
            render_fold_details(planned, proof, None, planned, proof, None)
        report = output.getvalue()

        self.assertIn(
            "<strong>Precommit 1</strong><br>"
            "<em>Opening method</em><br>Evaluation trace<br>"
            "Opening width: full A ring D64 base-field coefficients<br>"
            "<em>Commitment matrices used at this fold</em><br>"
            "A commitment: ring D64 · module rank 3",
            report,
        )
        self.assertIn(
            "<strong>Final group</strong><br>"
            "<em>Opening method</em><br>Evaluation trace<br>"
            "Opening width: full A ring D256 base-field coefficients<br>"
            "<em>Commitment matrices used at this fold</em><br>"
            "A commitment: ring D256 · module rank 1",
            report,
        )
        self.assertIn(
            "<strong>Setup offload → L1</strong><br>"
            "<em>Opening method</em><br>Evaluation trace<br>"
            "Opening width: full A ring D256 base-field coefficients<br>"
            "<em>Commitment matrices used at this fold</em><br>"
            "A commitment: ring D256 · module rank 2",
            report,
        )
        self.assertIn(
            "<em>Setup prefix</em><br>Natural → padded: "
            "11,294,208 → 16,777,216",
            report,
        )
        self.assertIn(
            "<strong>Output to L1</strong><br>Field elements: 47,963,968",
            report,
        )
        self.assertIn("Maximum coefficient magnitude (Linf)", report)
        self.assertEqual(report.count("r shared quotient:"), 1)
        self.assertNotIn("setup fields; relation", report)

    def test_proof_breakdown_omits_zero_components(self) -> None:
        from scripts.profile_bench_report import (
            extract_summary,
            proof_level_component_bytes,
            render_fold_details,
        )

        log = (
            'INFO proof fold level label=onehot_fp128 level=0 d=64 total_bytes=28 '
            'fold_grind_nonce_bytes=4 grind_nonce=3 grind_attempts=4 '
            'stage1_range_image_evaluation_bytes=16 '
            'stage1_norm_proof_bytes=8 response_l2_sq=Some(14) '
            'root_variant=terminal\n'
        )
        levels = extract_summary(log, "onehot_fp128", 24, 1)["proof_levels"]
        self.assertEqual(proof_level_component_bytes(levels[0]), 28)
        self.assertEqual(levels[0]["response_l2_sq"], 14)
        planned_log = (
            'INFO planned fold level label=onehot_fp128 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=4 n_b=6 n_d=8 response_l2_sq_cap=Some(100) challenge_l1_mass=53 '
            'challenge_count_pm1=31 challenge_count_pm2=11 '
            'challenge_operator_norm_threshold=Some(18) log_basis=6 position_index_bits=7 '
            'block_index_bits=3 num_live_ring_elements_per_claim=768 num_live_blocks=6 '
            'block_index_domain_size=8 num_positions_per_block=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048\n'
        )
        planned = extract_summary(
            planned_log,
            "onehot_fp128",
            24,
            1,
            allow_legacy_relation_mode=True,
        )["planned_levels"]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(planned, levels, None, planned, levels, None)
        report = output.getvalue()

        self.assertIn("Fold by fold", report)
        self.assertIn("<strong>Stage 1</strong><br>24 bytes", report)
        self.assertIn("range image 16", report)
        self.assertIn("L2 norm proof 8", report)
        self.assertIn("Sum of squared response coefficients", report)
        self.assertIn("14 ≤ cap 100", report)
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

    def test_grind_attempts_are_truthful_and_nonzero(self) -> None:
        from scripts.profile_bench_report import extract_summary

        derived_log = (
            "INFO proof fold level label=onehot_fp128_d64 level=0 d=64 total_bytes=4 "
            "fold_grind_nonce_bytes=4 grind_nonce=0\n"
        )
        level = extract_summary(derived_log, "onehot_fp128_d64", 24, 1)[
            "proof_levels"
        ][0]
        self.assertEqual(level["grind_attempts"], 1)

        impossible_log = derived_log.replace(
            "grind_nonce=0", "grind_nonce=0 grind_attempts=0"
        )
        with self.assertRaisesRegex(ValueError, "accepted nonce plus one"):
            extract_summary(impossible_log, "onehot_fp128_d64", 24, 1)

    def test_grinding_plan_reports_exact_bits_by_fold_and_query(self) -> None:
        from scripts.profile_bench_report import extract_summary, render_fold_details

        log = "\n".join(
            [
                "INFO proof summary label=onehot_fp128 levels=1 proof_size_bytes=107 "
                "accounted_bytes=107 akita_fold_bytes=100 nonce_stream_bytes=7 tail_bytes=0",
                "INFO grinding plan summary label=onehot_fp128 nominal_capacity_bits=256 "
                "total_nonce_bits=54 nonce_stream_bytes=7 padding_bits=2 run_count=5 "
                "expanded_query_count=12",
                "INFO grinding plan run label=onehot_fp128 run_index=0 level=0 "
                "component=fold_response query=response_search protocol=none stage=None "
                "round=None group=None kind=fold_response loss_factor=0 grind_bits=0 "
                "nonce_bits=12 multiplicity=1 run_nonce_bits=12",
                "INFO grinding plan run label=onehot_fp128 run_index=1 level=0 "
                "component=stage1 query=sumcheck_round protocol=stage1 stage=Some(0) "
                "round=Some(2) group=None kind=proof_of_work loss_factor=4 grind_bits=3 "
                "nonce_bits=14 multiplicity=1 run_nonce_bits=14",
                "INFO grinding plan run label=onehot_fp128 run_index=2 level=0 "
                "component=stage1 query=sumcheck_round protocol=stage1 stage=Some(0) "
                "round=Some(3) group=None kind=proof_of_work loss_factor=4 grind_bits=3 "
                "nonce_bits=14 multiplicity=1 run_nonce_bits=14",
                "INFO grinding plan run label=onehot_fp128 run_index=3 level=0 "
                "component=stage1 query=sumcheck_round protocol=stage1 stage=Some(0) "
                "round=Some(4) group=None kind=proof_of_work loss_factor=4 grind_bits=3 "
                "nonce_bits=14 multiplicity=1 run_nonce_bits=14",
                "INFO grinding plan run label=onehot_fp128 run_index=4 level=0 "
                "component=fold_challenge query=challenge_coordinates protocol=none "
                "stage=None round=None group=Some(1) kind=fold_challenge_coordinates "
                "loss_factor=0 grind_bits=0 nonce_bits=0 multiplicity=8 run_nonce_bits=0",
                "INFO proof fold level label=onehot_fp128 level=0 d=64 total_bytes=100 "
                "fold_grind_nonce_bytes=0 grind_nonce=0 grind_attempts=1 root_variant=terminal",
            ]
        )
        summary = extract_summary(log, "onehot_fp128", 24, 1)
        plan = summary["grinding_plan"]

        self.assertEqual(summary["nonce_stream_bits"], 54)
        self.assertEqual(summary["nonce_stream_padding_bits"], 2)
        self.assertEqual(plan["runs"][1]["stage"], 0)
        self.assertEqual(plan["runs"][1]["round"], 2)
        self.assertEqual(plan["runs"][4]["group"], 1)

        baseline_proof = [{"level": 0, "fold_grind_nonce_bytes": 4}]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(
                [],
                summary["proof_levels"],
                None,
                None,
                baseline_proof,
                None,
                plan,
                None,
            )
        report = output.getvalue()

        self.assertIn("Transcript grinding bits", report)
        self.assertIn("Proof-global packed nonce stream", report)
        self.assertIn("54<br><sub>Merge base</sub><br>12", report)
        self.assertIn("2<br><sub>Merge base</sub><br>20", report)
        self.assertIn(
            "Stage 1 | sumcheck stage 0, rounds 2 to 4 | 4 | 3 | 14 | 3 | 42",
            report,
        )
        self.assertEqual(report.count("Stage 1 | sumcheck stage 0"), 1)
        self.assertNotIn("challenge coordinates group 1", report)
        self.assertIn("8 plan queries across 1 entry", report)
        self.assertIn("because they require no nonce bits", report)
        self.assertIn("**L0 subtotal**", report)
        self.assertIn("rounds are shown as ranges", report)
        self.assertIn("rounds the stream to bytes once", report)
        self.assertIn("stored in a 32 bit field", report)
        self.assertIn("12 bit fold response", report)

    def test_grinding_round_groups_split_when_security_terms_change(self) -> None:
        from scripts.profile_bench_fold_details import aggregate_grinding_runs

        common = {
            "level": 0,
            "component": "stage1",
            "query": "sumcheck_round",
            "protocol": "stage1",
            "stage": 0,
            "group": None,
            "kind": "proof_of_work",
            "loss_factor": 4,
            "nonce_bits": 14,
            "multiplicity": 1,
            "run_nonce_bits": 14,
        }
        runs = [
            {**common, "round": 0, "grind_bits": 3},
            {**common, "round": 1, "grind_bits": 4},
        ]

        grouped = aggregate_grinding_runs(runs)

        self.assertEqual(len(grouped), 2)
        self.assertEqual([run["grind_bits"] for run in grouped], [3, 4])

    def test_l2_grinding_observations_survive_sample_aggregation(self) -> None:
        from scripts.profile_bench_report import (
            combine_case_run_summaries,
            extract_summary,
            render_fold_details,
        )

        planned = (
            "INFO planned fold level label=onehot_fp128_d64 level=5 d=64 d_a=64 d_b=64 "
            "d_d=64 n_a=4 n_b=4 n_d=4 response_l2_sq_cap=Some(4294967296) "
            "challenge_l1_mass=8 log_basis=5 position_index_bits=7 block_index_bits=3 "
            "num_live_ring_elements_per_claim=768 num_live_blocks=6 "
            "block_index_domain_size=8 num_positions_per_block=128 delta_commit=4 "
            "delta_open=5 delta_fold=6 current_w_len=1024 next_w_len=2048\n"
        )
        summaries = []
        for run_index, nonce in enumerate((0, 2), start=1):
            proof = (
                "INFO proof fold level label=onehot_fp128_d64 level=5 d=64 total_bytes=4 "
                f"fold_grind_nonce_bytes=4 grind_nonce={nonce} grind_attempts={nonce + 1} "
                f"response_l2_sq=Some({80 + run_index})\n"
            )
            summary = extract_summary(
                planned + proof,
                "onehot_fp128_d64",
                24,
                1,
                allow_legacy_relation_mode=True,
            )
            summary["run_index"] = run_index
            summary["exit_code"] = 0
            summaries.append(summary)

        combined = combine_case_run_summaries(summaries)
        observation = combined["l2_grind_observations"][0]
        self.assertEqual(observation["samples"], 2)
        self.assertEqual(observation["attempts"], 4)
        self.assertEqual(observation["rejected_attempts"], 2)
        self.assertEqual(observation["accepted_nonces"], [0, 2])
        self.assertEqual(observation["response_l2_sq_values"], [81, 82])
        self.assertEqual(observation["maximum_cap_utilization"], 82 / 4_294_967_296)
        self.assertEqual(observation["observed_failure_rate"], 0.5)
        self.assertEqual(
            combined["grind_retry_observations"],
            [{"level": 5, "retries": [0, 2]}],
        )
        self.assertEqual(
            combined["samples"][1]["l2_grind_observations"][0]["accepted_nonce"], 2
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_fold_details(
                combined["planned_levels"],
                combined["proof_levels"],
                None,
                None,
                None,
                None,
            )
        report = output.getvalue()
        self.assertNotIn("L2 cap grinding observations", report)
        self.assertNotIn("50.00%", report)

    def test_terminal_l2_response_is_joined_to_its_scheduled_cap(self) -> None:
        from scripts.profile_bench_report import extract_summary, l2_grind_observations_for_run

        log = (
            "INFO planned terminal state label=onehot_fp128_d64 level=6 "
            "input_witness_len=512 d_a=64 n_a=6 inner_width=64 "
            "log_basis_inner=5 num_digits_inner=1 fold_log_basis=6 "
            "fold_digit_count=2 response_l2_sq_cap=Some(100)\n"
            "INFO proof fold level label=onehot_fp128_d64 level=6 d=64 "
            "total_bytes=4 fold_grind_nonce_bytes=4 grind_nonce=2 "
            "grind_attempts=3 response_l2_sq=Some(90)\n"
        )
        summary = extract_summary(log, "onehot_fp128_d64", 24, 1)
        summary["run_index"] = 1
        summary["exit_code"] = 0

        self.assertEqual(
            l2_grind_observations_for_run(summary),
            [
                {
                    "level": 6,
                    "response_l2_sq_cap": 100,
                    "response_l2_sq": 90,
                    "cap_utilization": 0.9,
                    "accepted_nonce": 2,
                    "attempts": 3,
                    "rejected_attempts": 2,
                }
            ],
        )

    def test_failed_l2_run_preserves_partial_sample_without_grind_diagnostics(self) -> None:
        from scripts.profile_bench_report import combine_case_run_summaries

        failed = {
            "run_index": 1,
            "exit_code": 1,
            "error": "prover failed before measured L2 level",
            "planned_levels": [
                {
                    "level": 5,
                    "security_route": "L2",
                    "response_l2_sq_cap": 100,
                }
            ],
            "proof_levels": [
                {
                    "level": 0,
                    "grind_nonce_val": 0,
                    "grind_attempts": 1,
                }
            ],
        }

        combined = combine_case_run_summaries([failed])

        self.assertEqual(combined["exit_code"], 1)
        self.assertEqual(combined["samples"][0]["exit_code"], 1)
        self.assertNotIn("l2_grind_observations", combined)

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
                "grind_retry_observations": [
                    {"level": 0, "retries": [0, 2, 1]},
                    {"level": 2, "retries": [0, 0, 0]},
                ],
            }
        )
        baseline = dict(current)
        baseline["grind_retry_observations"] = [
            {"level": 0, "retries": [0, 0, 0]},
            {"level": 2, "retries": [1, 0, 0]},
        ]
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
        self.assertNotIn("Statement preparation", report)
        self.assertNotIn("vs base</sub>", report)
        self.assertNotIn("vs merge base</sub>", report)
        self.assertIn("### Phase time", report)
        self.assertIn("### Memory and setup size", report)
        self.assertIn("### Proof size and protocol shape", report)
        self.assertLess(
            report.index("### Proof size and protocol shape"),
            report.index("### Memory and setup size"),
        )
        self.assertNotIn("### Protocol shape", report)
        self.assertNotIn("| Status |", report)
        self.assertIn("Setup vector", report)
        self.assertIn("Prepared NTT cache", report)
        self.assertIn("Verify, multi-threaded", report)
        self.assertIn("Verify, single-threaded", report)
        self.assertIn("Fold A/B/D schedule", report)
        self.assertIn("| Fold levels | Grinding retries |", report)
        self.assertIn("L0: 0 / 2 / 1", report)
        self.assertIn("L2: 0 / 0 / 0", report)
        self.assertIn("<sub>Merge base</sub>", report)
        self.assertIn("L2: 1 / 0 / 0", report)
        self.assertIn("listed in measured-run order", report)
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

    def test_adaptive_multi_group_case_label_exposes_workload_shape(self) -> None:
        from scripts.profile_bench_report import (
            human_case_label,
            normalize_case_summary,
            render_matrix_summary,
        )

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
            "Fp128 multi-group (final nv32, 4 polys total), direct setup check",
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_matrix_summary([summary], None)
        self.assertIn(
            "two precommitted nv16 singleton polynomials and two polynomials "
            "in the displayed final group",
            output.getvalue(),
        )

        summary["config"] = "adaptive recursive multi-group W8R2"
        self.assertEqual(
            human_case_label(summary),
            "Fp128 multi-group W8R2 (final nv32, 4 polys total), direct setup check",
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
                "num_vars": 34,
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
                                "public_num_vars": 34,
                                "public_num_polynomials": 2,
                            },
                        ],
                    }
                ],
            }
        )

        statement = public_opening_statement(summary)
        name = benchmark_name(
            "onehot_fp128_multi_group_recursive", 34, 4, "recursive"
        )
        self.assertIn("one 16 variable polynomial", statement)
        self.assertIn("at its own point", statement)
        self.assertIn("2 34 variable polynomials", statement)
        self.assertEqual(
            name,
            "fp128 multi-group opening, final nv34, 4 polynomials total "
            "(recursive setup contribution)",
        )
        self.assertNotIn("same-point", name)

    def test_profile_definitions_separate_ci_shards_from_public_statements(self) -> None:
        from scripts.profile_bench_report import normalize_case_summary, render_profile_definitions

        cases = [
            normalize_case_summary(
                {
                    "mode": "dense_fp32",
                    "num_vars": 30,
                    "num_polys": 1,
                    "benchmark_shard": "1-fp32-base",
                }
            ),
            normalize_case_summary(
                {
                    "mode": "onehot_fp32",
                    "num_vars": 34,
                    "num_polys": 1,
                    "benchmark_shard": "1-fp32-base",
                }
            ),
            normalize_case_summary(
                {
                    "mode": "dense_fp64",
                    "num_vars": 29,
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
            "| <code>1-fp32-base</code> | Fp32 dense nv30, direct setup check<br>Fp32 one\\-hot nv34, direct setup check |",
            shard_section,
        )
        self.assertIn(
            "| <code>2-fp64-base</code> | Fp64 dense nv29, direct setup check |",
            shard_section,
        )
        self.assertIn("Over Fp32", statement_section)
        self.assertIn("Fp32 dense nv30, direct setup check", statement_section)
        self.assertIn("Over Fp64", statement_section)
        self.assertIn("Fp64 dense nv29, direct setup check", statement_section)

    def test_partial_merge_base_coverage_is_explicit(self) -> None:
        from scripts.profile_bench_report import render_report

        case = {
            "mode": "dense_fp32",
            "num_vars": 30,
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
            "stage1_norm_proof_bytes": 0,
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
        self.assertIn("Commitment matrices used at this fold", report)
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

        case = BenchmarkCaseSpec(mode="dense_fp32", num_vars=30, num_polys=1)
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
