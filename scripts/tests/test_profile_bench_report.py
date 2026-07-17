import argparse
import contextlib
import io
import json
import pathlib
import tempfile
import unittest


class ProfileBenchReportTests(unittest.TestCase):
    def test_plan_case_runs_orders_warmups_then_measured(self) -> None:
        from scripts.profile_bench_report import BenchmarkCaseSpec, ScheduledRun, plan_case_runs

        case = BenchmarkCaseSpec(mode="onehot_fp128_d64", num_vars=24, num_polys=1)
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

        case = BenchmarkCaseSpec(mode="onehot_fp128_d64", num_vars=24, num_polys=1)
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
                "case": ["onehot_fp128_d64:24:1", "onehot_fp128_d64:24:1"],
                "mode": "onehot_fp128_d64",
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
                "final_w_encoding": "segment_typed",
                "z_witness_linf_cap": "4096",
                "z_rice_low_bits_wire": "10",
                "z_rice_low_bits_cap": "12",
                "z_bits_per_coord_golomb": "12.50",
            },
        )
        self.assertEqual(summary["z_rice_low_bits_wire"], 10)
        self.assertEqual(summary["z_rice_low_bits_cap"], 12)
        self.assertAlmostEqual(summary["z_bits_per_coord_golomb"], 12.50)

    def test_z_fold_encoding_stats_prefers_wire_low_bits(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            'INFO z fold encoding stats label=onehot_fp128_d64 '
            'z_coords=100 witness_linf_cap=4096 rice_low_bits_wire=10 rice_low_bits_cap=12 '
            'bits_per_coord_at_wire=12.5 bits_per_coord_packed=15.0 z_payload_bytes=200\n'
        )
        summary = extract_summary(log, mode="onehot_fp128_d64", num_vars=24, num_polys=1)
        self.assertEqual(summary["z_rice_low_bits_wire"], 10)
        self.assertEqual(summary["z_rice_low_bits_cap"], 12)
        self.assertAlmostEqual(summary["z_bits_per_coord_golomb"], 12.5)

    def test_planned_fold_level_parses_physical_geometry(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            'INFO planned fold level label=onehot_fp128_d64 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=2 n_b=3 n_d=4 '
            'challenge_l1_mass=8 log_basis=5 position_index_bits=7 block_index_bits=3 '
            'num_live_ring_elements_per_claim=768 num_live_blocks=6 block_index_domain_size=8 '
            'num_positions_per_block=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )

        summary = extract_summary(log, mode="onehot_fp128_d64", num_vars=24, num_polys=1)

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
                "level_bytes": 4096,
            },
        )

    def test_planned_fold_level_normalizes_merge_base_geometry(self) -> None:
        from scripts.profile_bench_report import extract_summary

        log = (
            'INFO planned fold level label=onehot_fp128_d64 level=0 d=64 n_a=2 n_b=3 n_d=4 '
            'challenge_l1_mass=8 log_basis=5 m_vars=7 r_vars=3 '
            'num_blocks=8 block_len=2 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )

        summary = extract_summary(log, mode="onehot_fp128_d64", num_vars=24, num_polys=1)
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
            'INFO planned fold level label=onehot_fp128_d64 level=0 d=64 n_a=2 n_b=3 n_d=4 '
            'challenge_l1_mass=8 log_basis=5 position_bits=7 block_bits=3 '
            'num_blocks=8 block_len=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )

        summary = extract_summary(log, mode="onehot_fp128_d64", num_vars=24, num_polys=1)
        level = summary["planned_levels"][0]

        self.assertEqual(level["position_index_bits"], 7)
        self.assertEqual(level["block_index_bits"], 3)
        self.assertEqual(level["num_positions_per_block"], 128)
        self.assertEqual(level["num_live_blocks"], 1)

    def test_rendered_schedule_uses_names_and_main_deltas(self) -> None:
        from scripts.profile_bench_report import extract_summary, render_planned_levels

        current_log = (
            'INFO planned fold level label=onehot_fp128_d64 level=0 d=64 d_a=64 d_b=32 d_d=16 '
            'n_a=4 n_b=6 n_d=8 challenge_l1_mass=16 log_basis=6 position_index_bits=7 '
            'block_index_bits=3 num_live_ring_elements_per_claim=768 num_live_blocks=6 '
            'block_index_domain_size=8 num_positions_per_block=128 delta_commit=4 delta_open=5 '
            'delta_fold=6 current_w_len=1024 next_w_len=2048 level_bytes=4096\n'
        )
        baseline_log = current_log.replace("n_a=4", "n_a=2").replace(
            "level_bytes=4096", "level_bytes=2048"
        )
        current = extract_summary(current_log, "onehot_fp128_d64", 24, 1)["planned_levels"]
        baseline = extract_summary(baseline_log, "onehot_fp128_d64", 24, 1)["planned_levels"]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_planned_levels(current, baseline)
        report = output.getvalue()

        self.assertIn("A ring dimension", report)
        self.assertIn("Number of positions in each block", report)
        self.assertIn("Number of live source A-ring elements in each claim", report)
        self.assertIn("+100.00% vs main", report)
        self.assertNotIn("| M |", report)
        self.assertNotIn("r_pos", report)

    def test_proof_breakdown_marks_absent_components(self) -> None:
        from scripts.profile_bench_report import extract_summary, render_proof_levels

        log = (
            'INFO proof fold level label=onehot_fp128_d64 level=0 d=64 total_bytes=12 '
            'fold_grind_nonce_bytes=4 grind_nonce=3 grind_attempts=4 '
            'stage2_sumcheck_bytes=8 root_variant=terminal\n'
        )
        levels = extract_summary(log, "onehot_fp128_d64", 24, 1)["proof_levels"]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_proof_levels(levels, levels)
        report = output.getvalue()

        self.assertIn("Fold-level bytes", report)
        self.assertIn("—", report)
        self.assertIn("+0.00% vs main", report)
        self.assertIn("final witness", report)
        proof_table_lines = [line for line in report.splitlines() if line.startswith("| ")][:3]
        self.assertEqual(len({line.count("|") for line in proof_table_lines}), 1)

    def test_matrix_embeds_main_delta_in_every_numeric_metric(self) -> None:
        from scripts.profile_bench_report import normalize_case_summary, render_matrix_summary

        current = normalize_case_summary(
            {
                "mode": "onehot_fp128_d64",
                "num_vars": 32,
                "num_polys": 1,
                "exit_code": 0,
                "setup_s": 2.0,
                "setup_vector_bytes": 4 * 1024 * 1024,
                "setup_ntt_cache_bytes": 8 * 1024 * 1024,
                "commit_s": 4.0,
                "prove_total_s": 6.0,
                "verify_total_s": 0.008,
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
            "max_rss_kib",
            "proof_size_bytes",
        ):
            baseline[key] = float(current[key]) / 2.0

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            render_matrix_summary([current], {str(current["case_id"]): baseline})
        report = output.getvalue()

        self.assertEqual(report.count("+100.00% vs main"), 8)
        self.assertIn("Setup vector size", report)
        self.assertIn("Prepared NTT cache size", report)
        self.assertIn("4.0 MiB", report)
        self.assertIn("8.0 MiB", report)
        self.assertIn("nv32Onehot256", report)
        self.assertIn("D=64", report)
        self.assertNotIn("Proof B", report)
        self.assertNotIn("Setup Mode", report)

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
            "total_bytes": 12,
            "present_byte_fields": ["fold_grind_nonce_bytes", "stage2_sumcheck_bytes"],
            "extension_opening_partials_bytes": 0,
            "extension_opening_sumcheck_bytes": 0,
            "fold_grind_nonce_bytes": 4,
            "v_bytes": 0,
            "stage1_sumcheck_bytes": 0,
            "stage1_interstage_claims_bytes": 0,
            "stage1_s_claim_bytes": 0,
            "stage2_sumcheck_bytes": 8,
            "stage3_sumcheck_bytes": 0,
            "next_w_commitment_bytes": 0,
            "next_w_eval_bytes": 0,
            "root_variant": "terminal",
        }
        case = {
            "mode": "onehot_fp128_d64",
            "num_vars": 32,
            "num_polys": 1,
            "setup_contribution_mode": "direct",
            "exit_code": 0,
            "setup_s": 2.0,
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

        self.assertIn("Delta versus main", report)
        self.assertIn("unchanged", report)
        self.assertIn("A ring dimension", report)
        self.assertIn("Proof size by fold level", report)
        self.assertNotIn("Proof framing", report)

    def test_configured_cases_treats_setup_mode_as_case_dimension(self) -> None:
        from scripts.profile_bench_report import configured_cases

        args = type(
            "Args",
            (),
            {
                "case": [
                    "onehot_fp128_d64:32:1",
                    "onehot_fp128_d64:32:1:recursive",
                ],
                "mode": "onehot_fp128_d64",
                "num_vars": 32,
                "num_polys": 1,
            },
        )()

        cases = configured_cases(args)

        self.assertEqual([case.setup_mode for case in cases], ["direct", "recursive"])
        self.assertNotEqual(cases[0].case_id, cases[1].case_id)
        self.assertTrue(cases[1].case_id.endswith("-setup-recursive"))

    def test_write_aggregate_summaries_propagates_sibling_failure(self) -> None:
        from scripts.profile_bench_report import (
            BenchmarkCaseSpec,
            ScheduledRun,
            case_status,
            write_aggregate_summaries,
        )

        case = BenchmarkCaseSpec(mode="onehot_fp128_d64", num_vars=24, num_polys=1)
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


if __name__ == "__main__":
    unittest.main()
