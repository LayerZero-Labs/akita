#!/usr/bin/env python3
"""Compare bounded sampler-only RISC-V traces from two Akita checkouts."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import sys


def prepare_baseline(candidate, reference, destination):
    """Prepare a separate local clone; never change the candidate's git state."""
    if destination.exists():
        raise RuntimeError(f"Baseline destination already exists: {destination}")
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(candidate), str(destination)], check=True)
    subprocess.run(["git", "switch", "--detach", reference], cwd=destination, check=True)
    relative = Path("profile/akita-recursion")
    shutil.copytree(candidate / relative / "sampler", destination / relative / "sampler", ignore=shutil.ignore_patterns("__pycache__"))
    manifest = destination / relative / "Cargo.toml"
    content = manifest.read_text()
    content = content.replace('  "inline-parity/host",', '  "inline-parity/host",\n  "sampler/guest",\n  "sampler/host",')
    profile = (candidate / relative / "Cargo.toml").read_text().split("[profile.sampler-guest]", 1)[1]
    manifest.write_text(content + "\n[profile.sampler-guest]" + profile)
    shutil.copy2(candidate / relative / "Cargo.lock", destination / relative / "Cargo.lock")


def read_rows(output, log):
    cycles = re.findall(
        r'"sampler_batch": (\d+) RV64IMAC cycles \+ (\d+) virtual instructions = (\d+) total cycles',
        log.read_text(),
    )
    lines = output.read_text().splitlines()
    if len(lines) != len(cycles):
        raise RuntimeError(f"Incomplete marker/output pairs: {output}")
    rows = []
    for line, counts in zip(lines, cycles):
        row = dict(field.split("=", 1) for field in line.split())
        for key in ["seed", "D", "pm1", "pm2", "draws", "trace"]:
            row[key] = int(row[key])
        row.update(zip(["real", "virtual", "cycles"], map(int, counts)))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Existing dedicated baseline checkout with identical benchmark sources")
    parser.add_argument("--baseline-ref", default="c4a650bd89f169fc874c3bdeef3c7ef5c684fd23", help="Local git ref used when creating an isolated baseline")
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--host", type=Path, help="Prebuilt host binary; otherwise build the small host automatically")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--production-seeds", type=int, default=32)
    parser.add_argument("--stress-seeds", type=int, default=8)
    parser.add_argument("--production-batch", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.candidate = args.candidate.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.baseline is None:
        args.baseline = args.output_dir / "baseline-source"
        prepare_baseline(args.candidate, args.baseline_ref, args.baseline)
    args.baseline = args.baseline.resolve()
    env = dict(os.environ, CARGO_BUILD_JOBS="2")
    env.setdefault("ZEROOS_GUEST_RUSTFLAGS", "-Zunstable-options")
    if args.host is None:
        target = args.output_dir / "host-target"
        subprocess.run(["cargo", "build", "--release", "-p", "akita-sampler-host", "--target-dir", str(target)], cwd=args.candidate / "profile/akita-recursion", env=env, check=True)
        args.host = target / "release/akita-sampler-host"
    results = {}
    for variant, checkout in [("baseline", args.baseline), ("candidate", args.candidate)]:
        results[variant] = []
        for selection, seeds in [("production", args.production_seeds), ("stress", args.stress_seeds)]:
            out = args.output_dir / f"{variant}-{selection}.out"
            log = args.output_dir / f"{variant}-{selection}.log"
            command = [str(args.host.resolve()), str((args.output_dir / f"guest-{variant}").resolve()), "matrix", str(seeds), selection, str(args.production_batch)]
            with out.open("w") as stdout, log.open("w") as stderr:
                run = subprocess.run(command, cwd=checkout / "profile/akita-recursion", env=env, stdout=stdout, stderr=stderr)
            if run.returncode:
                raise RuntimeError(f"{variant}/{selection} failed ({run.returncode}); see {log}")
            rows = read_rows(out, log)
            results[variant].extend(rows)
            print(f"{variant}/{selection}: {len(rows)} traces", flush=True)

    def key(row):
        return tuple(row[k] for k in ["D", "pm1", "pm2", "rejection", "draws", "seed"])

    baseline = {key(r): r for r in results["baseline"]}
    candidate = {key(r): r for r in results["candidate"]}
    if baseline.keys() != candidate.keys():
        raise RuntimeError("Baseline and candidate case matrices differ")
    paired = []
    groups = collections.defaultdict(list)
    for k, old in baseline.items():
        new = candidate[k]
        if old["digest"] != new["digest"]:
            raise RuntimeError(f"Output mismatch: {k}")
        row = {name: old[name] for name in ["D", "pm1", "pm2", "rejection", "production", "draws", "seed", "digest"]}
        row.update(old_cycles=old["cycles"], new_cycles=new["cycles"], old_trace=old["trace"], new_trace=new["trace"])
        row["change_pct"] = 100 * (new["cycles"] - old["cycles"]) / old["cycles"]
        paired.append(row)
        groups[k[:-1]].append(row)
    summaries = []
    for k, rows in groups.items():
        summaries.append(dict(zip(["D", "pm1", "pm2", "rejection", "draws"], k)) | {
            "seeds": len(rows), "production": rows[0]["production"],
            "old_cycles_per_draw": statistics.mean(r["old_cycles"] / r["draws"] for r in rows),
            "new_cycles_per_draw": statistics.mean(r["new_cycles"] / r["draws"] for r in rows),
            "mean_change_pct": statistics.mean(r["change_pct"] for r in rows),
            "worst_change_pct": max(r["change_pct"] for r in rows),
            "regressed_seeds": sum(r["new_cycles"] > r["old_cycles"] for r in rows),
        })
    with (args.output_dir / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    report = {
        "baseline_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.baseline, text=True).strip(),
        "candidate_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.candidate, text=True).strip(),
        "candidate_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=args.candidate)).hexdigest(),
        "paired_traces": len(paired), "all_output_fingerprints_match": True,
        "execution_mode": "sampler-only execute-only emulation; no trace rows or prover state retained",
        "guest_profile": env.get("SAMPLER_GUEST_PROFILE", "sampler-guest"),
        "jolt_cli_version": subprocess.check_output([env.get("JOLT_PATH", "jolt"), "--version"], text=True).strip(),
        "max_trace_rows": max(max(r["old_trace"], r["new_trace"]) for r in paired),
        "child_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "all_cases_nonregressing": all(r["new_cycles"] <= r["old_cycles"] for r in paired),
        "worst_change_pct": max(r["change_pct"] for r in paired),
        "cases": summaries,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in summaries:
        if row["production"] == "true":
            print(f"D={row['D']:4} ({row['pm1']},{row['pm2']}) rejection={row['rejection']}: {row['mean_change_pct']:+.3f}% worst={row['worst_change_pct']:+.3f}%")
    print(f"{len(summaries)} cases, {len(paired)} matched runs; worst change {report['worst_change_pct']:+.3f}%")


if __name__ == "__main__":
    main()
