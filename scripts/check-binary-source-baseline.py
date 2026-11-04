#!/usr/bin/env python3
"""Linux-only byte compatibility gate against the pinned typed-Stage-3 dev.

Compare three word fixtures and sixteen prime/projection selections (114 files).

Capture setup, commitments, proof and continuation from independently compiled
base/head worktrees with separate fresh Cargo targets. Any failed command,
missing capture or byte drift is fatal. Only the temporary fixtures are modified;
retain fixture snapshots, build artifacts and captures for review, and remove
the disposable worktrees on exit. Requires Git 2.36 or later and cargo.
"""

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile

BASE = "357d12b748b356b8da69eb20fe556b79b3234bec"
FIXTURE = Path("crates/akita-pcs/tests/binary_source_end_to_end.rs")
CASES = ("false-128", "false-8", "true-128")
TESTS = (
    "projection_binary_group_opens_once_inside_the_callers_transcript",
    "binary_and_ordinary_groups_share_one_opening",
    "field_switch_narrow_source_mutations_authenticate",
)
PRIME_FIXTURE = Path("crates/akita-pcs/tests/binary_source_end_to_end/prime_plane.rs")
PRIME_TESTS = (
    "prime_plane::prime_plane_prime_only",
    "prime_plane::prime_plane_word_and_one_two_five_planes",
    "prime_plane::prime_plane_mixed_ordinary_group",
    "prime_plane::prime_plane_several_polynomials_do_not_share",
    "prime_plane::projection_two_polynomials_and_prime_plane_rejects_y_mutations",
)
PRIME_CASES = (
    "false-false-1-1", "false-false-2-1", "false-false-5-1",
    "false-true-1-1", "false-true-2-1", "false-true-5-1",
    "true-true-1-1", "true-true-2-1", "true-true-5-1",
    "true-false-1-1", "true-false-2-1", "true-false-5-1",
    "false-true-0-2", "false-true-2-2", "false-false-2-2", "false-false-1-2",
)
PARTS = ("", ".setup", ".prefixes", ".commitment", ".ordinary", ".continuation")

# Instrument only the existing honest fixture, after it has completed the proof.
# Base and head retain their own complete setup, compiler and transcript code.
PROOF_ANCHOR = "    let proof = prover.narg_string().to_vec();\n"
PROOF_CAPTURE = r'''
    capture_binary_baseline(mixed, word_width, "", |file| {
        std::io::Write::write_all(file, &proof).unwrap();
    });
    capture_binary_baseline(mixed, word_width, ".setup", |file| {
        setup.expanded.serialize_compressed(file).unwrap();
    });
    capture_binary_baseline(mixed, word_width, ".prefixes", |file| {
        // Compare both ordinary setup slots and the installed typed A registry.
        setup.prefix_slots.serialize_compressed(&mut *file).unwrap();
        verifier_setup.prefix_slots().serialize_compressed(file).unwrap();
    });
    capture_binary_baseline(mixed, word_width, ".commitment", |file| {
        commitment.serialize_compressed(file).unwrap();
    });
    capture_binary_baseline(mixed, word_width, ".ordinary", |file| {
        if let Some(ordinary) = &ordinary_commitment {
            ordinary.serialize_compressed(file).unwrap();
        }
    });
'''
CONTINUATION_ANCHOR = "    let continuation: [u8; 32] = prover.verifier_message();\n"
CONTINUATION_CAPTURE = r'''
    capture_binary_baseline(mixed, word_width, ".continuation", |file| {
        std::io::Write::write_all(file, &continuation).unwrap();
    });
'''
CAPTURE_FUNCTION = r'''
fn capture_binary_baseline(
    mixed: bool,
    word_width: usize,
    part: &str,
    write: impl FnOnce(&mut std::fs::File),
) {
    let directory = std::env::var_os("AKITA_BINARY_CAPTURE_DIR").unwrap();
    let path = std::path::PathBuf::from(directory)
        .join(format!("binary-root-{mixed}-{word_width}.bin{part}"));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    write(&mut file);
}
'''


def capture(root, run_dir, worktrees, label, revision, features, baseline=None):
    checkout = run_dir / label
    # Git and its population child inherit the blocked mask. Defer handled
    # signals until creation finishes, so cleanup cannot race population.
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, (signal.SIGHUP, signal.SIGINT, signal.SIGTERM),
    )
    try:
        # Reserve ownership before Git can register the checkout.
        worktrees.append(checkout)
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "worktree", "add",
             "--detach", str(checkout), revision],
            cwd=root, check=True,
        )
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    fixture = checkout / FIXTURE
    text = fixture.read_text()
    for anchor, addition in (
        (PROOF_ANCHOR, PROOF_CAPTURE),
        (CONTINUATION_ANCHOR, CONTINUATION_CAPTURE),
    ):
        if text.count(anchor) != 1:
            raise RuntimeError(f"{label}: expected exactly one fixture anchor: {anchor!r}")
        text = text.replace(anchor, anchor + addition)
    # Upstream's fixed vectors cover Keccak. The external baseline supplies
    # Blake2b expectations; no proof constants are taken from the candidate.
    known = '    compatibility::assert_base_bytes(&proof, mixed, word_width);'
    if label == "base":
        if text.count(known) != 1:
            raise RuntimeError("baseline known-vector anchor changed")
        text = text.replace(known, '    #[cfg(feature = "transcript-keccak")]\n' + known)
        declaration = '#[path = "binary_source_end_to_end/compatibility.rs"]'
        if text.count(declaration) != 1:
            raise RuntimeError("baseline known-vector module changed")
        text = text.replace(declaration, '#[cfg(feature = "transcript-keccak")]\n' + declaration)
    fixture.write_text(text + CAPTURE_FUNCTION)
    (run_dir / f"{label}-fixture.rs").write_text(text + CAPTURE_FUNCTION)
    prime_fixture = checkout / PRIME_FIXTURE
    prime_text = prime_fixture.read_text()
    prime_anchor = '    eprintln!(\n        "prime_plane dev='
    if prime_text.count(prime_anchor) != 1:
        raise RuntimeError(f"{label}: expected one prime observation site")
    # Reuse the same serialization statements for each selected upstream case.
    prime_capture = (PROOF_CAPTURE + CONTINUATION_CAPTURE).replace(
        "capture_binary_baseline(mixed, word_width,", "capture_prime_baseline(mixed, word, planes, owners,",
    ).replace("ordinary_commitment", "ordinary")
    prime_function = CAPTURE_FUNCTION.replace(
        "capture_binary_baseline", "capture_prime_baseline",
    ).replace(
        "    word_width: usize,", "    word: bool,\n    planes: usize,\n    owners: usize,",
    ).replace("binary-root-{mixed}-{word_width}", "binary-prime-{mixed}-{word}-{planes}-{owners}")
    prime_text = prime_text.replace(prime_anchor, prime_capture + prime_anchor) + prime_function
    prime_fixture.write_text(prime_text)
    (run_dir / f"{label}-prime-fixture.rs").write_text(prime_text)
    captures = run_dir / f"{label}-captures"
    captures.mkdir()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("AKITA_BINARY_") or key == "AKITA_CAPTURE_BINARY_BYTE_GOLDENS":
            env.pop(key)
    env["AKITA_BINARY_CAPTURE_DIR"] = str(captures)
    # Never reuse fingerprints or compiled dependencies across revisions.
    target = run_dir / f"{label}-target"
    target.mkdir()
    env["CARGO_TARGET_DIR"] = str(target)
    env.setdefault("CARGO_BUILD_JOBS", "2")
    if baseline is not None:
        env["AKITA_BINARY_BASELINE_DIR"] = str(baseline)
    # Only these fixtures exist on both revisions. Additional selected-path
    # tests run in the full suite; they cannot have pinned dev byte captures.
    for test in TESTS + PRIME_TESTS:
        subprocess.run([
            "cargo", "test", "--locked", "--release", "-p", "akita-pcs",
            "--test", "binary_source_end_to_end", "--no-default-features",
            "--features", features, "--", test, "--exact", "--test-threads=1",
        ], cwd=checkout, env=env, check=True)
    expected = {f"binary-root-{case}.bin{part}" for case in CASES for part in PARTS}
    expected.update(f"binary-prime-{case}.bin{part}" for case in PRIME_CASES for part in PARTS)
    if {path.name for path in captures.iterdir()} != expected:
        raise RuntimeError(f"{label}: incomplete or unexpected fixture captures")
    for name in expected:
        empty_ordinary = name in {
            "binary-root-false-128.bin.ordinary", "binary-root-false-8.bin.ordinary",
        }
        empty_ordinary = empty_ordinary or (
            name.startswith("binary-prime-false-") and name.endswith(".ordinary")
        )
        if not empty_ordinary and (captures / name).stat().st_size == 0:
            raise RuntimeError(f"{label}: empty capture: {name}")
    return captures


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("transcript-blake2b", "transcript-keccak"),
                        default="transcript-keccak")
    parser.add_argument("--dev-protocol", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("this gate must run on Linux")
    root = Path(__file__).resolve().parent.parent
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if head == BASE:
        raise RuntimeError("head must be the extension revision, not its own baseline")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head], cwd=root, check=True)
    features = ["labinius-trinomial", args.backend]
    if not args.no_parallel:
        features.append("parallel")
    if args.dev_protocol:
        features.append("akita-params/dev-protocol")
    features = ",".join(features)
    output = root / "target"
    output.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="binary-source-baseline-", dir=output)).resolve()
    print(f"Captures: {run_dir}", flush=True)
    receipt = {"base": BASE, "head": head, "features": features}
    (run_dir / "revisions.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (run_dir / "git-version.txt").open("wb") as stream:
        subprocess.run(["git", "--version"], cwd=root, stdout=stream, check=True)
    # Verify cleanup support before reserving or creating any worktree.
    try:
        with (run_dir / "preflight-registrations.z").open("wb") as stream:
            subprocess.run(
                ["git", "worktree", "list", "--porcelain", "-z"],
                cwd=root, stdout=stream, check=True,
            )
    except subprocess.CalledProcessError:
        print(
            f"Git worktree preflight failed; Git 2.36 or later is required. Receipts: {run_dir}",
            file=sys.stderr,
        )
        raise
    worktrees = []

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    try:
        base = capture(root, run_dir, worktrees, "base", BASE, features)
        candidate = capture(root, run_dir, worktrees, "head", head, features, base)
        records = {}
        for path in sorted(base.iterdir()):
            if not filecmp.cmp(path, candidate / path.name, shallow=False):
                raise RuntimeError(f"base/head bytes differ: {path.name}")
            records[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
        receipt["captures"] = records
        (run_dir / "PASS.json.pending").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        failed = sys.exc_info()[0] is not None
        # Finish both removals even if another handled signal arrives.
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        cleanup_failed = False
        for checkout in reversed(worktrees):
            try:
                registrations = subprocess.run(
                    ["git", "worktree", "list", "--porcelain", "-z"],
                    cwd=root, capture_output=True, check=True,
                ).stdout.split(b"\0")
                if b"worktree " + os.fsencode(checkout) not in registrations:
                    continue
                # An interrupted add can leave an initialization lock.
                subprocess.run(
                    ["git", "worktree", "remove", "--force", "--force", str(checkout)],
                    cwd=root, check=True,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                cleanup_failed = True
                print(f"Worktree cleanup failed for {checkout}: {error}", file=sys.stderr)
        if cleanup_failed and not failed:
            raise RuntimeError("worktree cleanup failed after byte comparisons passed")
    os.replace(run_dir / "PASS.json.pending", run_dir / "PASS.json")
    print(f"PASS: {len(records)} independent base/head captures match ({features})")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        # Keep command failures, including signal exits, through cleanup.
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode)
