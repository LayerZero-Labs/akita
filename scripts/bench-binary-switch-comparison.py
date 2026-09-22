#!/usr/bin/env python3
"""Build a matched native-CPU comparison without importing reference code.

Run on an AVX-512/GFNI x86 machine, under taskset for a fixed core. Pass a
separately obtained LaBinius checkout; its code is never copied into Akita.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("labinius", type=Path)
parser.add_argument("--target-dir", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
reference = args.labinius.resolve()
if not (reference / "crates/pcs/Cargo.toml").is_file():
    parser.error("LaBinius checkout must contain crates/pcs/Cargo.toml")
for name, path in [("Akita", root), ("LaBinius", reference)]:
    print(f"{name}: {path}", flush=True)
    revision = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    if revision.returncode == 0:
        print(revision.stdout.strip(), flush=True)
        subprocess.run(["git", "-C", str(path), "status", "--short"], check=True)
    else:
        print("Archive checkout; identify this run by the source digest below.", flush=True)
# Include paths and bytes so archive runs can be matched to a local commit.
source_files = sorted((root / "crates/akita-algebra").rglob("*.rs")) + [
    root / "crates/akita-algebra/Cargo.toml", root / "Cargo.toml", root / "Cargo.lock",
    Path(__file__).resolve(), root / "scripts/benchmarks/binary_switch_comparison.rs",
]
digest = hashlib.sha256()
for path in sorted(source_files):
    digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
print(f"Akita comparison source SHA256: {digest.hexdigest()}", flush=True)
subprocess.run(["rustc", "--version"], check=True)
print("Both dependencies: opt-level=3, fat LTO, one codegen unit, target-cpu=native", flush=True)
with tempfile.TemporaryDirectory(prefix="akita-switch-compare-") as directory:
    work = Path(directory)
    (work / "src").mkdir()
    (work / "src/main.rs").write_text((root / "scripts/benchmarks/binary_switch_comparison.rs").read_text())
    manifest = f'''[package]
name = "akita-switch-comparison"
version = "0.0.0"
edition = "2021"
[dependencies]
akita-algebra = {{ path = {json.dumps(str(root / 'crates/akita-algebra'))} }}
labinius = {{ path = {json.dumps(str(reference / 'crates/pcs'))} }}
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
'''
    (work / "Cargo.toml").write_text(manifest)
    environment = dict(os.environ, RUSTFLAGS="-C target-cpu=native", CARGO_TARGET_DIR=str(args.target_dir.resolve()))
    subprocess.run(["cargo", "run", "--release", "--manifest-path", str(work / "Cargo.toml")], env=environment, check=True)
