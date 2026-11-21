#!/usr/bin/env python3
"""Freeze indexed/CT outputs from an independent Jolt preview checkout.

Usage: macslot python3 crates/akita-challenges/scripts/gen_indexed_preview.py /path/to/jolt-preview
The temporary reference crate uses preview sampler bodies verbatim. Only its
error/type adapters and module wiring are supplied here. Invoke this script
under macslot so its Cargo child uses a capped build slot. No Jolt workspace targets or Akita samplers are compiled.
"""

import hashlib
from pathlib import Path
import subprocess
import sys

DRIVER = r'''fn encode(challenge: &SparseChallenge) -> String {
    challenge
        .positions
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .chain(challenge.coeffs.iter().map(|&x| x as u8))
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn main() {
    println!("# Preview PREVIEW_REV; each row: seed-byte/ramp D pm1 pm2 norm coordinate encoded-positions-le-u32-then-i8 attempts");
    for tag in [0usize, 11, 23, 31, 255, 256] {
        let seed: [u8; 32] = std::array::from_fn(|i| if tag == 256 { i as u8 } else { tag as u8 });
        let mut cases = [64, 128, 256, 512, 1024, 2048]
            .into_iter()
            .map(|d| {
                (
                    d,
                    SparseChallengeConfig::production_for_ring_dim(d).unwrap(),
                    None,
                )
            })
            .collect::<Vec<_>>();
        cases.extend([
            (
                64,
                D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                Some(OperatorNormRejection::D64_SELECTIVE_L2),
            ),
            (
                128,
                D128_SELECTIVE_L2_CHALLENGE_CONFIG,
                Some(OperatorNormRejection::D128_SELECTIVE_L2),
            ),
        ]);
        cases.extend([
            (2048, SparseChallengeConfig::pm1_only(1), None),
            (0, SparseChallengeConfig::pm1_only(0), None),
            (
                64,
                SparseChallengeConfig {
                    count_pm1: 0,
                    count_pm2: 64,
                },
                None,
            ),
        ]);
        for (d, cfg, policy) in cases {
            let cs = sample_indexed_challenges_from_seed(&seed, d, 3, &cfg, policy).unwrap();
            for (i, c) in cs.iter().enumerate() {
                let attempts = if d == 64 && policy.is_some() && i == 0 {
                    let (ct, count) =
                        challenge::sampler::sample_challenge_ct_counted(&seed).unwrap();
                    assert_eq!(&ct, c);
                    count
                } else {
                    0
                };
                println!(
                    "{tag} {d} {} {} {} {i} {} {attempts}",
                    cfg.count_pm1,
                    cfg.count_pm2,
                    usize::from(policy.is_some()),
                    if c.positions.is_empty() {
                        "-".into()
                    } else {
                        encode(c)
                    }
                );
            }
        }
    }
}
'''


def main():
    repo = Path(__file__).resolve().parents[3]
    preview = Path(sys.argv[1]).resolve()
    source = preview / "crates/jolt-crypto/src/challenge"
    rev = subprocess.check_output(["git", "-C", str(preview), "rev-parse", "HEAD"], text=True).strip()
    temporary = repo / "target/a2-preview-reference"
    (temporary / "src").mkdir(parents=True, exist_ok=True)
    (temporary / "Cargo.toml").write_text(
        '[package]\nname = "a2-preview-reference"\nversion = "0.0.0"\nedition = "2021"\n'
        '[workspace]\n[features]\nsample = []\nsample-diagnostics = ["sample"]\n'
        'parallel = []\nont-static-base = []\n[dependencies]\nkeccak = "0.1.6"\n'
        'smallvec = { version = "1.15.2", features = ["const_generics"] }\n'
        'serde = { version = "1", features = ["derive"] }\n'
        f'akita-error = {{ path = "{repo}/crates/akita-error" }}\n'
    )
    sampler = (source / "sampler/mod.rs").read_text()
    sampler = sampler.replace("mod stream;\n", "").replace(
        "pub use stream::{SparseChallengeShell, SparseChallengeStream};\n", ""
    )
    hashes = []
    for name in ("constant_time", "indexed", "op_norm", "op_norm_accumulate", "position_sample", "signed_sparse", "xof"):
        path = source / f"sampler/{name}.rs"
        sampler = sampler.replace(f"mod {name};", f'#[path = "{path}"]\nmod {name};')
        hashes.append(f"# {name}.rs sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    (temporary / "src/sampler.rs").write_text(sampler)
    sparse = (source / "sparse.rs").read_text()
    masks = sparse[sparse.index("/// `0xFF`"):sparse.index("\n#[expect(", sparse.index("/// `0xFF`"))]
    adapter = f'''#![allow(dead_code, unexpected_cfgs)]
use akita_error::AkitaError as CryptoError;
mod lattice {{ pub const CHALLENGE_RING_DIM: usize = 64; }}
mod challenge {{
    #[path = "{source}/config.rs"] mod config;
    pub use config::*;
    pub const INLINE_SPARSE_WEIGHT: usize = 42;
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct SparseChallenge {{
        pub positions: smallvec::SmallVec<[u32; INLINE_SPARSE_WEIGHT]>,
        pub coeffs: smallvec::SmallVec<[i8; INLINE_SPARSE_WEIGHT]>,
    }}
    pub mod sparse {{
        use crate::lattice::CHALLENGE_RING_DIM;
        {masks}
    }}
    #[path = "{temporary}/src/sampler.rs"] pub mod sampler;
    pub use sampler::sample_indexed_challenges_from_seed;
}}
use challenge::*;
'''
    (temporary / "src/main.rs").write_text(adapter + DRIVER.replace("PREVIEW_REV", rev))
    output = subprocess.check_output([
        "cargo", "+1.99.1", "run", "--quiet", "--manifest-path", str(temporary / "Cargo.toml"),
        "--features", "sample-diagnostics",
    ], cwd=repo, text=True)
    fixture = repo / "crates/akita-challenges/tests/fixtures/indexed_preview.txt"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text("\n".join(hashes) + "\n" + output)
    print(f"Wrote {fixture}")


if __name__ == "__main__":
    main()
