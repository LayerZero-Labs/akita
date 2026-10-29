//! Golden commitment and proof digests for clear-mode catalog rows.
//!
//! `dev` must produce the same commitment and proof bytes as `main` for every
//! existing configuration when no extension is selected
//! (`docs/branches.md`, rules 1 and 2). This test pins those bytes for a few
//! cheap catalog rows: a two-level one-hot schedule, five-level dense and
//! one-hot schedules, a precommitted multi-group root, and an
//! extension-field opening. Each line records the row, the Blake2b-256 digest
//! of the serialized commitments, the proof length, and the Blake2b-256 digest
//! of the proof. The setup is seed-expanded and its matrices enter every
//! commitment, so a setup change also moves the commitment digest.
//!
//! The digests are frozen prover outputs, not an independent model. Change
//! them only with a protocol change that is meant to move these bytes. To
//! regenerate, run
//!
//! ```text
//! cargo nextest run -p akita-pcs --test golden_proof_digests
//! cargo nextest run -p akita-pcs --test golden_proof_digests --features dev-protocol
//! ```
//!
//! and copy the `actual` table from each failure message into
//! `CURRENT_PROTOCOL` or `DEV_PROTOCOL` respectively.

#![allow(missing_docs)]

mod common;

use akita_config::proof_optimized::fp64;
use akita_cpu_backend::{CommitOutput, CpuBackend, CpuSource, GroupContext};
use akita_serialization::AkitaSerialize;
use common::*;
use jolt_algebra::{ExtField, Ring};

/// Digests under the current protocol.
const CURRENT_PROTOCOL: &[&str] = &[
    "fp128_onehot final=12:1 commitments=32e3bf3534003ccd6d3371a3a4304e650caefab86d311612ef61998bc356eb92 proof_len=42052 proof=9bb1c30fccb581f763d3b45ea67a419d8c6e7b4e2a3e390f8935f9ea4a8686c7",
    "fp128_dense final=14:1 commitments=7c16802e31dbe69151fe27e4ee4d6666ce7e2e4428c6a348016b618f76c890f1 proof_len=73844 proof=c29d541d8fc2da811bc460dc5cdc594e8fe69427c4355299b2ee4dcd60b41496",
    "fp128_onehot final=18:1 commitments=9b6cc92466df8bb492a8247e2b0f9aa070f8b098c795174875f7919e4c42cf24 proof_len=70585 proof=267d96d187caf49a4e6978f27e78269f227bd0a48ec68c6331fc63f7c63d1a7a",
    "fp128_onehot final=16:1;precommitted=14:1 commitments=a31f2ae7274d259c6ce505af798b9c8d0ebe891864c9c8414370d1cbc6b231b0 proof_len=70598 proof=90ffca158387063524f8e18577925375ed4a883bc60cafe24448855f826988fc",
    "fp64_dense final=14:1 commitments=4f6c809555b2a7ecd98b9de86721835257ae388a5de590817e2456f5b5ffeb23 proof_len=68381 proof=23976febd0956f6ac0a3bf3100560af31ee8de896118c202f09d7f5fe24ed693",
];

/// Digests under the dev protocol (`akita_params::DEV_PROTOCOL`).
const DEV_PROTOCOL: &[&str] = &[
    "fp128_onehot final=12:1 commitments=32e3bf3534003ccd6d3371a3a4304e650caefab86d311612ef61998bc356eb92 proof_len=37078 proof=be954264c86e60be72890d7cfb0931f4f2f366bcc4a0ff3fb54fc72c5f3fe45d",
    "fp128_dense final=14:1 commitments=7c16802e31dbe69151fe27e4ee4d6666ce7e2e4428c6a348016b618f76c890f1 proof_len=68808 proof=73341efb53dd6a864e9045356c66a1970377b56a9356bd3cba73030a10b05166",
    "fp128_onehot final=18:1 commitments=9b6cc92466df8bb492a8247e2b0f9aa070f8b098c795174875f7919e4c42cf24 proof_len=65554 proof=0cb16f1454037d55331bec0873f3b2c2f307c21a3a187e271229e53608ccc258",
    "fp128_onehot final=16:1;precommitted=14:1 commitments=a31f2ae7274d259c6ce505af798b9c8d0ebe891864c9c8414370d1cbc6b231b0 proof_len=65502 proof=07ea0d92c43d9a41335ee8d701941b1261ce32a1afc01b0933ab27b93b30a77d",
    "fp64_dense final=14:1 commitments=4f6c809555b2a7ecd98b9de86721835257ae388a5de590817e2456f5b5ffeb23 proof_len=63890 proof=948707a0293e0d38879a61c9126e51a6430c684ca51c3fd92fd7ec5c5c9ebdbe",
];

const SESSION: &[u8] = b"golden_proof_digests";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_line<Base: jolt_algebra::Field>(
    row: &str,
    commitments: &[&CommittedGroup<Base>],
    proof: &[u8],
) -> String
where
    CommittedGroup<Base>: AkitaSerialize,
{
    let mut commitment_bytes = Vec::new();
    for commitment in commitments {
        commitment
            .serialize_compressed(&mut commitment_bytes)
            .expect("serialize commitment");
    }
    format!(
        "{row} commitments={} proof_len={} proof={}",
        hex(&akita_params::digest_descriptor_bytes(&commitment_bytes)),
        proof.len(),
        hex(&akita_params::digest_descriptor_bytes(proof)),
    )
}

/// One fp128 single-group opening.
fn fp128_single_group<Cfg, P>(row: &str, poly: P, point: &[F], opening: F) -> String
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
    P: CpuSource<F, F>,
{
    let scheme =
        load_workspace_scheme::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked())
            .expect("workspace schedule artifact");
    let setup = scheme.setup_prover(point.len(), 1).expect("setup");
    let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
    let CommitOutput {
        committed_group: commitment,
        private_handle: hint,
    } = stack
        .commit(
            scheme.schedules(),
            &stack.import_source(vec![poly]).expect("source"),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let prover_data = prove_input::<Cfg>(point, &[opening], &commitment, hint, scheme.schedules());
    let proof = scheme
        .batched_prove(&setup, prover_data, &stack, SESSION, BasisMode::Lagrange)
        .expect("prove");
    digest_line(row, &[&commitment], &proof)
}

fn fp128_dense_14() -> String {
    const NV: usize = 14;
    let evals = dense_field_evals(NV, 0x601d_0001);
    let point = random_point(NV, 0x601d_0002);
    let opening = dense_opening_lagrange(&evals, &point);
    let poly = DensePoly::<F>::from_field_evals(NV, &evals).expect("dense poly");
    fp128_single_group::<fp128::Dense, _>("fp128_dense final=14:1", poly, &point, opening)
}

fn fp128_onehot(nv: usize) -> String {
    let poly = make_onehot_poly::<fp128::OneHot>(nv, 0x601d_0003 ^ nv as u64);
    let point = random_point(nv, 0x601d_0004 ^ nv as u64);
    let opening = onehot_opening_lagrange(&poly, &point);
    fp128_single_group::<fp128::OneHot, _>(
        &format!("fp128_onehot final={nv}:1"),
        poly,
        &point,
        opening,
    )
}

/// A one-hot final group at `nv = 16` opened jointly with one precommitted
/// one-hot group at `nv = 14`.
fn fp128_onehot_16_precommitted_14() -> String {
    type Cfg = fp128::OneHot;
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 16;
    let scheme =
        load_workspace_scheme::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked())
            .expect("workspace schedule artifact");
    let setup = scheme.setup_prover(FINAL_NV, 2).expect("setup");
    let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");

    let pre_poly = make_onehot_poly::<Cfg>(PRE_NV, 0x601d_0005);
    let CommitOutput {
        committed_group: pre_commitment,
        private_handle: pre_hint,
    } = stack
        .commit(
            scheme.schedules(),
            &stack.import_source(vec![pre_poly.clone()]).expect("source"),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("precommit");
    let final_poly = make_onehot_poly::<Cfg>(FINAL_NV, 0x601d_0006);
    let precommitteds = PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile])
        .expect("nonempty precommitted groups");
    let CommitOutput {
        committed_group: final_commitment,
        private_handle: final_hint,
    } = stack
        .commit(
            scheme.schedules(),
            &stack
                .import_source(vec![final_poly.clone()])
                .expect("source"),
            GroupContext::scheduler_with_precommitted_groups(&precommitteds),
        )
        .expect("final commit");

    let point = random_point(FINAL_NV, 0x601d_0007);
    let claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(
            point[..PRE_NV].to_vec(),
            vec![onehot_opening_lagrange(&pre_poly, &point[..PRE_NV])],
            pre_commitment.clone(),
        )
        .expect("precommitted group"),
        PolynomialGroupClaims::new(
            point.clone(),
            vec![onehot_opening_lagrange(&final_poly, &point)],
            final_commitment.clone(),
        )
        .expect("final group"),
    ])
    .expect("claims");
    let prover_data =
        selected_prover_data::<Cfg>(claims, vec![pre_hint, final_hint], scheme.schedules());
    let proof = scheme
        .batched_prove(&setup, prover_data, &stack, SESSION, BasisMode::Lagrange)
        .expect("prove");
    digest_line(
        "fp128_onehot final=16:1;precommitted=14:1",
        &[&pre_commitment, &final_commitment],
        &proof,
    )
}

/// A dense fp64 opening at an extension-field point.
fn fp64_dense_14() -> String {
    type Cfg = fp64::Dense;
    type Base = fp64::Field;
    type Ext = fp64::ExtensionField;
    const NV: usize = 14;
    let evals: Vec<Base> = (0..1u64 << NV)
        .map(|i| Base::from_u64(i.wrapping_mul(0x9e37_79b9).wrapping_add(17)))
        .collect();
    let point: Vec<Ext> = (0..NV as u64)
        .map(|i| Ext::from_u64(i.wrapping_mul(5).wrapping_add(3)))
        .collect();
    let lifted: Vec<Ext> = evals.iter().map(|&eval| Ext::lift_base(eval)).collect();
    let opening = dense_opening_lagrange(&lifted, &point);

    let scheme =
        load_workspace_scheme::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked())
            .expect("workspace schedule artifact");
    let setup = scheme.setup_prover(NV, 1).expect("setup");
    let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
    let CommitOutput {
        committed_group: commitment,
        private_handle: hint,
    } = stack
        .commit(
            scheme.schedules(),
            &stack
                .import_source(vec![
                    DensePoly::<Base>::from_field_evals(NV, &evals).expect("poly")
                ])
                .expect("source"),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let prover_data = prove_input::<Cfg>(&point, &[opening], &commitment, hint, scheme.schedules());
    let proof = scheme
        .batched_prove(&setup, prover_data, &stack, SESSION, BasisMode::Lagrange)
        .expect("prove");
    digest_line("fp64_dense final=14:1", &[&commitment], &proof)
}

#[test]
fn clear_catalog_rows_keep_their_golden_digests() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let actual = [
            fp128_onehot(12),
            fp128_dense_14(),
            fp128_onehot(18),
            fp128_onehot_16_precommitted_14(),
            fp64_dense_14(),
        ];
        let expected = if akita_params::DEV_PROTOCOL {
            DEV_PROTOCOL
        } else {
            CURRENT_PROTOCOL
        };
        // Each line starts with its row name, up to " commitments=".
        let moved: Vec<&str> = actual
            .iter()
            .enumerate()
            .filter(|(index, line)| expected.get(*index) != Some(&line.as_str()))
            .map(|(_, line)| line.split(" commitments=").next().unwrap_or(line))
            .collect();
        assert!(
            moved.is_empty() && actual.len() == expected.len(),
            "golden digests moved for {moved:?} (DEV_PROTOCOL = {}); actual:\n{}",
            akita_params::DEV_PROTOCOL,
            actual
                .iter()
                .map(|line| format!("    \"{line}\",\n"))
                .collect::<String>(),
        );
    });
}
