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

/// Digests under the current protocol: Akita main's (v8) bytes for the
/// same rows, which a clear dev build must reproduce.
const CURRENT_PROTOCOL: &[&str] = &[
    "fp128_onehot final=12:1 commitments=585a7fa528f7cdc42bbfee9ad971b2bee0acb2a1f6714f486de340ae06d16a2c proof_len=42590 proof=79d815aed68966d9b08bbf2cf9ac7e305e8e10ce660187e0300316741abab9ef",
    "fp128_dense final=14:1 commitments=e31d644f319cc37df129d0cb1ba9bc191e657b7913595c27dfb4788c036713b2 proof_len=68796 proof=a92a6e6e5617a4e40fddca7cd2284530e6352a27fe09de12e2f52c3100081397",
    "fp128_onehot final=18:1 commitments=fa674c084bc34e59763b2526da5c12c4266b8d6376a528fa2b46b76682ca3981 proof_len=66254 proof=e24649caab7ab6715bcb23979f24df56a45dccc1ee01003b267f9037473daa7c",
    "fp128_onehot final=16:1;precommitted=14:1 commitments=152c73e0607a7d106c5abc1726adac8ac7b5058f849c514c1073bb44b71a8c0a proof_len=65529 proof=39550ec5e5d83612d930c14e92e6bc8a01c42bf201f0420bce6a6475cf6832e8",
    "fp64_dense final=14:1 commitments=8009f78c8031d000921aca10a9a1be5d36831d5df8c9ffa7d2be70fee1402752 proof_len=63818 proof=8d7766eb3a2b6a60023f48afca1f44e7c20dd615dae7d86fa30950f6786fbeb8",
];

/// Digests under the dev protocol (`akita_params::DEV_PROTOCOL`). On v8 the
/// dev wire format is the current one; only its artifact set differs.
const DEV_PROTOCOL: &[&str] = CURRENT_PROTOCOL;

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
    let precommitteds = PrecommittedGroupProfiles::from_profiles(vec![*pre_commitment.profile()])
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
