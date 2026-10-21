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
    "fp128_onehot final=12:1 commitments=d84c67e04b7d74d6ef6543bde7b7746dca6cee0b00b86e3d4ef693cafdc32954 proof_len=36755 proof=ce3794d316781586f36bdb5bfd2aadd008eee5ad5344ad92d39f67bb38dd038a",
    "fp128_dense final=14:1 commitments=49976505877e18fce8ccd10e660e063919641d63863c09425c942bcfc123f0bb proof_len=62430 proof=c9ceab90c9648c9696b14fe15b17c5d4decd154517500c8ee8b5405ff860d408",
    "fp128_onehot final=18:1 commitments=8296c20f4eff9761e9a3b0de022da93bae29d9e3c76c23404ef0ae7bc45d7544 proof_len=61819 proof=6c7c40f7c6e5fbd1e724aa5987a30f222e1861c543239fc0f5514c7a98ddc636",
    "fp128_onehot final=16:1;precommitted=14:1 commitments=9a83dbdd32c30f09a985d0ed489dea18497297a69162fd3b7f18c691152974cb proof_len=61782 proof=000d1d023e990e392aa1a90be1fac648b26ec0c2b6d0c074d56876dc7aba7bd0",
    "fp64_dense final=14:1 commitments=7db9315ca12c861b643c2a3a023084189ae3b8dcea9f9aedf00b30c7ed6abd0c proof_len=58832 proof=a465e03784e99f327ff3fcf7145ba9006957b73cdf11f59ae0ac8fefd2eb6929",
];

/// Digests under the dev protocol (`akita_params::DEV_PROTOCOL`).
const DEV_PROTOCOL: &[&str] = &[
    "fp128_onehot final=12:1 commitments=d84c67e04b7d74d6ef6543bde7b7746dca6cee0b00b86e3d4ef693cafdc32954 proof_len=32642 proof=dcde664098b6f51fdc2011634b3b7f70ac7d57320bd6cad50546d1cb4d630a65",
    "fp128_dense final=14:1 commitments=49976505877e18fce8ccd10e660e063919641d63863c09425c942bcfc123f0bb proof_len=58397 proof=0ce69d1734bc2714d18085aab9cb436f4a1774b8b338aef25a285c53c98d8bcd",
    "fp128_onehot final=18:1 commitments=8296c20f4eff9761e9a3b0de022da93bae29d9e3c76c23404ef0ae7bc45d7544 proof_len=57780 proof=8a1908a916f01a11701e171ce408350e1cc69fb6d305a7838d9cd7011c542ba5",
    "fp128_onehot final=16:1;precommitted=14:1 commitments=9a83dbdd32c30f09a985d0ed489dea18497297a69162fd3b7f18c691152974cb proof_len=57775 proof=5d82f331a949c9bca25250c6ad924e3075f0ac9312ca0b1edea7fcdecae0600f",
    "fp64_dense final=14:1 commitments=7db9315ca12c861b643c2a3a023084189ae3b8dcea9f9aedf00b30c7ed6abd0c proof_len=54855 proof=6e0018441900547195f8b1000a913df4dd85e0b07d0ca2fe71a5541fe293730e",
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
        assert!(
            actual.iter().eq(expected.iter()),
            "golden digests changed (DEV_PROTOCOL = {}); actual:\n{}",
            akita_params::DEV_PROTOCOL,
            actual
                .iter()
                .map(|line| format!("    \"{line}\",\n"))
                .collect::<String>(),
        );
    });
}
