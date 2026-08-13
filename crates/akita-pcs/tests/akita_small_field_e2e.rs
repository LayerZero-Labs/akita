//! Correctness matrix for small-field Akita PCS prove→verify roundtrips.
//!
//! # Group A — Small fields
//!
//! Tests the full cartesian product for configurations where `ExtField ≠ Field`
//! (fp32, fp64).  Because the generic fp128 driver cannot be reused, each cell
//! inlines its own Lagrange-weight opening computation via the `small_field_test!`
//! macro.
//!
//! Legend:
//!   ✓   — runs in default `cargo test` (schedules-default feature)
//!   ign — supported, but production-sized nv; run with `-- --ignored`
//!   NA  — no production schedule row exists; cell is intentionally absent
//!
//! Every cell resolves against a real shipped catalog row. No cell is backed by
//! a schedule added purely to make a test pass: where the production catalog
//! has no row, the cell is NA rather than propped up by a test-only fixture.
//!
//! ```text
//! ╔══════════╦═════════════════════════════╦═════════════════════════════╗
//! ║ field    ║ Dense                       ║ OneHot                      ║
//! ╠══════════╬══════════════╦══════════════╬══════════════╦══════════════╣
//! ║          ║ direct       ║ pre          ║ direct       ║ pre          ║
//! ╠══════════╬══════════════╬══════════════╬══════════════╬══════════════╣
//! ║  fp32    ║ ✓ nv=20      ║ ✓ pre=14     ║ ✓ nv=14,16   ║ ✓ pre=14     ║
//! ║          ║              ║   final=20   ║              ║   final=20   ║
//! ║  fp64    ║ ✓ nv=20      ║ ✓ pre=16     ║ ign nv=28    ║ NA           ║
//! ║          ║              ║   final=20   ║              ║              ║
//! ╚══════════╩══════════════╩══════════════╩══════════════╩══════════════╝
//! ```
//!
//! fp64 × OneHot: the family's smallest production size is nv=28, and it ships
//! no combined precommit+final row. The direct cell is therefore ign and the
//! pre cell NA.
//!
//! fp64 × Dense × pre uses a 16-variable pre-group rather than 14: at pre=14
//! or 15 the prover and the planned schedule disagree on the fold-level-1
//! witness length. That is a pre-existing fp64::Dense issue (the same class as
//! the nv=14 direct mismatch), not something this matrix introduces.
//!
//! # Group E (small-field) — Heterogeneous configurations
//!
//! `fp32_onehot_multi_group`: two precommit groups proved jointly, verifying the
//! multi-group code path with a small field.

#![allow(missing_docs)]
#![cfg(feature = "schedules-default")]

mod common;
mod small_field_drivers;

use akita_config::proof_optimized::{fp32, fp64};
use akita_config::CommitmentConfig;
use akita_field::LiftBase;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend, UniformProverStack};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode, GroupBatchStatement,
    OpeningClaims, OpeningClaimsLayout, PolynomialGroupClaims, PolynomialGroupLayout,
};
use common::*;
use small_field_drivers::*;

// ============================================================================
// small_field_test! — inline driver for small fields (ExtField ≠ Field)
//
// The opening is computed directly using Lagrange weights over the extension
// field rather than the CpuBackend fold kernel — an oracle independent of the
// prover, and necessary anyway since the generic fp128 helper is hardcoded to
// fp128::Field.
//
// The single-group arms delegate their setup/commit/prove/serialize/verify tail
// to `small_field_drivers::single_group_roundtrip`, so only the polynomial and
// its expected opening are written per cell.
//
// Arms:
//   dense          — single-group, non-precommitted, dense polynomial
//   dense_pre      — two-group (precommit + final), dense polynomial
//   onehot         — single-group, non-precommitted, one-hot polynomial
//   onehot_pre     — two-group (precommit + final), one-hot polynomial
//
// Parameters:
//   $name      — test function identifier
//   $cfg       — CommitmentConfig type (e.g. fp32::Dense)
//   $sf        — base field type  (Cfg::Field)
//   $se        — extension field type  (Cfg::ExtField)
//   nvs        — list of num_vars to test (non-precommitted arms)
//   pre_nv     — pre-group num_vars (precommitted arms); per config, because the
//                smallest usable pre size differs between families
//   final_nvs  — list of final-group num_vars (precommitted arms)
//   k          — one-hot group size K (onehot arms)
// ============================================================================

macro_rules! small_field_test {
    // ------------------------------------------------------------------
    // dense — single-group, non-precommitted
    // ------------------------------------------------------------------
    ($(#[$attr:meta])* dense; $name:ident; $cfg:ty; $sf:ty; $se:ty; nvs=[$($nv:expr),+]) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                for &nv in &[$($nv),+] {
                    let n = 1usize << nv;
                    let opening_batch =
                        OpeningClaimsLayout::new(nv, 1).expect("opening batch");
                    let layout = <$cfg as CommitmentConfig>::select_schedule_for_opening(
                        &opening_batch,
                    )
                    .expect("layout")
                    .into_schedule()
                    .root
                    .params
                    .final_group
                    .commitment;
                    let d = layout.d_a();

                    let evals: Vec<$sf> = (0..n)
                        .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(7).wrapping_add(13)))
                        .collect();
                    let poly =
                        akita_prover::DensePoly::<$sf>::from_field_evals(nv, d, &evals)
                            .expect("dense poly");

                    let point: Vec<$se> = (0..nv)
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(3).wrapping_add(1)))
                        .collect();
                    let weights = lagrange_weights::<$se>(&point).expect("weights");
                    let expected: $se = (0..n)
                        .map(|i| weights[i] * <$se>::lift_base(evals[i]))
                        .fold(<$se>::from_u64(0), |a, b| a + b);

                    single_group_roundtrip::<$cfg>(
                        nv,
                        &akita_prover::MultilinearPolynomial::dense(poly),
                        point,
                        expected,
                        label,
                        stringify!($name),
                    );
                }
            });
        }
    };

    // ------------------------------------------------------------------
    // dense_pre — two-group precommitted, dense polynomial
    // pre-group: nv=pre_nv  |  final-group: nv from final_nvs list
    // ------------------------------------------------------------------
    ($(#[$attr:meta])* dense_pre; $name:ident; $cfg:ty; $sf:ty; $se:ty; pre_nv=$pnv:expr; final_nvs=[$($fnv:expr),+]) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                const PRE_NV: usize = $pnv;

                // An independent precommit commits with its own row without
                // precommitted groups, so take the ring dimension from that row.
                let pre_d = <$cfg as CommitmentConfig>::profile_without_precommitted_groups(
                    PolynomialGroupLayout::new(PRE_NV, 1),
                )
                .expect("pre profile without precommitted groups")
                .inner_commit_matrix
                .ring_dimension();
                let pre_n = 1usize << PRE_NV;
                let pre_evals: Vec<$sf> = (0..pre_n)
                    .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(7).wrapping_add(13)))
                    .collect();

                for &final_nv in &[$($fnv),+] {
                    let setup = AkitaCommitmentScheme::<$cfg>::setup_prover(
                        final_nv.max(PRE_NV),
                        2,
                    )
                    .expect("setup");
                    let prepared =
                        CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
                    let stack = UniformProverStack::uniform(
                        &CpuBackend::DEFAULT,
                        &prepared,
                        setup.expanded.as_ref(),
                    )
                    .expect("stack");
                    let verifier_setup =
                        AkitaCommitmentScheme::<$cfg>::setup_verifier(&setup)
                            .expect("verifier setup");

                    let pre_poly = akita_prover::DensePoly::<$sf>::from_field_evals(
                        PRE_NV,
                        pre_d,
                        &pre_evals,
                    )
                    .expect("pre dense poly");
                    let akita_prover::CommitOutput {
                        committed_group: pre_commitment,
                        hint: pre_hint,
                    } = AkitaCommitmentScheme::<$cfg>::commit(
                        &setup,
                        std::slice::from_ref(&pre_poly),
                        &stack,
                        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .expect("precommit");

                    let multi_schedule = <$cfg as CommitmentConfig>::select_schedule_for_key(
                        &AkitaScheduleLookupKey {
                            final_group: PolynomialGroupLayout::new(final_nv, 1),
                            precommitteds: vec![pre_commitment.profile],
                        },
                    )
                    .expect("multi-group schedule")
                    .into_schedule();
                    let final_d = multi_schedule.root.params.final_group.commitment.d_a();
                    let final_n = 1usize << final_nv;
                    let final_evals: Vec<$sf> = (0..final_n)
                        .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(11).wrapping_add(7)))
                        .collect();
                    let final_poly = akita_prover::DensePoly::<$sf>::from_field_evals(
                        final_nv,
                        final_d,
                        &final_evals,
                    )
                    .expect("final dense poly");
                    let precommitteds =
                        PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile]).expect("nonempty precommitted groups");
                    let akita_prover::CommitOutput {
                        committed_group: final_commitment,
                        hint: final_hint,
                    } = AkitaCommitmentScheme::<$cfg>::commit(
                        &setup,
                        std::slice::from_ref(&final_poly),
                        &stack,
                        akita_prover::GroupContext::scheduler_with_precommitted_groups(
                            &precommitteds,
                        ),
                    )
                    .expect("final commit");

                    let point: Vec<$se> = (0..final_nv.max(PRE_NV))
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(3).wrapping_add(1)))
                        .collect();
                    let pre_weights =
                        lagrange_weights::<$se>(&point[..PRE_NV]).expect("pre weights");
                    let pre_opening: $se = (0..pre_n)
                        .map(|i| pre_weights[i] * <$se>::lift_base(pre_evals[i]))
                        .fold(<$se>::from_u64(0), |a, b| a + b);
                    let final_weights =
                        lagrange_weights::<$se>(&point[..final_nv]).expect("final weights");
                    let final_opening: $se = (0..final_n)
                        .map(|i| final_weights[i] * <$se>::lift_base(final_evals[i]))
                        .fold(<$se>::from_u64(0), |a, b| a + b);

                    let pre_refs = [&pre_poly];
                    let final_refs = [&final_poly];
                    let prover_data = selected_prover_data::<$cfg, _>(
                        OpeningClaims::from_groups(vec![
                            PolynomialGroupClaims::new(
                                point[..PRE_NV].to_vec(),
                                vec![pre_opening],
                                pre_commitment.clone(),
                            )
                            .expect("pre prover group"),
                            PolynomialGroupClaims::new(
                                point[..final_nv].to_vec(),
                                vec![final_opening],
                                final_commitment.clone(),
                            )
                            .expect("final prover group"),
                        ])
                        .expect("prover claims"),
                        vec![pre_hint, final_hint],
                        vec![&pre_refs[..], &final_refs[..]],
                    );
                    let selection = prover_data.selection();

                    let mut pt = AkitaTranscript::<$sf>::new(label);
                    let proof = AkitaCommitmentScheme::<$cfg>::batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut pt,
                        BasisMode::Lagrange,
                    )
                    .expect("prove");

                    two_group_verify_roundtrip::<$cfg>(
                        &proof,
                        &verifier_setup,
                        selection,
                        (&pre_commitment, &point[..PRE_NV], pre_opening),
                        (&final_commitment, &point[..final_nv], final_opening),
                        label,
                        &format!(
                            "{} pre_nv={PRE_NV} final_nv={final_nv}",
                            stringify!($name)
                        ),
                    );
                }
            });
        }
    };

    // ------------------------------------------------------------------
    // onehot — single-group, non-precommitted, one-hot polynomial
    // ------------------------------------------------------------------
    ($(#[$attr:meta])* onehot; $name:ident; $cfg:ty; $sf:ty; $se:ty; nvs=[$($nv:expr),+]; k=$k:expr) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                let onehot_k: usize = $k;
                for &nv in &[$($nv),+] {
                    let opening_batch =
                        OpeningClaimsLayout::new(nv, 1).expect("opening batch");
                    let layout = <$cfg as CommitmentConfig>::select_schedule_for_opening(
                        &opening_batch,
                    )
                    .expect("layout")
                    .into_schedule()
                    .root
                    .params
                    .final_group
                    .commitment;
                    let d = layout.d_a();
                    let num_chunks = (1usize << nv) / onehot_k;
                    let indices: Vec<Option<u8>> = (0..num_chunks)
                        .map(|chunk| {
                            Some(((chunk * 29 + nv * 41 + 7) % onehot_k) as u8)
                        })
                        .collect();
                    let poly =
                        akita_prover::OneHotPoly::<$sf, u8>::new(onehot_k, d, indices)
                            .expect("onehot poly");

                    let point: Vec<$se> = (0..nv)
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
                        .collect();
                    let expected = onehot_opening_lagrange(&poly, &point);

                    single_group_roundtrip::<$cfg>(
                        nv,
                        &akita_prover::MultilinearPolynomial::onehot(poly),
                        point,
                        expected,
                        label,
                        stringify!($name),
                    );
                }
            });
        }
    };

    // ------------------------------------------------------------------
    // onehot_pre — two-group precommitted, one-hot polynomial
    // pre-group: nv=pre_nv  |  final-group: nv from final_nvs list
    // ------------------------------------------------------------------
    ($(#[$attr:meta])* onehot_pre; $name:ident; $cfg:ty; $sf:ty; $se:ty; pre_nv=$pnv:expr; final_nvs=[$($fnv:expr),+]; k=$k:expr) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                const PRE_NV: usize = $pnv;
                let onehot_k: usize = $k;

                // An independent precommit commits with its own row without
                // precommitted groups, so take the ring dimension from that row.
                let pre_d = <$cfg as CommitmentConfig>::profile_without_precommitted_groups(
                    PolynomialGroupLayout::new(PRE_NV, 1),
                )
                .expect("pre profile without precommitted groups")
                .inner_commit_matrix
                .ring_dimension();
                let pre_chunks = (1usize << PRE_NV) / onehot_k;
                let pre_indices: Vec<Option<u8>> = (0..pre_chunks)
                    .map(|chunk| Some(((chunk * 29 + 7) % onehot_k) as u8))
                    .collect();

                for &final_nv in &[$($fnv),+] {
                    let setup = AkitaCommitmentScheme::<$cfg>::setup_prover(
                        final_nv.max(PRE_NV),
                        2,
                    )
                    .expect("setup");
                    let prepared =
                        CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
                    let stack = UniformProverStack::uniform(
                        &CpuBackend::DEFAULT,
                        &prepared,
                        setup.expanded.as_ref(),
                    )
                    .expect("stack");
                    let verifier_setup =
                        AkitaCommitmentScheme::<$cfg>::setup_verifier(&setup)
                            .expect("verifier setup");

                    let pre_poly = akita_prover::OneHotPoly::<$sf, u8>::new(
                        onehot_k,
                        pre_d,
                        pre_indices.clone(),
                    )
                    .expect("pre onehot poly");
                    let akita_prover::CommitOutput {
                        committed_group: pre_commitment,
                        hint: pre_hint,
                    } = AkitaCommitmentScheme::<$cfg>::commit(
                        &setup,
                        std::slice::from_ref(&pre_poly),
                        &stack,
                        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .expect("precommit");

                    let multi_schedule = <$cfg as CommitmentConfig>::select_schedule_for_key(
                        &AkitaScheduleLookupKey {
                            final_group: PolynomialGroupLayout::new(final_nv, 1),
                            precommitteds: vec![pre_commitment.profile],
                        },
                    )
                    .expect("multi-group schedule")
                    .into_schedule();
                    let final_d = multi_schedule.root.params.final_group.commitment.d_a();
                    let final_chunks = (1usize << final_nv) / onehot_k;
                    let final_indices: Vec<Option<u8>> = (0..final_chunks)
                        .map(|chunk| Some(((chunk * 37 + 11) % onehot_k) as u8))
                        .collect();
                    let final_poly = akita_prover::OneHotPoly::<$sf, u8>::new(
                        onehot_k,
                        final_d,
                        final_indices,
                    )
                    .expect("final onehot poly");
                    let precommitteds =
                        PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile]).expect("nonempty precommitted groups");
                    let akita_prover::CommitOutput {
                        committed_group: final_commitment,
                        hint: final_hint,
                    } = AkitaCommitmentScheme::<$cfg>::commit(
                        &setup,
                        std::slice::from_ref(&final_poly),
                        &stack,
                        akita_prover::GroupContext::scheduler_with_precommitted_groups(
                            &precommitteds,
                        ),
                    )
                    .expect("final commit");

                    let point: Vec<$se> = (0..final_nv.max(PRE_NV))
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
                        .collect();
                    let pre_weights =
                        lagrange_weights::<$se>(&point[..PRE_NV]).expect("pre weights");
                    let pre_opening: $se = pre_poly
                        .indices()
                        .iter()
                        .enumerate()
                        .filter_map(|(chunk, hot)| {
                            hot.map(|idx| pre_weights[chunk * onehot_k + usize::from(idx)])
                        })
                        .fold(<$se>::from_u64(0), |a, b| a + b);
                    let final_weights =
                        lagrange_weights::<$se>(&point[..final_nv]).expect("final weights");
                    let final_opening: $se = final_poly
                        .indices()
                        .iter()
                        .enumerate()
                        .filter_map(|(chunk, hot)| {
                            hot.map(|idx| final_weights[chunk * onehot_k + usize::from(idx)])
                        })
                        .fold(<$se>::from_u64(0), |a, b| a + b);

                    let pre_refs = [&pre_poly];
                    let final_refs = [&final_poly];
                    let prover_data = selected_prover_data::<$cfg, _>(
                        OpeningClaims::from_groups(vec![
                            PolynomialGroupClaims::new(
                                point[..PRE_NV].to_vec(),
                                vec![pre_opening],
                                pre_commitment.clone(),
                            )
                            .expect("pre prover group"),
                            PolynomialGroupClaims::new(
                                point[..final_nv].to_vec(),
                                vec![final_opening],
                                final_commitment.clone(),
                            )
                            .expect("final prover group"),
                        ])
                        .expect("prover claims"),
                        vec![pre_hint, final_hint],
                        vec![&pre_refs[..], &final_refs[..]],
                    );
                    let selection = prover_data.selection();

                    let mut pt = AkitaTranscript::<$sf>::new(label);
                    let proof = AkitaCommitmentScheme::<$cfg>::batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut pt,
                        BasisMode::Lagrange,
                    )
                    .expect("prove");

                    two_group_verify_roundtrip::<$cfg>(
                        &proof,
                        &verifier_setup,
                        selection,
                        (&pre_commitment, &point[..PRE_NV], pre_opening),
                        (&final_commitment, &point[..final_nv], final_opening),
                        label,
                        &format!(
                            "{} pre_nv={PRE_NV} final_nv={final_nv}",
                            stringify!($name)
                        ),
                    );
                }
            });
        }
    };
}

// ============================================================================
// GROUP A — Small fields (fp32, fp64)
//
// Cartesian product: field × {Dense, OneHot} × {direct, precommitted}
// Opening computed via Lagrange weights over the extension field.
// ============================================================================

// ----------------------------------------------------------------------------
// fp32  (Field = Prime32Offset99, ExtField = FpExt4)
// ----------------------------------------------------------------------------

// fp32 × Dense × direct              catalog: single(20,1)
small_field_test!(dense;     fp32_dense;     fp32::Dense;  fp32::Field; fp32::ExtensionField; nvs=[20]);
// fp32 × Dense × precommitted        catalog: final=(20,1) <- pre=[(20,1)]
//
// pre_nv=20 rather than 14: an independent precommit commits with its own row
// without precommitted groups, and `fp32::Dense` has no schedule with at least two
// folds below 20, so no such row exists at 14.
small_field_test!(dense_pre; fp32_dense_pre; fp32::Dense; fp32::Field; fp32::ExtensionField; pre_nv=20; final_nvs=[20]);
// fp32 × OneHot × direct             catalog: single(14,1), single(16,1)
small_field_test!(onehot;     fp32_onehot;     fp32::OneHot; fp32::Field; fp32::ExtensionField; nvs=[14, 16]; k=256);
// fp32 × OneHot × precommitted       catalog: final=(20,1) <- pre=[(14,1)]
small_field_test!(onehot_pre; fp32_onehot_pre; fp32::OneHot; fp32::Field; fp32::ExtensionField; pre_nv=14; final_nvs=[20]; k=256);

// ----------------------------------------------------------------------------
// fp64  (Field = Prime64Offset59, ExtField = Ext2)
// Also covered by schedules-default; no extra feature gate needed.
// ----------------------------------------------------------------------------

// fp64 × Dense × direct              catalog: single(20,1)
// (nv=14 has a pre-existing witness mismatch; use nv=20)
small_field_test!(dense;     fp64_dense;     fp64::Dense;  fp64::Field; fp64::ExtensionField; nvs=[20]);
// fp64 × Dense × precommitted        catalog: final=(20,1) <- pre=[(16,1)]
// pre_nv=16 specifically: with pre_nv=14 or 15 the prover and the planned
// schedule disagree on the fold-level-1 witness length (expected 3203968,
// actual 3204096). Same class as the pre-existing fp64::Dense nv=14 mismatch
// noted above; tracked separately, not introduced here.
small_field_test!(dense_pre; fp64_dense_pre; fp64::Dense; fp64::Field; fp64::ExtensionField; pre_nv=16; final_nvs=[20]);
// fp64 × OneHot × direct             catalog: single(28,1)
//
// The smallest fp64::OneHot production size is nv=28, so this cell is
// production-sized and skipped by default; run it with `-- --ignored`. It is
// runnable: the independent oracle no longer materializes a 2^28 weight table,
// which is what previously made it infeasible.
small_field_test!(#[ignore = "production-sized: fp64::OneHot starts at nv=28; run with --ignored --release"] onehot; fp64_onehot; fp64::OneHot; fp64::Field; fp64::ExtensionField; nvs=[28]; k=256);
//
// fp64 × OneHot × precommitted — NA. The fp64::OneHot catalog ships no combined
// precommit+final row, and its smallest final size is nv=28. Adding one purely
// to make this cell run would widen the shipped production schedule surface
// (it would also pull ring dimension 128 into the fp64 one-hot catalog), so the
// cell is intentionally absent rather than backed by a test-only schedule.

// ============================================================================
// GROUP E (small-field) — fp32 multi-group
//
// fp32 one-hot: two separate commitment groups (precommit + final) proved jointly.
// ============================================================================

// fp32 one-hot: two separate commitment groups (precommit + final) proved jointly.
#[test]
fn fp32_onehot_multi_group() {
    type SmallCfg = fp32::OneHot;
    type SmallF = fp32::Field;
    type SmallE = fp32::ExtensionField;
    type SmallScheme = AkitaCommitmentScheme<SmallCfg>;
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;

    init_rayon_pool();
    run_on_large_stack(|| {
        let grouped_poly = |params: &CommittedGroupParams, seed: usize| {
            let onehot_k = 256usize;
            let total = params.num_live_blocks * params.num_positions_per_block * params.d_a();
            let indices = (0..total / onehot_k)
                .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
                .collect();
            akita_prover::OneHotPoly::<SmallF, u8>::new(onehot_k, params.d_a(), indices)
                .expect("grouped fp32 poly")
        };

        let pre_group_schedule = SmallCfg::select_schedule_for_key(
            &AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(PRE_NV, 1)),
        )
        .expect("pre schedule")
        .into_schedule();
        let pre_params = &pre_group_schedule.root.params.final_group.commitment;
        let pre_poly = grouped_poly(pre_params, 1);

        let pre_setup = SmallScheme::setup_prover(PRE_NV, 1).expect("pre setup");
        let pre_prepared = CpuBackend::DEFAULT
            .prepare_setup(&pre_setup)
            .expect("prepared");
        let pre_stack = UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &pre_prepared,
            pre_setup.expanded.as_ref(),
        )
        .expect("pre stack");
        let akita_prover::CommitOutput {
            committed_group: pre_commitment,
            hint: pre_hint,
        } = SmallScheme::commit(
            &pre_setup,
            std::slice::from_ref(&pre_poly),
            &pre_stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("precommit");

        let multi_schedule = SmallCfg::select_schedule_for_key(&AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(FINAL_NV, 1),
            precommitteds: vec![pre_commitment.profile],
        })
        .expect("multi-group schedule")
        .into_schedule();
        let final_params = &multi_schedule.root.params.final_group.commitment;
        let final_poly = grouped_poly(final_params, 2);

        let setup = SmallScheme::setup_prover(FINAL_NV, 2).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup = SmallScheme::setup_verifier(&setup).expect("verifier setup");

        let precommitteds = PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile])
            .expect("nonempty precommitted groups");
        let akita_prover::CommitOutput {
            committed_group: final_commitment,
            hint: final_hint,
        } = SmallScheme::commit(
            &setup,
            std::slice::from_ref(&final_poly),
            &stack,
            akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
        )
        .expect("final commit");

        let mut pre_point = (0..PRE_NV)
            .map(|i| SmallE::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
            .collect::<Vec<_>>();
        pre_point[0] += SmallE::one();
        let final_point = (0..FINAL_NV)
            .map(|i| SmallE::from_u64((i as u64).wrapping_mul(5).wrapping_add(2)))
            .collect::<Vec<_>>();
        let pre_opening = onehot_opening_lagrange(&pre_poly, &pre_point);
        let final_opening = onehot_opening_lagrange(&final_poly, &final_point);

        let pre_refs = [&pre_poly];
        let final_refs = [&final_poly];
        let prover_data = selected_prover_data::<SmallCfg, _>(
            OpeningClaims::from_groups(vec![
                PolynomialGroupClaims::new(
                    pre_point.clone(),
                    vec![pre_opening],
                    pre_commitment.clone(),
                )
                .expect("pre prover group"),
                PolynomialGroupClaims::new(
                    final_point.clone(),
                    vec![final_opening],
                    final_commitment.clone(),
                )
                .expect("final prover group"),
            ])
            .expect("prover claims"),
            vec![pre_hint, final_hint],
            vec![&pre_refs[..], &final_refs[..]],
        );
        let selection = prover_data.selection();

        let mut prover_transcript =
            AkitaTranscript::<SmallF>::new(b"completeness/fp32_onehot_multi_group");
        let proof = SmallScheme::batched_prove(
            &setup,
            prover_data,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("fp32 multi-group prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_uncompressed(&mut bytes).expect("serialize");
        let decoded =
            AkitaBatchedProof::<SmallF, SmallE>::deserialize_uncompressed(&bytes[..], &shape)
                .expect("deserialize");

        let verify_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(pre_point, vec![pre_opening], &pre_commitment)
                .expect("pre verifier group"),
            PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
                .expect("final verifier group"),
        ])
        .expect("verifier claims");
        let mut verifier_transcript =
            AkitaTranscript::<SmallF>::new(b"completeness/fp32_onehot_multi_group");
        SmallScheme::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            GroupBatchStatement::new(selection, verify_claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("fp32 multi-group verify");
    });
}

fn fp32_l2_onehot_poly(
    params: &CommittedGroupParams,
    seed: usize,
) -> akita_prover::OneHotPoly<fp32::Field, u8> {
    let onehot_k = 256;
    let total_field = params
        .num_live_blocks
        .checked_mul(params.num_positions_per_block)
        .and_then(|count| count.checked_mul(params.d_a()))
        .expect("fp32 L2 fixture length");
    assert_eq!(total_field % onehot_k, 0);
    let indices = (0..total_field / onehot_k)
        .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
        .collect();
    akita_prover::OneHotPoly::new(onehot_k, params.d_a(), indices)
        .expect("fp32 L2 one-hot polynomial")
}

fn encode_test_golomb_rice(values: &[i64], rice_low_bits: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut bit_position = 0usize;
    let mut write_bit = |bit: bool| {
        let byte_index = bit_position / 8;
        if byte_index == bytes.len() {
            bytes.push(0);
        }
        if bit {
            bytes[byte_index] |= 1 << (bit_position % 8);
        }
        bit_position += 1;
    };
    for &value in values {
        let zigzag = ((value << 1) ^ (value >> 63)) as u64;
        let quotient = zigzag >> rice_low_bits;
        for _ in 0..quotient {
            write_bit(true);
        }
        write_bit(false);
        let remainder = zigzag & ((1u64 << rice_low_bits) - 1);
        for bit in 0..rice_low_bits {
            write_bit((remainder >> bit) & 1 == 1);
        }
    }
    bytes
}

#[test]
fn fp32_ext4_multiblock_l2_pcs_roundtrip_and_stage2_rejections() {
    type Cfg = fp32::OneHot;
    type F = fp32::Field;
    type E = fp32::ExtensionField;
    type Scheme = AkitaCommitmentScheme<Cfg>;
    const NUM_VARS: usize = 28;
    const LABEL: &[u8] = b"test/fp32-ext4-multiblock-l2-pcs";

    init_rayon_pool();
    run_on_large_stack(|| {
        let opening_layout = OpeningClaimsLayout::new(NUM_VARS, 1).expect("L2 opening layout");
        let schedule = Cfg::select_schedule_for_opening(&opening_layout)
            .expect("shipped L2 schedule")
            .into_schedule();
        let l2_step = schedule
            .recursive_folds
            .iter()
            .find(|step| {
                matches!(
                    step.params.witness.inner_commit_matrix.security_route(),
                    akita_types::InnerCommitSecurityRoute::L2 { .. }
                )
            })
            .expect("schedule-selected small-field L2 fold");
        assert_eq!(l2_step.params.witness.d_a(), 128);
        assert_eq!(
            l2_step.params.witness.fold_challenge_config,
            akita_challenges::D128_SELECTIVE_L2_CHALLENGE_CONFIG,
        );
        assert_eq!(
            akita_challenges::selective_l2_operator_norm_rejection(
                128,
                &l2_step.params.witness.fold_challenge_config,
            ),
            Some(akita_challenges::OperatorNormRejection::D128_SELECTIVE_L2),
        );
        let akita_types::InnerCommitSecurityRoute::L2 {
            norm_proof_shape, ..
        } = l2_step.params.witness.inner_commit_matrix.security_route()
        else {
            unreachable!("selected route checked above")
        };
        assert!(
            norm_proof_shape
                .limb_gram_layout()
                .expect("checked LimbGram shape")
                .expect("shipped small-field route must use LimbGram")
                .block_count()
                > 1
        );

        let poly = fp32_l2_onehot_poly(&schedule.root.params.final_group.commitment, 3);
        let point = (0..NUM_VARS)
            .map(|i| E::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
            .collect::<Vec<_>>();
        let opening = onehot_opening_lagrange(&poly, &point);
        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("L2 prover setup");
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepared L2 setup");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("L2 prover stack");
        let verifier_setup = Scheme::setup_verifier(&setup).expect("L2 verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = Scheme::commit(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("L2 commitment");
        let poly_refs = [&poly];
        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![E::zero()],
            commitment.clone(),
        )
        .expect("L2 prover group")])
        .expect("L2 prover claims");
        let mut prover_transcript = AkitaTranscript::<F>::new(LABEL);
        let proof = Scheme::batched_prove(
            &setup,
            selected_prover_data::<Cfg, _>(prover_claims, vec![hint], vec![&poly_refs]),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("small-field L2 proof");
        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof
            .serialize_uncompressed(&mut bytes)
            .expect("serialize small-field L2 PCS proof");
        let proof = AkitaBatchedProof::<F, E>::deserialize_uncompressed(&bytes[..], &shape)
            .expect("deserialize small-field L2 PCS proof");
        let l2_index = proof
            .recursive_folds
            .iter()
            .position(|fold| fold.stage1.norm_proof.is_some())
            .expect("proof must carry the selected L2 norm");
        assert!(
            proof.recursive_folds[l2_index]
                .stage1
                .norm_proof
                .as_ref()
                .expect("L2 norm proof")
                .subclaims
                .len()
                > 1
        );

        let verify = |candidate: &AkitaBatchedProof<F, E>| {
            let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                &commitment,
            )
            .expect("L2 verifier group")])
            .expect("L2 verifier claims");
            let mut transcript = AkitaTranscript::<F>::new(LABEL);
            Scheme::batched_verify(
                candidate,
                &verifier_setup,
                &mut transcript,
                selected_statement::<Cfg>(claims),
                BasisMode::Lagrange,
            )
        };
        verify(&proof).expect("verify serialized small-field L2 PCS proof");

        let mut bad_subclaim = proof.clone();
        bad_subclaim.recursive_folds[l2_index]
            .stage1
            .norm_proof
            .as_mut()
            .expect("L2 norm proof")
            .subclaims[0] += E::one();
        assert!(verify(&bad_subclaim).is_err());

        let mut bad_virtual = proof.clone();
        bad_virtual.recursive_folds[l2_index]
            .stage1
            .norm_proof
            .as_mut()
            .expect("L2 norm proof")
            .virtual_evaluations[0] += E::one();
        assert!(verify(&bad_virtual).is_err());

        let mut bad_nonce = proof.clone();
        bad_nonce.recursive_folds[l2_index].fold_grind_nonce += 1;
        assert!(verify(&bad_nonce).is_err());

        let mut bad_stage2 = proof;
        bad_stage2.recursive_folds[l2_index]
            .stage2
            .sumcheck_proof
            .round_polys[0]
            .coeffs_except_linear_term[0] += E::one();
        assert!(verify(&bad_stage2).is_err());
    });
}

#[test]
fn fp32_nv20_shipped_d128_terminal_l2_roundtrip_and_rejections() {
    type Cfg = fp32::OneHot;
    type F = fp32::Field;
    type E = fp32::ExtensionField;
    type Scheme = AkitaCommitmentScheme<Cfg>;
    const NUM_VARS: usize = 20;
    const LABEL: &[u8] = b"test/fp32-nv20-shipped-d128-terminal-l2";

    init_rayon_pool();
    run_on_large_stack(|| {
        let opening_layout = OpeningClaimsLayout::new(NUM_VARS, 1).expect("terminal L2 layout");
        let schedule = Cfg::select_schedule_for_opening(&opening_layout)
            .expect("shipped fp32 schedule")
            .into_schedule();
        let terminal_params = &schedule.terminal.params.witness;
        let response_l2_sq_cap = terminal_params
            .response_l2_sq_cap()
            .expect("nv20 must ship a direct terminal L2 cap");
        assert_eq!(terminal_params.d_a(), 128);
        assert_eq!(
            schedule.terminal.params.sparse_challenge_config,
            akita_challenges::D128_SELECTIVE_L2_CHALLENGE_CONFIG,
        );
        assert_eq!(
            akita_challenges::selective_l2_operator_norm_rejection(
                terminal_params.d_a(),
                &schedule.terminal.params.sparse_challenge_config,
            ),
            Some(akita_challenges::OperatorNormRejection::D128_SELECTIVE_L2),
        );

        let poly = fp32_l2_onehot_poly(&schedule.root.params.final_group.commitment, 9);
        let point = (0..NUM_VARS)
            .map(|i| E::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
            .collect::<Vec<_>>();
        let opening = onehot_opening_lagrange(&poly, &point);
        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("terminal L2 prover setup");
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepared terminal L2 setup");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("terminal L2 prover stack");
        let verifier_setup = Scheme::setup_verifier(&setup).expect("terminal L2 verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = Scheme::commit(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("terminal L2 commitment");
        let poly_refs = [&poly];
        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![E::zero()],
            commitment.clone(),
        )
        .expect("terminal L2 prover group")])
        .expect("terminal L2 prover claims");
        let mut prover_transcript = AkitaTranscript::<F>::new(LABEL);
        let proof = Scheme::batched_prove(
            &setup,
            selected_prover_data::<Cfg, _>(prover_claims, vec![hint], vec![&poly_refs]),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("shipped D128 terminal L2 proof");

        let verify = |candidate: &AkitaBatchedProof<F, E>| {
            let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                &commitment,
            )
            .expect("terminal L2 verifier group")])
            .expect("terminal L2 verifier claims");
            let mut transcript = AkitaTranscript::<F>::new(LABEL);
            Scheme::batched_verify(
                candidate,
                &verifier_setup,
                &mut transcript,
                selected_statement::<Cfg>(claims),
                BasisMode::Lagrange,
            )
        };
        verify(&proof).expect("verify shipped D128 terminal L2 proof");

        let mut bad_nonce = proof.clone();
        bad_nonce.terminal.fold_grind_nonce = bad_nonce
            .terminal
            .fold_grind_nonce
            .checked_add(1)
            .expect("terminal nonce increment");
        assert!(verify(&bad_nonce).is_err());

        let mut over_cap = proof;
        let group = *over_cap
            .terminal
            .terminal_response
            .layout
            .groups
            .first()
            .expect("single terminal group");
        let payload = over_cap
            .terminal
            .terminal_response
            .z_payloads
            .first_mut()
            .expect("terminal z payload");
        let mut values = akita_types::decode_terminal_z_golomb_payload(payload, &group)
            .expect("honest terminal z decode")
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        let coordinate = i64::try_from(group.z_admission_linf_cap).expect("i64 terminal cap");
        let coordinate_sq = u128::try_from(coordinate * coordinate).expect("positive square");
        let mut forced_l2_sq = 0u128;
        for value in &mut values {
            *value = coordinate;
            forced_l2_sq += coordinate_sq;
            if forced_l2_sq > response_l2_sq_cap {
                break;
            }
        }
        assert!(forced_l2_sq > response_l2_sq_cap);
        *payload = encode_test_golomb_rice(&values, group.z_rice_low_bits);
        assert!(payload.len() <= group.z_payload_bytes);
        assert!(verify(&over_cap).is_err());
    });
}
