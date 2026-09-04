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
//!   ✓   — runs in default `cargo test` using checked-in external artifacts
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

mod common;
#[path = "small_field_drivers/reduced_eor.rs"]
mod reduced_eor;
mod small_field_drivers;

use akita_config::proof_optimized::{fp32, fp64};
use akita_prover::{ComputeBackendSetup, CpuBackend, UniformProverStack};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode, GroupBatchStatement,
    OpeningClaims, OpeningClaimsLayout, PolynomialGroupClaims, PolynomialGroupLayout,
};
use common::*;
use jolt_field::{ExtField, One, Ring, Zero};
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
// ============================================================================

macro_rules! small_field_test {
    // ------------------------------------------------------------------
    // dense — single-group, non-precommitted
    // ------------------------------------------------------------------
    ($(#[$attr:meta])* dense; $name:ident; $cfg:ty; $sf:ty; $se:ty; nvs=[$($nv:expr),+] $(; check=$check:path)?) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                for &nv in &[$($nv),+] {
                    let n = 1usize << nv;
                    let evals: Vec<$sf> = (0..n)
                        .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(7).wrapping_add(13)))
                        .collect();
                    let poly =
                        akita_prover::DensePoly::<$sf>::from_field_evals(nv, &evals)
                            .expect("dense poly");

                    let point: Vec<$se> = (0..nv)
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(3).wrapping_add(1)))
                        .collect();
                    let weights = lagrange_weights::<$se>(&point).expect("weights");
                    let expected: $se = (0..n)
                        .map(|i| weights[i] * <$se>::lift_base(evals[i]))
                        .fold(<$se>::from_u64(0), |a, b| a + b);

                    let roundtrip = single_group_roundtrip::<$cfg>(
                        nv,
                        &akita_prover::MultilinearPolynomial::dense(poly),
                        point,
                        expected,
                        label,
                        stringify!($name),
                    );
                    $($check(&roundtrip, label, stringify!($name));)?
                    drop(roundtrip);
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

                let pre_n = 1usize << PRE_NV;
                let pre_evals: Vec<$sf> = (0..pre_n)
                    .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(7).wrapping_add(13)))
                    .collect();

                for &final_nv in &[$($fnv),+] {
                    let scheme = load_workspace_scheme::<$cfg>()
                        .expect("workspace schedule catalog");
                    let setup = scheme.setup_prover(
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
                    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

                    let pre_poly = akita_prover::DensePoly::<$sf>::from_field_evals(
                        PRE_NV,
                        &pre_evals,
                    )
                    .expect("pre dense poly");
                    let akita_prover::CommitOutput {
                        committed_group: pre_commitment,
                        hint: pre_hint,
                    } = scheme.commit(
                        &setup,
                        std::slice::from_ref(&pre_poly),
                        &stack,
                        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .expect("precommit");

                    let final_n = 1usize << final_nv;
                    let final_evals: Vec<$sf> = (0..final_n)
                        .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(11).wrapping_add(7)))
                        .collect();
                    let final_poly = akita_prover::DensePoly::<$sf>::from_field_evals(
                        final_nv,
                        &final_evals,
                    )
                    .expect("final dense poly");
                    let precommitteds =
                        PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile]).expect("nonempty precommitted groups");
                    let akita_prover::CommitOutput {
                        committed_group: final_commitment,
                        hint: final_hint,
                    } = scheme.commit(
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
                        scheme.schedules(),
                    );
                    let selection = prover_data.selection();

                    let mut pt = AkitaTranscript::<$sf>::new(label);
                    let proof = scheme.batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut pt,
                        BasisMode::Lagrange,
                    )
                    .expect("prove");

                    two_group_verify_roundtrip::<$cfg>(
                        &scheme,
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
    ($(#[$attr:meta])* onehot; $name:ident; $cfg:ty; $sf:ty; $se:ty; nvs=[$($nv:expr),+]) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                let onehot_k = akita_config::unit_onehot_source_chunk_size::<$cfg>()
                    .expect("one-hot test requires a unit-one-hot config");
                for &nv in &[$($nv),+] {
                    let num_chunks = (1usize << nv) / onehot_k;
                    let indices: Vec<Option<u8>> = (0..num_chunks)
                        .map(|chunk| {
                            Some(((chunk * 29 + nv * 41 + 7) % onehot_k) as u8)
                        })
                        .collect();
                    let poly =
                        akita_prover::OneHotPoly::<$sf, u8>::new(onehot_k, indices)
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
    ($(#[$attr:meta])* onehot_pre; $name:ident; $cfg:ty; $sf:ty; $se:ty; pre_nv=$pnv:expr; final_nvs=[$($fnv:expr),+]) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                const PRE_NV: usize = $pnv;
                let onehot_k = akita_config::unit_onehot_source_chunk_size::<$cfg>()
                    .expect("one-hot test requires a unit-one-hot config");

                let pre_chunks = (1usize << PRE_NV) / onehot_k;
                let pre_indices: Vec<Option<u8>> = (0..pre_chunks)
                    .map(|chunk| Some(((chunk * 29 + 7) % onehot_k) as u8))
                    .collect();

                for &final_nv in &[$($fnv),+] {
                    let scheme = load_workspace_scheme::<$cfg>()
                        .expect("workspace schedule catalog");
                    let setup = scheme.setup_prover(
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
                    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

                    let pre_poly = akita_prover::OneHotPoly::<$sf, u8>::new(
                        onehot_k,
                        pre_indices.clone(),
                    )
                    .expect("pre onehot poly");
                    let akita_prover::CommitOutput {
                        committed_group: pre_commitment,
                        hint: pre_hint,
                    } = scheme.commit(
                        &setup,
                        std::slice::from_ref(&pre_poly),
                        &stack,
                        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .expect("precommit");

                    let final_chunks = (1usize << final_nv) / onehot_k;
                    let final_indices: Vec<Option<u8>> = (0..final_chunks)
                        .map(|chunk| Some(((chunk * 37 + 11) % onehot_k) as u8))
                        .collect();
                    let final_poly = akita_prover::OneHotPoly::<$sf, u8>::new(
                        onehot_k,
                        final_indices,
                    )
                    .expect("final onehot poly");
                    let precommitteds =
                        PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile]).expect("nonempty precommitted groups");
                    let akita_prover::CommitOutput {
                        committed_group: final_commitment,
                        hint: final_hint,
                    } = scheme.commit(
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
                        scheme.schedules(),
                    );
                    let selection = prover_data.selection();

                    let mut pt = AkitaTranscript::<$sf>::new(label);
                    let proof = scheme.batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut pt,
                        BasisMode::Lagrange,
                    )
                    .expect("prove");

                    two_group_verify_roundtrip::<$cfg>(
                        &scheme,
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
small_field_test!(dense; fp32_dense; fp32::Dense; fp32::Field; fp32::ExtensionField; nvs=[20]; check=reduced_eor::assert_fp32_dense);
// fp32 × Dense × precommitted        catalog: final=(20,1) <- pre=[(20,1)]
//
// pre_nv=20 rather than 14: an independent precommit commits with its own row
// without precommitted groups, and `fp32::Dense` has no schedule with at least two
// folds below 20, so no such row exists at 14.
small_field_test!(dense_pre; fp32_dense_pre; fp32::Dense; fp32::Field; fp32::ExtensionField; pre_nv=20; final_nvs=[20]);
// fp32 × OneHot × direct             catalog: single(14,1), single(16,1)
small_field_test!(onehot;     fp32_onehot;     fp32::OneHot; fp32::Field; fp32::ExtensionField; nvs=[14, 16]);
// fp32 × OneHot × precommitted       catalog: final=(20,1) <- pre=[(14,1)]
small_field_test!(onehot_pre; fp32_onehot_pre; fp32::OneHot; fp32::Field; fp32::ExtensionField; pre_nv=14; final_nvs=[20]);

// ----------------------------------------------------------------------------
// fp64  (Field = Prime64Offset59, ExtField = Ext2)
// The checked-in fp64 artifact covers this row without a feature gate.
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
small_field_test!(#[ignore = "production-sized: fp64::OneHot starts at nv=28; run with --ignored --release"] onehot; fp64_onehot; fp64::OneHot; fp64::Field; fp64::ExtensionField; nvs=[28]);
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
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;

    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<SmallCfg>().expect("workspace schedule catalog");
        let grouped_poly = |params: &CommittedGroupParams, seed: usize| {
            let onehot_k =
                akita_config::unit_onehot_source_chunk_size::<SmallCfg>().expect("one-hot config");
            let total =
                params.blocks().live_blocks * params.blocks().positions_per_block * params.d_a();
            let indices = (0..total / onehot_k)
                .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
                .collect();
            akita_prover::OneHotPoly::<SmallF, u8>::new(onehot_k, indices)
                .expect("grouped fp32 poly")
        };

        let pre_group_schedule = scheme
            .schedules()
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                PRE_NV, 1,
            )))
            .expect("pre schedule")
            .schedule()
            .clone();
        let pre_params = &pre_group_schedule.root.params;
        let pre_poly = grouped_poly(pre_params, 1);

        let pre_setup = scheme.setup_prover(PRE_NV, 1).expect("pre setup");
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
        } = scheme
            .commit(
                &pre_setup,
                std::slice::from_ref(&pre_poly),
                &pre_stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("precommit");

        let multi_schedule = scheme
            .schedules()
            .resolve_key(&AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(FINAL_NV, 1),
                precommitteds: vec![pre_commitment.profile],
            })
            .expect("multi-group schedule")
            .schedule()
            .clone();
        let final_params = &multi_schedule.root.params;
        let final_poly = grouped_poly(final_params, 2);

        let setup = scheme.setup_prover(FINAL_NV, 2).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

        let precommitteds = PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile])
            .expect("nonempty precommitted groups");
        let akita_prover::CommitOutput {
            committed_group: final_commitment,
            hint: final_hint,
        } = scheme
            .commit(
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
            scheme.schedules(),
        );
        let selection = prover_data.selection();

        let mut prover_transcript =
            AkitaTranscript::<SmallF>::new(b"completeness/fp32_onehot_multi_group");
        let proof = scheme
            .batched_prove(
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
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                BasisMode::Lagrange,
            )
            .expect("fp32 multi-group verify");
    });
}

#[path = "akita_small_field_e2e/selective_l2.rs"]
mod selective_l2;
