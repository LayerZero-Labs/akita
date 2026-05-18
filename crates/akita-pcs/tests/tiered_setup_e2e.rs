//! End-to-end tests for the book §5.4 tiered setup commitment path.
//!
//! These tests exercise the routed tiered fourth-root verifier of book
//! Figure 12 and §5.4: the prover splits the shared setup polynomial
//! `S` into `k = f²` chunks under shared `D_chunk/B_chunk` matrices,
//! binds the per-chunk B-side commitments via a tier-3 meta commit,
//! and routes both into the next fold level's joint multi-claim
//! recursive open. The verifier mirrors by deterministically re-deriving
//! the per-chunk and meta commitments from the public setup matrix on
//! every verification call.
//!
//! The wrapper config is [`akita_config::ClaimReductionCfg<Base, F>`]
//! at the chosen tiered shrink factor. The small tests at `f = 2,
//! k = 4` exercise the 10 stage-2 check groups end-to-end at the
//! tightest schedule that still runs under typical CI memory; the
//! `tiered_production_prove_verify` test exercises the book sweet spot
//! `f = 8, k = 64`.

#![allow(missing_docs)]

mod common;

use akita_config::{
    ClaimReductionCascadeCfg, ClaimReductionCfg, CommitmentConfig, TieredClaimReductionCfg,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_transcript::Blake2bTranscript;
use akita_verifier::CommitmentVerifier;
use common::*;
use std::sync::Mutex;
use std::time::Instant;

static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

type TieredDenseSmallCfg = ClaimReductionCfg<DenseCfg, 2>;
type TieredOneHotSmallCfg = ClaimReductionCfg<OneHotCfg, 2>;
type TieredDenseMidCfg = ClaimReductionCfg<DenseCfg, 4>;
type TieredOneHotMidCfg = ClaimReductionCfg<OneHotCfg, 4>;
type TieredDenseProductionCfg = TieredClaimReductionCfg<DenseCfg>;

/// Book §5.8 line 1170 headline cascade: `f_{L0} = 8`, `f_{L1} = 4`,
/// `f_{Lk} = 1` for `k ≥ 2`. This is the configuration the book
/// claims for the "T1+T2 @ L0+L1" speedup row (Table 1141–1158: 16×
/// at NV=32, 35× at NV=38, 265× at NV=44).
type DenseCascadeCfg = ClaimReductionCascadeCfg<DenseCfg, 8, 4>;
type OneHotCascadeCfg = ClaimReductionCascadeCfg<OneHotCfg, 8, 4>;

/// Book §5.8 cascade probe: scan `f_{L0}=2`, `f_{L1}=2` at small
/// NV to validate the per-level tier plumbing without hitting the
/// production OOM. Smaller `f` means easier scheduling; the point
/// is to exercise the cascade path itself.
type DenseCascadeSmallCfg = ClaimReductionCascadeCfg<DenseCfg, 2, 2>;
type OneHotCascadeSmallCfg = ClaimReductionCascadeCfg<OneHotCfg, 2, 2>;

#[test]
fn tiered_dense_prove_verify_small() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        // Tier f=2 requires the SETUP polynomial's `(m_S, r_S)` to be
        // >= log_2(f) = 1 on each axis, which forces NV >= 19 at the
        // dense d=128 config (see `probe_min_viable_nv_for_tier_f2`).
        // NV=19 is half the prover work of NV=20 and exercises the
        // same tiered routing.
        const NV: usize = 19;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseSmallCfg>;

        let layout = TieredDenseSmallCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);
        let poly = make_dense_poly(NV, 0x715e_d001);
        let pt = random_point(NV, 0x715e_d002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_small");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered dense prove");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_small");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tiered dense verify");
    });
}

#[test]
fn tiered_onehot_prove_verify_small() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        let test_start = Instant::now();
        // With book-correct per-chunk D rows, the onehot f=2 routed
        // schedule first becomes viable at NV=25.
        // With the book §5.4 10-group layout the tiered chunk fold row
        // budgets for all k chunks in one grouped z_pre, which makes
        // NV=28 the smallest f=2 onehot schedule supported by the planner.
        const NV: usize = 28;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, TieredOneHotSmallCfg>;

        tracing::debug!("[onehot_e2e] start NV={NV} D={D}");
        let t = Instant::now();
        let layout = TieredOneHotSmallCfg::commitment_layout(NV).expect("layout");
        tracing::debug!("[onehot_e2e] layout ok after {:?}", t.elapsed());
        assert!(layout.use_setup_claim_reduction);
        let t = Instant::now();
        let poly = make_onehot_poly(&layout, 0x715e_0001);
        tracing::debug!("[onehot_e2e] poly built after {:?}", t.elapsed());
        let t = Instant::now();
        let pt = random_point(NV, 0x715e_0002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        tracing::debug!("[onehot_e2e] opening computed after {:?}", t.elapsed());
        let t = Instant::now();
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        tracing::debug!("[onehot_e2e] prover setup built after {:?}", t.elapsed());
        let t = Instant::now();
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        tracing::debug!("[onehot_e2e] verifier setup built after {:?}", t.elapsed());
        let t = Instant::now();
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        tracing::debug!("[onehot_e2e] root commit done after {:?}", t.elapsed());
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let t = Instant::now();
        tracing::debug!(
            "[onehot_e2e] batched_prove start at {:?}",
            test_start.elapsed()
        );
        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/onehot_small");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered onehot prove");
        tracing::debug!("[onehot_e2e] batched_prove done after {:?}", t.elapsed());

        let t = Instant::now();
        tracing::debug!(
            "[onehot_e2e] batched_verify start at {:?}",
            test_start.elapsed()
        );
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/onehot_small");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tiered onehot verify");
        tracing::debug!(
            "[onehot_e2e] batched_verify done after {:?}; total {:?}",
            t.elapsed(),
            test_start.elapsed()
        );
    });
}

#[test]
fn tiered_dense_prove_verify_mid_f4() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        // First NV at which the planner accepts f=4 dense
        // (probe_min_viable_nv_for_tier_f4: dense_f4_ok flips true at
        // NV=19).
        const NV: usize = 19;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseMidCfg>;

        let layout = TieredDenseMidCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);
        let poly = make_dense_poly(NV, 0x715e_4001);
        let pt = random_point(NV, 0x715e_4002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_f4");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered dense f=4 prove");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_f4");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tiered dense f=4 verify");
    });
}

#[test]
#[ignore = "diagnostic: scan dense f=4 across NVs to find scaling limit"]
fn probe_dense_f4_scaling() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseMidCfg>;
        for nv in [19usize, 22, 25, 28, 32] {
            let t0 = Instant::now();
            let layout = match TieredDenseMidCfg::commitment_layout(nv) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("NV={nv}: layout err={e:?}");
                    continue;
                }
            };
            if !layout.use_setup_claim_reduction {
                eprintln!("NV={nv}: no claim reduction");
                continue;
            }
            let poly = make_dense_poly(nv, 0x715e_4201 + nv as u64);
            let pt = random_point(nv, 0x715e_4202 + nv as u64);
            let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
            let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
            let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
            let (commitment, hint) = match <Scheme as CommitmentProver<F, D>>::commit(
                std::slice::from_ref(&poly),
                &setup,
            ) {
                Ok(out) => out,
                Err(e) => {
                    eprintln!("NV={nv}: commit err={e:?}");
                    continue;
                }
            };
            let poly_refs = [&poly];
            let commitments = [commitment];
            let openings = [opening];
            let mut prover_transcript = Blake2bTranscript::<F>::new(b"probe_dense_f4");
            let prove_t0 = Instant::now();
            let proof_res = <Scheme as CommitmentProver<F, D>>::batched_prove(
                &setup,
                prove_input(&pt, &poly_refs, &commitments[0], hint),
                &mut prover_transcript,
                BasisMode::Lagrange,
            );
            let prove_elapsed = prove_t0.elapsed();
            let proof = match proof_res {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("NV={nv}: prove FAILED after {prove_elapsed:?}: {e:?}");
                    continue;
                }
            };
            let mut verifier_transcript = Blake2bTranscript::<F>::new(b"probe_dense_f4");
            let verify_t0 = Instant::now();
            let verify_res = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt, &openings, &commitments[0]),
                BasisMode::Lagrange,
            );
            let verify_elapsed = verify_t0.elapsed();
            let total = t0.elapsed();
            match verify_res {
                Ok(_) => eprintln!(
                    "NV={nv}: OK prove={prove_elapsed:?} verify={verify_elapsed:?} total={total:?}"
                ),
                Err(e) => eprintln!(
                    "NV={nv}: VERIFY FAILED prove={prove_elapsed:?} verify={verify_elapsed:?} total={total:?} err={e:?}"
                ),
            }
        }
    });
}

#[test]
fn tiered_onehot_prove_verify_mid_f4() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        // First NV at which the planner accepts f=4 onehot
        // (probe_min_viable_nv_for_tier_f4: onehot_f4_ok flips true at
        // NV=25).
        const NV: usize = 25;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, TieredOneHotMidCfg>;

        let layout = TieredOneHotMidCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);
        let poly = make_onehot_poly(&layout, 0x715e_4101);
        let pt = random_point(NV, 0x715e_4102);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/onehot_f4");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered onehot f=4 prove");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/onehot_f4");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tiered onehot f=4 verify");
    });
}

/// Book §5.8 line 1170 per-level cascade probe: scan small NV values
/// to find the lowest NV at which both `DenseCascadeSmallCfg`
/// (`f_{L0}=2, f_{L1}=2`) and a schedule with a fold at L1 are
/// jointly viable. Reports both schedulability and the actual
/// per-level tier the planner picks so we can confirm `f_{L0}`,
/// `f_{L1}` flow through.
#[test]
#[ignore = "diagnostic probe; run explicitly with --ignored"]
fn probe_cascade_schedules() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        for nv in [19usize, 22, 25, 28, 32] {
            let dense_small = DenseCascadeSmallCfg::commitment_layout(nv);
            let onehot_small = OneHotCascadeSmallCfg::commitment_layout(nv);
            let dense_headline = DenseCascadeCfg::commitment_layout(nv);
            let onehot_headline = OneHotCascadeCfg::commitment_layout(nv);
            eprintln!(
                "NV={nv}: dense_cascade(f=2,2)={} dense_cascade(f=8,4)={} onehot_cascade(f=2,2)={} onehot_cascade(f=8,4)={}",
                dense_small.is_ok_and(|l| l.use_setup_claim_reduction),
                dense_headline.is_ok_and(|l| l.use_setup_claim_reduction),
                onehot_small.is_ok_and(|l| l.use_setup_claim_reduction),
                onehot_headline.is_ok_and(|l| l.use_setup_claim_reduction),
            );
            // Per-level tier discovery: report how many routing folds
            // the planner emits per cascade config, so we know which
            // (NV, cascade) combinations exercise multi-level cascade.
            use akita_planner::PlannerConfig;
            for (label, l0, l1) in [
                ("dense_small(2,2)", 2usize, 2usize),
                ("dense_headline(8,4)", 8, 4),
            ] {
                let cfg_l0 = DenseCascadeSmallCfg::planner_setup_shrink_factor_at_level(0);
                let _ = (cfg_l0, l0, l1, label);
            }
            for (label, sched) in [
                (
                    "dense(2,2)",
                    DenseCascadeSmallCfg::get_params_for_prove(
                        nv,
                        nv,
                        1,
                        akita_types::AkitaRootBatchSummary::singleton(),
                    ),
                ),
                (
                    "dense(8,4)",
                    DenseCascadeCfg::get_params_for_prove(
                        nv,
                        nv,
                        1,
                        akita_types::AkitaRootBatchSummary::singleton(),
                    ),
                ),
            ] {
                match sched {
                    Ok(s) => {
                        let (count, tiers) = inspect_cascade_schedule(&s);
                        eprintln!("        {label}: routing_folds={count} tiers={tiers:?}");
                    }
                    Err(e) => eprintln!("        {label}: schedule err={e:?}"),
                }
            }
        }
    });
}

/// Inspect a cascade schedule and report the per-level routing tier.
/// Returns `(routing_fold_count, tiers_per_routing_fold)` where the
/// second is the list of `shrink_factor` values across routing folds
/// in order (L0, L1, …).
fn inspect_cascade_schedule(schedule: &akita_types::Schedule) -> (usize, Vec<usize>) {
    use akita_types::Step;
    let mut tiers = Vec::new();
    for step in &schedule.steps {
        if let Step::Fold(fold) = step {
            if fold.s_field_len_emitted > 0 {
                tiers.push(fold.tier_setup_params.shrink_factor);
            }
        }
    }
    (tiers.len(), tiers)
}

/// Book §5.8 cascade end-to-end at small NV. Exercises the per-level
/// tier plumbing (`planner_setup_shrink_factor_at_level`) with
/// `f_{L0} = 2`, `f_{L1} = 2` — the lowest non-trivial cascade that
/// activates per-level tier selection. Validates that:
///
/// - The planner produces a schedule where `L0` and `L1` carry their
///   own tier params.
/// - The prover/verifier handle the per-level tier shape correctly
///   end-to-end without uniform-tier assumptions.
///
/// This is the proof point that the per-level cascade infrastructure
/// works; production-scale `f_{L0}=8, f_{L1}=4` reuses the same plumbing
/// and is gated only by the existing scale ceilings (audit B-5).
#[test]
fn tiered_dense_cascade_l0_l1_small() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 19;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, DenseCascadeSmallCfg>;

        let layout = DenseCascadeSmallCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);

        // Confirm the schedule actually applies cascade per-level: a
        // pre-flight check on `planner_setup_shrink_factor_at_level`
        // ensures the wrapper carries our per-level tier policy
        // through to the planner before we run the slow E2E.
        use akita_planner::PlannerConfig;
        assert_eq!(
            DenseCascadeSmallCfg::planner_setup_shrink_factor_at_level(0),
            2,
            "L0 tier must be 2 (cascade pre-flight)"
        );
        assert_eq!(
            DenseCascadeSmallCfg::planner_setup_shrink_factor_at_level(1),
            2,
            "L1 tier must be 2 (cascade pre-flight)"
        );
        assert_eq!(
            DenseCascadeSmallCfg::planner_setup_shrink_factor_at_level(2),
            1,
            "L2+ must fall back to un-tiered (cascade pre-flight)"
        );

        // Deeper assertion: inspect the actual planner output to
        // confirm the per-level tier policy flows through. Each
        // `Step::Fold` step that emits an `S` payload must carry
        // `tier_setup_params` matching the per-level policy:
        // - The first such step has `shrink_factor = 2` (L0 routing
        //   to L1, both with `f = 2` per cascade).
        // - The second has `shrink_factor` per `_at_level(1)` = 2.
        // - Any later step falls back to un-tiered (`f = 1`) per
        //   `_at_level(k≥2)` = 1.
        let schedule = DenseCascadeSmallCfg::get_params_for_prove(
            NV,
            NV,
            1,
            akita_types::AkitaRootBatchSummary::singleton(),
        )
        .expect("cascade schedule");
        let (routing_count, tiers) = inspect_cascade_schedule(&schedule);
        eprintln!(
            "[cascade_l0_l1_small] NV={NV}, routing folds={routing_count}, \
             per-level tiers={:?}",
            tiers
        );
        for (i, &t) in tiers.iter().enumerate() {
            let expected = DenseCascadeSmallCfg::planner_setup_shrink_factor_at_level(i);
            assert_eq!(
                t, expected,
                "routing fold {i}: schedule tier {t} must match \
                 planner_setup_shrink_factor_at_level({i}) = {expected}"
            );
        }
        assert!(
            routing_count >= 1,
            "cascade test must exercise at least one routing fold step (L0); \
             planner emitted no S-routed fold"
        );

        let poly = make_dense_poly(NV, 0x715e_ca01);
        let pt = random_point(NV, 0x715e_ca02);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_cascade_l0_l1");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("dense cascade L0+L1 prove");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_cascade_l0_l1");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("dense cascade L0+L1 verify");
    });
}

/// Book §5.8 line 1170 headline cascade end-to-end at the smallest
/// schedulable NV. Uses `f_{L0} = 8`, `f_{L1} = 4`, `f_{Lk} = 1` for
/// `k ≥ 2`. This is exactly the configuration the book Table at
/// 1141–1158 measures as "T1+T2 @ L0+L1" (16× / 35× / 265× speedup at
/// NV=32 / 38 / 44).
///
/// Currently ignored: at NV=19 the planner only emits one routing
/// fold (L0 with `f = 8`), and the prover hits the same scale ceiling
/// as `tiered_production_prove_verify` (audit B-5). Will be un-ignored
/// once B-5 is profiled and the OOM / CRT-overflow path at production
/// `f = 8` is unblocked. The per-level cascade plumbing itself is
/// validated by [`tiered_dense_cascade_l0_l1_small`].
#[test]
#[ignore = "headline cascade hits scale ceiling — gated on audit B-5"]
fn tiered_dense_cascade_l0_l1_headline_small() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 19;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, DenseCascadeCfg>;

        let layout = DenseCascadeCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);

        use akita_planner::PlannerConfig;
        assert_eq!(
            DenseCascadeCfg::planner_setup_shrink_factor_at_level(0),
            8,
            "headline cascade L0 tier must be 8 (book §5.8 line 1170)"
        );
        assert_eq!(
            DenseCascadeCfg::planner_setup_shrink_factor_at_level(1),
            4,
            "headline cascade L1 tier must be 4 (book §5.8 line 1170)"
        );

        let schedule = DenseCascadeCfg::get_params_for_prove(
            NV,
            NV,
            1,
            akita_types::AkitaRootBatchSummary::singleton(),
        )
        .expect("headline cascade schedule");
        let (routing_count, tiers) = inspect_cascade_schedule(&schedule);
        eprintln!(
            "[cascade_l0_l1_headline_small] NV={NV}, routing folds={routing_count}, \
             per-level tiers={:?}",
            tiers
        );
        for (i, &t) in tiers.iter().enumerate() {
            let expected = DenseCascadeCfg::planner_setup_shrink_factor_at_level(i);
            assert_eq!(
                t, expected,
                "routing fold {i}: schedule tier {t} must match \
                 planner_setup_shrink_factor_at_level({i}) = {expected}"
            );
        }

        let poly = make_dense_poly(NV, 0x715e_ca81);
        let pt = random_point(NV, 0x715e_ca82);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_cascade_headline");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("dense cascade headline prove");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_cascade_headline");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("dense cascade headline verify");
    });
}

#[test]
fn tiered_production_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 32;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseProductionCfg>;

        let layout = TieredDenseProductionCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);
        let poly = make_dense_poly(NV, 0x715e_f801);
        let pt = random_point(NV, 0x715e_f802);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/production");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered production prove");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/production");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tiered production verify");
    });
}

#[test]
fn tiered_rejects_tampered_s_opening_value() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 20;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseSmallCfg>;

        let layout = TieredDenseSmallCfg::commitment_layout(NV).expect("layout");
        let poly = make_dense_poly(NV, 0x715e_5001);
        let pt = random_point(NV, 0x715e_5002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/tamper_s_opening");
        let mut proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered tamper prove");

        let fold_root = proof
            .root
            .as_fold_mut()
            .expect("tiered test must exercise root fold");
        let payload = fold_root
            .stage2
            .setup_claim_reduction
            .as_mut()
            .expect("tiered proof should carry setup claim reduction");
        payload.s_opening_value += F::from_u64(1);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/tamper_s_opening");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(result.is_err(), "tampered s_opening_value must reject");
    });
}

#[test]
#[ignore = "diagnostic probe; run explicitly to find the smallest viable NV"]
fn probe_min_viable_nv_for_tier_f2() {
    // Iterate small NV values to find the minimum the tiered f=2
    // planner accepts. Useful when debugging tiered failures with a
    // fast prove/verify cycle.
    for nv in 6..=36 {
        let dense = TieredDenseSmallCfg::commitment_layout(nv);
        let onehot = TieredOneHotSmallCfg::commitment_layout(nv);
        tracing::debug!(
            "NV={nv}: dense_layout_ok={} onehot_layout_ok={} dense_cr={} onehot_cr={}",
            dense.is_ok(),
            onehot.is_ok(),
            dense.as_ref().is_ok_and(|l| l.use_setup_claim_reduction),
            onehot.as_ref().is_ok_and(|l| l.use_setup_claim_reduction),
        );
    }
}

#[test]
#[ignore = "diagnostic probe; run explicitly to find the smallest viable NV"]
fn probe_min_viable_nv_for_tier_f4() {
    // Same scan as probe_min_viable_nv_for_tier_f2 but for f = 4. The
    // planner needs both axes (m_S, r_S) >= log2(4) = 2 to admit f = 4,
    // which raises the NV floor vs f = 2.
    tracing::debug!("NV  dense_f4_ok  onehot_f4_ok  dense_cr  onehot_cr");
    for nv in 8..=36 {
        let dense = TieredDenseMidCfg::commitment_layout(nv);
        let onehot = TieredOneHotMidCfg::commitment_layout(nv);
        tracing::debug!(
            "{nv}  {}  {}  {}  {}",
            dense.is_ok(),
            onehot.is_ok(),
            dense.as_ref().is_ok_and(|l| l.use_setup_claim_reduction),
            onehot.as_ref().is_ok_and(|l| l.use_setup_claim_reduction),
        );
    }
}
