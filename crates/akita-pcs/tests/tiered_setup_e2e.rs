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
use std::time::{Duration, Instant};

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

/// Book §5.8 cascade discovery probe: extended NV scan to find the
/// smallest NV at which the planner emits ≥ 2 routing folds for each
/// of the four cascade configs of interest. Used to align headline
/// (`f_{L0}=8, f_{L1}=4`) measurement points with book Table 1141–1158
/// at NV ∈ {32, 38, 44}.
///
/// Each row logs `routing_count + per-level tiers` if the schedule
/// resolves, `layout_err` if `commitment_layout` itself rejects, or
/// `sched_err` if the planner rejects. Single failures never abort
/// the scan; the test is purely diagnostic.
#[test]
#[ignore = "diagnostic probe; run explicitly with --ignored"]
fn probe_cascade_schedules_extended() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        let nvs = [19usize, 22, 25, 28, 32, 35, 38, 41, 44, 47, 50];
        for nv in nvs {
            probe_one::<DenseCascadeSmallCfg>(nv, "dense_small(2,2)  ");
            probe_one::<OneHotCascadeSmallCfg>(nv, "onehot_small(2,2) ");
            probe_one::<DenseCascadeCfg>(nv, "dense_headline(8,4)");
            probe_one::<OneHotCascadeCfg>(nv, "onehot_headline(8,4)");
        }

        fn probe_one<Cfg: CommitmentConfig + akita_planner::PlannerConfig>(
            nv: usize,
            cfg_label: &str,
        ) {
            match Cfg::commitment_layout(nv) {
                Ok(_layout) => {}
                Err(e) => {
                    eprintln!("NV={nv:<3} cfg={cfg_label}  layout=ERR err={e:?}");
                    return;
                }
            }
            match Cfg::get_params_for_prove(
                nv,
                nv,
                1,
                akita_types::AkitaRootBatchSummary::singleton(),
            ) {
                Ok(schedule) => {
                    let (count, tiers) = inspect_cascade_schedule(&schedule);
                    eprintln!(
                        "NV={nv:<3} cfg={cfg_label}  sched=OK routing={count} tiers={tiers:?}"
                    );
                }
                Err(e) => {
                    eprintln!("NV={nv:<3} cfg={cfg_label}  sched=ERR err={e:?}");
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
             per-level tiers={tiers:?}"
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

/// Book §5.8 line 1170 headline cascade SCHEDULE assertion at the
/// smallest variable count where the planner emits both routing folds.
///
/// Uses `f_{L0} = 8`, `f_{L1} = 4`, `f_{Lk} = 1` for `k ≥ 2` — exactly
/// the configuration the book Table at 1141–1158 measures as
/// "T1+T2 @ L0+L1" (16× / 35× / 265× speedup at NV=32 / 38 / 44).
///
/// The matched-tier planner sizing rejects NV ≤ 21 because the L1
/// receive shape needs `r_S, m_S ≥ log₂(f_{L0}) = 3` on both axes, so
/// the smallest viable NV is `22`. E2E prove + verify at this scale
/// still hits the same ceiling as
/// [`tiered_dense_cascade_l0_l1_headline_small`] (audit B-5), so this
/// sentinel asserts schedule-side correctness only — the planner is no
/// longer dead-code at L1. The per-level cascade plumbing under E2E
/// remains validated by [`tiered_dense_cascade_l0_l1_small`].
#[test]
fn tiered_dense_cascade_l0_l1_fires() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 22;

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
        assert_eq!(
            DenseCascadeCfg::planner_setup_shrink_factor_at_level(2),
            1,
            "headline cascade L2+ must fall back to un-tiered (book §5.8)"
        );

        let schedule = DenseCascadeCfg::get_params_for_prove(
            NV,
            NV,
            1,
            akita_types::AkitaRootBatchSummary::singleton(),
        )
        .expect("dense headline cascade schedule");
        let (routing_count, tiers) = inspect_cascade_schedule(&schedule);
        eprintln!(
            "[dense_cascade_l0_l1_fires] NV={NV}, routing folds={routing_count}, \
             per-level tiers={tiers:?}"
        );
        assert!(
            routing_count >= 2,
            "headline cascade must emit at least 2 routing folds (L0 + L1); \
             got routing_count={routing_count} tiers={tiers:?}"
        );
        assert_eq!(
            tiers,
            vec![8, 4],
            "headline cascade must use book §5.8 line 1170 per-level tiers \
             [F_L0=8, F_L1=4]; got {tiers:?}"
        );
    });
}

/// Book §5.8 line 1170 headline cascade end-to-end at the smallest
/// schedulable NV. Uses `f_{L0} = 8`, `f_{L1} = 4`, `f_{Lk} = 1` for
/// `k ≥ 2`. This is exactly the configuration the book Table at
/// 1141–1158 measures as "T1+T2 @ L0+L1" (16× / 35× / 265× speedup at
/// NV=32 / 38 / 44).
///
/// The per-level cascade plumbing itself is validated by
/// [`tiered_dense_cascade_l0_l1_small`]; the schedule firing for the
/// headline `(8, 4)` is asserted by [`tiered_dense_cascade_l0_l1_fires`].
#[test]
fn tiered_dense_cascade_l0_l1_headline_small() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 22;
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
             per-level tiers={tiers:?}"
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

/// Book §5.8 line 1170 headline cascade end-to-end at the smallest
/// schedulable NV for onehot D=64. The matched-tier planner first
/// emits an `(f_L0=8, f_L1=4)` schedule at NV=28 for onehot per
/// `probe_cascade_schedules_extended`; below that, no L1 routing
/// fires.
///
/// Per the same per-level cascade plumbing as
/// [`tiered_dense_cascade_l0_l1_headline_small`], this end-to-end
/// confirms the prover/verifier flow also closes on a lower-D
/// onehot config without the dense-d128 verifier setup-derivation
/// ceiling. The smaller per-NV memory of OneHot D=64 makes NV=28
/// fit on a 123 GiB host.
#[test]
fn tiered_onehot_cascade_l0_l1_headline_small() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 28;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, OneHotCascadeCfg>;

        let layout = OneHotCascadeCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);

        use akita_planner::PlannerConfig;
        assert_eq!(
            OneHotCascadeCfg::planner_setup_shrink_factor_at_level(0),
            8,
            "onehot headline cascade L0 tier must be 8 (book §5.8 line 1170)"
        );
        assert_eq!(
            OneHotCascadeCfg::planner_setup_shrink_factor_at_level(1),
            4,
            "onehot headline cascade L1 tier must be 4 (book §5.8 line 1170)"
        );

        let schedule = OneHotCascadeCfg::get_params_for_prove(
            NV,
            NV,
            1,
            akita_types::AkitaRootBatchSummary::singleton(),
        )
        .expect("onehot headline cascade schedule");
        let (routing_count, tiers) = inspect_cascade_schedule(&schedule);
        eprintln!(
            "[onehot_cascade_l0_l1_headline_small] NV={NV}, routing folds={routing_count}, \
             per-level tiers={tiers:?}"
        );
        for (i, &t) in tiers.iter().enumerate() {
            let expected = OneHotCascadeCfg::planner_setup_shrink_factor_at_level(i);
            assert_eq!(
                t, expected,
                "routing fold {i}: schedule tier {t} must match \
                 planner_setup_shrink_factor_at_level({i}) = {expected}"
            );
        }

        let poly = make_onehot_poly(&layout, 0x715e_ca91);
        let pt = random_point(NV, 0x715e_ca92);
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
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/onehot_cascade_headline");
        let prove_t = Instant::now();
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("onehot cascade headline prove");
        let prove_elapsed = prove_t.elapsed();

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/onehot_cascade_headline");
        let verify_t = Instant::now();
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("onehot cascade headline verify");
        let verify_elapsed = verify_t.elapsed();
        eprintln!(
            "[onehot_cascade_l0_l1_headline_small] prove={prove_elapsed:?} verify={verify_elapsed:?}"
        );
    });
}

/// Book §5.8 cascade verifier wall-clock measurement vs baseline at
/// a single NV.
///
/// Compares verify wall-clock (prove wall-clock printed for context)
/// across four claim-reduction shapes at the SAME NV:
///   - baseline: bare [`DenseCfg`] (no claim reduction).
///   - claim-reduction untiered: [`ClaimReductionCfg<DenseCfg, 1>`]
///     per book §5.3 (the pre-Slice-G shape, midway between baseline
///     and tiered T2).
///   - T2 @ L0 only: [`TieredClaimReductionCfg<DenseCfg>`] (= `f = 8`
///     at L0, uniform; book §5.4 single-tier sweet spot).
///   - T1+T2 @ L0+L1: [`DenseCascadeCfg`] (the headline `f_L0 = 8`,
///     `f_L1 = 4` cascade from book §5.8 line 1170).
///
/// Configs whose [`CommitmentConfig::commitment_layout`] rejects the
/// chosen NV are skipped silently. The chosen NV must accept at least
/// `baseline + T2-only + cascade` for the speedup ratio to be defined.
///
/// Each working config runs `iterations = 3` (1 warmup + 2 measured)
/// with a fresh transcript and fresh setup per iteration; verify and
/// prove medians are reported plus the verify-time ratio against
/// baseline.
///
/// This is a measurement, not a correctness assertion: absolute
/// speedup numbers vary run-to-run with thread scheduling and won't
/// match book Table 1141–1158 at NV=22 (the book measures at NV ≥ 32
/// with `D = 128` op count, a different shape). The test passes if
/// every selected config successfully proves and verifies at the
/// target NV; the qualitative trend `cascade ≤ T2-only ≤ baseline`
/// (lower verify time better) is printed but not asserted.
///
/// Crossover note (measured at NV ∈ {22, 23, 25} on a 123 GiB host):
/// the cascade plumbing is end-to-end functional at every dense NV
/// that fits, but the qualitative trend the book predicts is
/// VIOLATED for *wall-clock* in this regime. Median verify
/// wall-clock:
///
/// ```text
///   NV  baseline   untiered(f=1)   T2(f=8)   cascade(f=8,4)
///   22    0.011s        0.064s      6.76s        15.67s
///   23    0.026s        0.087s      7.02s        17.78s
///   25    0.037s        0.179s     13.99s        31.54s
/// ```
///
/// Baseline verify grows ~2×/NV (O(N)); cascade verify grows
/// ~1.3–1.4×/NV (per-level transcript / per-chunk MLE / meta-commit
/// overhead dominates). Extrapolating, the wall-clock crossover NV
/// where cascade verify < baseline verify is ≈ 32 — exactly the
/// lowest measurement point in book Table 1141–1158. Book numbers
/// are `D = 128` verifier *op counts* (~828K ops at NV=32 baseline),
/// which do not include the per-level constant overhead our
/// wall-clock measures; that gap is the structural reason for the
/// small-NV inversion. NV ≥ 32 dense (~64 GiB just for the
/// polynomial, much more for tiered setup matrices) does not fit on
/// a 123 GiB host, so the wall-clock crossover cannot be observed
/// here — the test still proves the cascade machinery is correct
/// end-to-end at every feasible NV.
///
/// Override the NV list via `AKITA_SPEEDUP_NVS=22,23,25` (default
/// `22`). Marked `#[ignore]` because the run is slow (~5 min at
/// NV=22, ~8 min at NV=23, ~12 min at NV=25; NV ≥ 28 takes hours
/// and is unlikely to flip the qualitative trend on this host).
#[test]
#[ignore = "measurement-only; run with --ignored"]
fn tiered_dense_cascade_speedup_measurement() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const ITERS: usize = 3;
        let nvs = parse_nvs_env("AKITA_SPEEDUP_NVS", &[22]);

        for nv in nvs {
            eprintln!();
            eprintln!("=== cascade speedup measurement: NV={nv}, D={DENSE_D} ===");

            type UntieredDenseClaimCfg = ClaimReductionCfg<DenseCfg, 1>;
            type T2OnlyDenseCfg = ClaimReductionCfg<DenseCfg, 8>;

            let baseline = measure_dense_speedup_config::<DenseCfg>(
                nv,
                "baseline (DenseCfg)",
                ITERS,
                0x715e_5b01,
                0x715e_5b02,
                b"tiered_setup_e2e/speedup/baseline",
            );
            let untiered = measure_dense_speedup_config::<UntieredDenseClaimCfg>(
                nv,
                "claim-reduction untiered (f=1)",
                ITERS,
                0x715e_5b03,
                0x715e_5b04,
                b"tiered_setup_e2e/speedup/untiered",
            );
            let t2_only = measure_dense_speedup_config::<T2OnlyDenseCfg>(
                nv,
                "T2 @ L0 (f=8)",
                ITERS,
                0x715e_5b05,
                0x715e_5b06,
                b"tiered_setup_e2e/speedup/t2_only",
            );
            let cascade = measure_dense_speedup_config::<DenseCascadeCfg>(
                nv,
                "T1+T2 @ L0+L1 (f=8,4)",
                ITERS,
                0x715e_5b07,
                0x715e_5b08,
                b"tiered_setup_e2e/speedup/cascade",
            );

            let baseline_amortized_s = baseline
                .as_ref()
                .and_then(|m| median_measured(&m.verify_times))
                .map(|d| d.as_secs_f64());

            eprintln!(
                "    {:<34}  {:>9}  {:>10}  {:>11}  {:>12}  routing  tiers",
                "config", "prove(s)", "cold_v(s)", "amort_v(s)", "vs_baseline",
            );
            for m in [&baseline, &untiered, &t2_only, &cascade]
                .into_iter()
                .flatten()
            {
                let prove_med = median_measured(&m.prove_times).unwrap_or(Duration::ZERO);
                let verify_cold = m.verify_times.first().copied().unwrap_or(Duration::ZERO);
                let verify_amortized = median_measured(&m.verify_times).unwrap_or(Duration::ZERO);
                let verify_s = verify_amortized.as_secs_f64();
                let ratio = match (baseline_amortized_s, verify_s) {
                    (Some(bv), v) if v > 0.0 => format!("{:>11.2}×", bv / v),
                    _ => format!("{:>12}", "n/a"),
                };
                eprintln!(
                    "    {:<34}  {:>9.3}  {:>10.3}  {:>11.3}  {}  {:>4}     {:?}",
                    m.label,
                    prove_med.as_secs_f64(),
                    verify_cold.as_secs_f64(),
                    verify_amortized.as_secs_f64(),
                    ratio,
                    m.routing_count,
                    m.tiers,
                );
            }
            eprintln!(
                "    (cold_v = iter 1, populates ntt_shared_cache + tiered_s_cache; \
                 amort_v = median of iters 2..N, book-aligned per Fig. 12 line 817)"
            );

            eprintln!();
            eprintln!(
                "Book Table 1141–1158 (at NV=32, D=128 verifier op count; \
                 different scale, qualitative only):"
            );
            eprintln!(
                "    {:<34}  {:>9}  {:>9}  {:>12}",
                "baseline", "—", "—", "       1.00×",
            );
            eprintln!(
                "    {:<34}  {:>9}  {:>9}  {:>12}",
                "T2 @ L0 (f=8)", "—", "—", "       3.80×",
            );
            eprintln!(
                "    {:<34}  {:>9}  {:>9}  {:>12}",
                "T1+T2 @ L0+L1 (f=8,4)", "—", "—", "      16.00×",
            );

            // Qualitative trend hint (not asserted; reported only).
            // Compared against amortized (cache-hit) verify time, which is
            // the book-aligned metric per Figure 12 line 817 ("preprocessed
            // shared-matrix commitment C_S").
            if let (Some(bv), Some(tv), Some(cv)) = (
                baseline_amortized_s,
                t2_only
                    .as_ref()
                    .and_then(|m| median_measured(&m.verify_times))
                    .map(|d| d.as_secs_f64()),
                cascade
                    .as_ref()
                    .and_then(|m| median_measured(&m.verify_times))
                    .map(|d| d.as_secs_f64()),
            ) {
                let monotonic = cv < tv && tv < bv;
                eprintln!();
                eprintln!(
                    "Qualitative trend cascade < T2-only < baseline (verify): {} \
                     (cascade={cv:.3}s, t2_only={tv:.3}s, baseline={bv:.3}s)",
                    if monotonic { "HOLDS" } else { "VIOLATED" },
                );
            }
        }
    });
}

/// Per-config measurement record for
/// [`tiered_dense_cascade_speedup_measurement`].
///
/// `verify_times[0]` is the cold-cache iteration that populates
/// `AkitaVerifierSetup::ntt_shared_cache` and `tiered_s_cache`;
/// `verify_times[1..]` are amortized cache-hit verifies that map to
/// book Table 1141-1158's "verifier op count" framing (book Figure 12
/// line 817 / §5.3 line 952: `C_S` is preprocessed at setup time).
#[derive(Clone, Debug)]
struct SpeedupMeasurement {
    label: String,
    routing_count: usize,
    tiers: Vec<usize>,
    prove_times: Vec<Duration>,
    verify_times: Vec<Duration>,
}

/// Run a fresh `iterations`-pass prove/verify of a single dense-D
/// `Cfg` and record per-iteration wall-clock.
///
/// Returns `None` when `Cfg::commitment_layout(nv)` rejects (the
/// caller skips the config silently); panics on any subsequent
/// `commit`/`prove`/`verify` failure since those would indicate a
/// real regression, not a planner skip.
fn measure_dense_speedup_config<Cfg>(
    nv: usize,
    label: &str,
    iterations: usize,
    poly_seed: u64,
    point_seed: u64,
    transcript_label: &[u8],
) -> Option<SpeedupMeasurement>
where
    Cfg: CommitmentConfig<Field = F> + akita_planner::PlannerConfig,
{
    let layout = match Cfg::commitment_layout(nv) {
        Ok(l) => l,
        Err(e) => {
            tracing::info!(
                target: "speedup",
                "skip {label} at NV={nv}: layout err={e:?}"
            );
            return None;
        }
    };

    let schedule =
        Cfg::get_params_for_prove(nv, nv, 1, akita_types::AkitaRootBatchSummary::singleton())
            .expect("schedule must resolve when layout succeeds");
    let (routing_count, tiers) = inspect_cascade_schedule(&schedule);
    tracing::info!(
        target: "speedup",
        nv,
        label = %label,
        routing_count,
        ?tiers,
        "speedup config schedule",
    );

    // Book Figure 12 line 817 / §5.3 line 952: the preprocessed
    // shared-matrix commitment `C_S` is derived once during setup and
    // reused across every verify call on that setup. We mirror that
    // model here by reusing one `AkitaVerifierSetup` (and the prover
    // setup that backs it) across all iterations: the first verify
    // pays the cold-cache cost of populating `ntt_shared_cache` and
    // `tiered_s_cache`; subsequent verifies hit the cache and pay only
    // the per-proof verifier op work the book's Table 1141-1158
    // measures. Both numbers are reported below.
    let poly = make_dense_poly(nv, poly_seed);
    let pt = random_point(nv, point_seed);
    let opening = opening_from_poly::<DENSE_D, _>(&poly, &pt, &layout);
    let setup = <AkitaCommitmentScheme<DENSE_D, Cfg> as CommitmentProver<F, DENSE_D>>::setup_prover(
        nv, 1, 1,
    );
    let verifier_setup =
        <AkitaCommitmentScheme<DENSE_D, Cfg> as CommitmentProver<F, DENSE_D>>::setup_verifier(
            &setup,
        );

    let mut prove_times = Vec::with_capacity(iterations);
    let mut verify_times = Vec::with_capacity(iterations);

    for iter in 0..iterations {
        let (commitment, hint) = <AkitaCommitmentScheme<DENSE_D, Cfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::commit(std::slice::from_ref(&poly), &setup)
        .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(transcript_label);
        let t0 = Instant::now();
        let proof =
            <AkitaCommitmentScheme<DENSE_D, Cfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
                &setup,
                prove_input(&pt, &poly_refs, &commitments[0], hint),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");
        let prove_elapsed = t0.elapsed();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(transcript_label);
        let t0 = Instant::now();
        <AkitaCommitmentScheme<DENSE_D, Cfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("verify");
        let verify_elapsed = t0.elapsed();

        let warmup = iter == 0;
        tracing::info!(
            target: "speedup",
            nv,
            label = %label,
            iter,
            warmup,
            prove_s = prove_elapsed.as_secs_f64(),
            verify_s = verify_elapsed.as_secs_f64(),
            "speedup iter",
        );

        prove_times.push(prove_elapsed);
        verify_times.push(verify_elapsed);
    }

    Some(SpeedupMeasurement {
        label: label.to_string(),
        routing_count,
        tiers,
        prove_times,
        verify_times,
    })
}

/// Book §5.8 cascade verifier wall-clock measurement vs baseline at
/// a single NV — onehot D=64 counterpart of
/// [`tiered_dense_cascade_speedup_measurement`].
///
/// Compares verify wall-clock across four claim-reduction shapes at
/// the SAME NV, on the lower-D onehot polynomial family:
///   - baseline: bare [`OneHotCfg`] (no claim reduction).
///   - claim-reduction untiered: [`ClaimReductionCfg<OneHotCfg, 1>`]
///     per book §5.3 (pre-Slice-G shape).
///   - T2 @ L0 only: `ClaimReductionCfg<OneHotCfg, 4>` — `f = 4`
///     uniform single-tier (book §5.4 sweet spot for onehot per the
///     working `tiered_onehot_prove_verify_mid_f4`; `f = 8` is not
///     guaranteed to schedule at every NV the cascade picks).
///   - T1+T2 @ L0+L1: [`OneHotCascadeCfg`] (`f_{L0}=8, f_{L1}=4`).
///
/// The smaller per-NV memory of OneHot D=64 (vs Dense D=128) lifts
/// the dense ceiling that bounds the dense speedup test at NV=22; the
/// onehot variant pushes the measurement point closer to where the
/// book Table 1141–1158 trend should be observable.
///
/// Configs whose [`CommitmentConfig::commitment_layout`] rejects the
/// chosen NV are skipped silently. Each working config runs
/// `iterations = 3` (1 warmup + 2 measured) with a fresh transcript
/// per iteration and a single shared verifier setup (book Figure 12
/// line 817 model: `C_S` and the tiered_s_cache are populated once
/// per setup and reused across verifies).
///
/// Override the NV list via `AKITA_SPEEDUP_NVS_ONEHOT=28,30` (default
/// `28`, the smallest NV at which the onehot headline `(8, 4)`
/// cascade schedule fires per `probe_cascade_schedules_extended`).
/// Marked `#[ignore]` because the run is slow (~9 min measured at
/// NV=28; higher NV proportional). NV=30+ is gated by the 123 GiB
/// host memory ceiling on the cascade and T2 configs (audit B-5).
///
/// Crossover note (measured at NV=28 on a 123 GiB host):
/// the cascade plumbing is end-to-end functional and *all four*
/// onehot configs (baseline through T1+T2) close at NV=28, but —
/// matching the dense observation in
/// [`tiered_dense_cascade_speedup_measurement`] — the book's
/// qualitative trend is VIOLATED for *wall-clock* in this regime.
/// Median verify wall-clock at NV=28, D=64 (amortized, cache-hit):
///
/// ```text
///   config                          prove(s)  cold_v(s)  amort_v(s)  vs_baseline
///   baseline (OneHotCfg)            16.946     0.011      0.011       1.00×
///   claim-reduction untiered (f=1)  15.093     0.104      0.065       0.17×
///   T2 @ L0 (f=4)                   38.270     0.897      0.169       0.07×
///   T1+T2 @ L0+L1 (f=8,4)          100.504     3.886      0.539       0.02×
/// ```
///
/// Same shape as dense: baseline verify is bounded by a tiny constant
/// (~11 ms at NV=28 onehot), while each added tier carries per-level
/// transcript / per-chunk MLE / meta-commit overhead that dominates
/// at this NV. Onehot D=64 lets us push 6 NV beyond the dense
/// ceiling, but the absolute baseline grows much more slowly than the
/// cascade overhead does, so the wall-clock crossover is still
/// further out — extrapolation suggests > NV=32. Book Table 1141–1158
/// is measured in `D=128` verifier *op counts*, which excludes the
/// constant overheads our wall-clock includes; that gap remains the
/// structural reason for the small-NV inversion. NV ≥ 30 onehot does
/// not fit (cascade + T2 setup matrices exceed the 123 GiB ceiling),
/// so the wall-clock crossover cannot be observed here. The test
/// still proves the cascade machinery is correct end-to-end on a
/// lower-D config at every feasible NV.
#[test]
#[ignore = "measurement-only; run with --ignored"]
fn tiered_onehot_cascade_speedup_measurement() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const ITERS: usize = 3;
        let nvs = parse_nvs_env("AKITA_SPEEDUP_NVS_ONEHOT", &[28]);

        for nv in nvs {
            eprintln!();
            eprintln!("=== onehot cascade speedup measurement: NV={nv}, D={ONEHOT_D} ===");

            type UntieredOneHotClaimCfg = ClaimReductionCfg<OneHotCfg, 1>;
            type T2OnlyOneHotCfg = ClaimReductionCfg<OneHotCfg, 4>;

            let baseline = measure_onehot_speedup_config::<OneHotCfg>(
                nv,
                "baseline (OneHotCfg)",
                ITERS,
                0x715e_6b01,
                0x715e_6b02,
                b"tiered_setup_e2e/speedup_onehot/baseline",
            );
            let untiered = measure_onehot_speedup_config::<UntieredOneHotClaimCfg>(
                nv,
                "claim-reduction untiered (f=1)",
                ITERS,
                0x715e_6b03,
                0x715e_6b04,
                b"tiered_setup_e2e/speedup_onehot/untiered",
            );
            let t2_only = measure_onehot_speedup_config::<T2OnlyOneHotCfg>(
                nv,
                "T2 @ L0 (f=4)",
                ITERS,
                0x715e_6b05,
                0x715e_6b06,
                b"tiered_setup_e2e/speedup_onehot/t2_only",
            );
            let cascade = measure_onehot_speedup_config::<OneHotCascadeCfg>(
                nv,
                "T1+T2 @ L0+L1 (f=8,4)",
                ITERS,
                0x715e_6b07,
                0x715e_6b08,
                b"tiered_setup_e2e/speedup_onehot/cascade",
            );

            let baseline_amortized_s = baseline
                .as_ref()
                .and_then(|m| median_measured(&m.verify_times))
                .map(|d| d.as_secs_f64());

            eprintln!(
                "    {:<34}  {:>9}  {:>10}  {:>11}  {:>12}  routing  tiers",
                "config", "prove(s)", "cold_v(s)", "amort_v(s)", "vs_baseline",
            );
            for m in [&baseline, &untiered, &t2_only, &cascade]
                .into_iter()
                .flatten()
            {
                let prove_med = median_measured(&m.prove_times).unwrap_or(Duration::ZERO);
                let verify_cold = m.verify_times.first().copied().unwrap_or(Duration::ZERO);
                let verify_amortized = median_measured(&m.verify_times).unwrap_or(Duration::ZERO);
                let verify_s = verify_amortized.as_secs_f64();
                let ratio = match (baseline_amortized_s, verify_s) {
                    (Some(bv), v) if v > 0.0 => format!("{:>11.2}×", bv / v),
                    _ => format!("{:>12}", "n/a"),
                };
                eprintln!(
                    "    {:<34}  {:>9.3}  {:>10.3}  {:>11.3}  {}  {:>4}     {:?}",
                    m.label,
                    prove_med.as_secs_f64(),
                    verify_cold.as_secs_f64(),
                    verify_amortized.as_secs_f64(),
                    ratio,
                    m.routing_count,
                    m.tiers,
                );
            }
            eprintln!(
                "    (cold_v = iter 1, populates ntt_shared_cache + tiered_s_cache; \
                 amort_v = median of iters 2..N, book-aligned per Fig. 12 line 817)"
            );

            // Qualitative trend hint (not asserted; reported only).
            if let (Some(bv), Some(tv), Some(cv)) = (
                baseline_amortized_s,
                t2_only
                    .as_ref()
                    .and_then(|m| median_measured(&m.verify_times))
                    .map(|d| d.as_secs_f64()),
                cascade
                    .as_ref()
                    .and_then(|m| median_measured(&m.verify_times))
                    .map(|d| d.as_secs_f64()),
            ) {
                let monotonic = cv < tv && tv < bv;
                eprintln!();
                eprintln!(
                    "Qualitative trend cascade < T2-only < baseline (verify): {} \
                     (cascade={cv:.3}s, t2_only={tv:.3}s, baseline={bv:.3}s)",
                    if monotonic { "HOLDS" } else { "VIOLATED" },
                );
            }
        }
    });
}

/// Run a fresh `iterations`-pass prove/verify of a single onehot-D
/// `Cfg` and record per-iteration wall-clock. Mirrors
/// [`measure_dense_speedup_config`] but uses [`ONEHOT_D`],
/// [`OneHotPoly`], and [`make_onehot_poly`]. The duplication is
/// deliberate per task constraints (over-engineering via traits is
/// out of scope for measurement-only infrastructure).
fn measure_onehot_speedup_config<Cfg>(
    nv: usize,
    label: &str,
    iterations: usize,
    poly_seed: u64,
    point_seed: u64,
    transcript_label: &[u8],
) -> Option<SpeedupMeasurement>
where
    Cfg: CommitmentConfig<Field = F> + akita_planner::PlannerConfig,
{
    let layout = match Cfg::commitment_layout(nv) {
        Ok(l) => l,
        Err(e) => {
            tracing::info!(
                target: "speedup",
                "skip {label} at NV={nv}: layout err={e:?}"
            );
            return None;
        }
    };

    let schedule =
        Cfg::get_params_for_prove(nv, nv, 1, akita_types::AkitaRootBatchSummary::singleton())
            .expect("schedule must resolve when layout succeeds");
    let (routing_count, tiers) = inspect_cascade_schedule(&schedule);
    tracing::info!(
        target: "speedup",
        nv,
        label = %label,
        routing_count,
        ?tiers,
        "onehot speedup config schedule",
    );

    let poly = make_onehot_poly(&layout, poly_seed);
    let pt = random_point(nv, point_seed);
    let opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);
    let setup =
        <AkitaCommitmentScheme<ONEHOT_D, Cfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(
            nv, 1, 1,
        );
    let verifier_setup =
        <AkitaCommitmentScheme<ONEHOT_D, Cfg> as CommitmentProver<F, ONEHOT_D>>::setup_verifier(
            &setup,
        );

    let mut prove_times = Vec::with_capacity(iterations);
    let mut verify_times = Vec::with_capacity(iterations);

    for iter in 0..iterations {
        let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, Cfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::commit(std::slice::from_ref(&poly), &setup)
        .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(transcript_label);
        let t0 = Instant::now();
        let proof =
            <AkitaCommitmentScheme<ONEHOT_D, Cfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
                &setup,
                prove_input(&pt, &poly_refs, &commitments[0], hint),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");
        let prove_elapsed = t0.elapsed();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(transcript_label);
        let t0 = Instant::now();
        <AkitaCommitmentScheme<ONEHOT_D, Cfg> as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("verify");
        let verify_elapsed = t0.elapsed();

        let warmup = iter == 0;
        tracing::info!(
            target: "speedup",
            nv,
            label = %label,
            iter,
            warmup,
            prove_s = prove_elapsed.as_secs_f64(),
            verify_s = verify_elapsed.as_secs_f64(),
            "onehot speedup iter",
        );

        prove_times.push(prove_elapsed);
        verify_times.push(verify_elapsed);
    }

    Some(SpeedupMeasurement {
        label: label.to_string(),
        routing_count,
        tiers,
        prove_times,
        verify_times,
    })
}

/// Drop the first (warmup) iteration and return the median of the
/// rest. Returns `None` if fewer than two iterations were recorded.
fn median_measured(times: &[Duration]) -> Option<Duration> {
    let measured = times.get(1..)?;
    if measured.is_empty() {
        return None;
    }
    let mut sorted: Vec<Duration> = measured.to_vec();
    sorted.sort();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        let lo = sorted[mid - 1].as_nanos();
        let hi = sorted[mid].as_nanos();
        // u128 avg of two Duration nanos fits trivially in u64 for
        // wall-clock measurements bounded well below 2^64 ns ≈ 584 yr.
        Some(Duration::from_nanos(((lo + hi) / 2) as u64))
    }
}

/// Parse a comma-separated list of NVs from `var`, falling back to
/// `default` on unset or empty.
fn parse_nvs_env(var: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(var) {
        Ok(s) => {
            let parsed: Vec<usize> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            if parsed.is_empty() {
                default.to_vec()
            } else {
                parsed
            }
        }
        Err(_) => default.to_vec(),
    }
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

/// Audit B-3 / S-3: tamper the proof-payload meta material on the wire
/// (specifically `proof.root.stage2.next_w_commitment`, which in the
/// cascade case binds the routed S chunks + meta into the joint L+1
/// witness commitment per book §5.4 lines 692-699). The verifier must
/// reject with `InvalidProof` because the recursive opening replay at
/// L+1 will catch any perturbation of the committed routed material.
///
/// Replaces the prior `tiered_rejects_tampered_meta_material` test
/// (deleted before this commit), which poisoned `tiered_s_cache` —
/// material the verifier did not read at the time. The cache is now
/// read by the verifier hot path (commits `831ccfc`, `4a4c40b`,
/// `8e87160`), but external tampering of cache values is now
/// structurally impossible because `tiered_s_cache_get_or_init`
/// accepts only a derivation closure that reads from
/// `setup.expanded.shared_matrix` — no public setter, no way to
/// inject arbitrary bytes. So the only tamper-relevant artifact is
/// the proof payload itself; this test exercises that path.
///
/// Uses `DenseCascadeSmallCfg` (cascade `(2, 2)`) at NV=19, the
/// smallest configuration where the cascade actually fires E2E
/// (routing_count = 2, tiers = [2, 2]).
#[test]
fn tiered_rejects_tampered_next_w_commitment() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 19;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, DenseCascadeSmallCfg>;

        let layout = DenseCascadeSmallCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);
        let poly = make_dense_poly(NV, 0x715e_b101);
        let pt = random_point(NV, 0x715e_b102);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/tamper_next_w");
        let mut proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("cascade prove");

        // Tamper a coefficient in the L+1 witness commitment. In the
        // cascade case this commitment binds the routed S chunks +
        // meta material as part of the joint W+S witness for L+1
        // (book §5.5 Round 5). Perturbing one field element here
        // either breaks the recursive trace check or the recursive
        // opening replay at L+1 — both reject paths exit with
        // `AkitaError::InvalidProof`.
        let fold_root = proof
            .root
            .as_fold_mut()
            .expect("cascade proof must carry a fold root");
        let coeffs = fold_root.stage2.next_w_commitment.coeffs_mut();
        assert!(
            !coeffs.is_empty(),
            "cascade next_w_commitment must carry routed material"
        );
        coeffs[0] += F::from_u64(1);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/tamper_next_w");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tampered next_w_commitment (routed meta material binding) must reject"
        );
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
