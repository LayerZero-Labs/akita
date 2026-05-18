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

use akita_config::{ClaimReductionCfg, CommitmentConfig, TieredClaimReductionCfg};
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
            dense
                .as_ref()
                .is_ok_and(|l| l.use_setup_claim_reduction),
            onehot
                .as_ref()
                .is_ok_and(|l| l.use_setup_claim_reduction),
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
            dense
                .as_ref()
                .is_ok_and(|l| l.use_setup_claim_reduction),
            onehot
                .as_ref()
                .is_ok_and(|l| l.use_setup_claim_reduction),
        );
    }
}
