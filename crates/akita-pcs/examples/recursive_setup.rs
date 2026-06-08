//! End-to-end demonstration of the **recursive setup-contribution** verifier
//! path (`SetupContributionMode::Recursive`).
//!
//! Most of Akita's examples and the `profile` harness run the verifier in
//! `SetupContributionMode::Direct`, where the verifier scans the expanded setup
//! matrix inline. This example instead proves and verifies a single one-hot
//! polynomial under `Recursive` mode, where each non-terminal fold level
//! delegates its setup contribution to a setup-product sumcheck (the Stage-3
//! `SetupSumcheckProver` / `SetupSumcheckVerifier` pair).
//!
//! It then re-proves and verifies the same opening under `Direct` mode and
//! prints a per-mode summary, so the two paths can be compared side by side.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p akita-pcs --example recursive_setup
//! AKITA_NUM_VARS=25 cargo run --release -p akita-pcs --example recursive_setup
//! ```

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::compute::{
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootOpeningSource,
};
use akita_prover::{
    CommitmentProver, CommittedPolynomials, ComputeBackendSetup, CpuBackend, OneHotPoly,
    RootCommitPolys,
};
use akita_transcript::AkitaTranscript;
use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaBatchedProof,
    AkitaBatchedRootProof, AkitaProofStep, BasisMode, BlockOrder, ClaimIncidenceSummary,
    CommittedOpenings, LevelParams, SetupContributionMode,
};
use akita_verifier::CommitmentVerifier;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type F = fp128::Field;
type Cfg = fp128::D64OneHot;
const D: usize = 64;
const ONEHOT_K: usize = D;
const TRANSCRIPT_DOMAIN: &[u8] = b"akita-pcs/example/recursive-setup";

type Scheme = AkitaCommitmentScheme<D, Cfg>;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Count of non-terminal fold levels — the ones that run the recursive
/// setup-product sumcheck.
fn setup_sumcheck_levels(proof: &AkitaBatchedProof<F, F>) -> usize {
    let root = matches!(proof.root, AkitaBatchedRootProof::Fold(_)) as usize;
    let suffix = proof
        .steps
        .iter()
        .filter(|step| matches!(step, AkitaProofStep::Intermediate(_)))
        .count();
    root + suffix
}

fn opening_from_poly(poly: &OneHotPoly<F, D, u8>, point: &[F], layout: &LevelParams) -> F {
    opening_from_poly_impl::<D, OneHotPoly<F, D, u8>>(poly, point, layout)
}

fn opening_from_poly_impl<const D: usize, P: RootOpeningSource<F, D>>(
    poly: &P,
    point: &[F],
    layout: &LevelParams,
) -> F
where
    CpuBackend: for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    let alpha_bits = D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);
    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");
    let OpeningFoldOutput { eval: y_ring, .. } = OpeningFoldKernel::evaluate_and_fold(
        &CpuBackend,
        None,
        poly.opening_view().expect("opening view"),
        OpeningFoldPlan::Base {
            eval_outer_scalars: &ring_opening_point.b,
            fold_scalars: &ring_opening_point.a,
            block_len: layout.block_len,
        },
    )
    .expect("evaluate_and_fold");
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, BasisMode::Lagrange)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

fn run_mode(
    mode: SetupContributionMode,
    nv: usize,
    poly: &OneHotPoly<F, D, u8>,
    point: &[F],
    opening: F,
) {
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(nv, 1, 1).expect("setup_prover");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepare_setup");
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let commit_input = std::slice::from_ref(poly);
    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::new(commit_input),
        &CpuBackend,
        &prepared,
    )
    .expect("commit");

    let poly_refs: [&OneHotPoly<F, D, u8>; 1] = [poly];
    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            point,
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitment,
                hint,
            },
        )],
        &CpuBackend,
        &prepared,
        &mut prover_transcript,
        BasisMode::Lagrange,
        mode,
    )
    .expect("batched_prove");

    let fold_levels = setup_sumcheck_levels(&proof);

    let openings = [opening];
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            point,
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitment,
            },
        )],
        BasisMode::Lagrange,
        mode,
    )
    .expect("batched_verify");

    let setup_note = match mode {
        SetupContributionMode::Recursive => {
            format!("{fold_levels} level(s) ran the setup-product sumcheck")
        }
        SetupContributionMode::Direct => {
            format!("{fold_levels} non-terminal level(s) scanned the setup matrix inline")
        }
    };
    println!("  {mode:?}: verified OK — {setup_note}");
}

fn main() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(256 * 1024 * 1024)
        .build_global()
        .ok();

    let nv = env_usize("AKITA_NUM_VARS", 20);

    let layout = Cfg::get_params_for_batched_commitment(
        &ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence"),
    )
    .expect("layout");
    let total_ring = layout.num_blocks * layout.block_len;
    assert_eq!(
        total_ring * ONEHOT_K,
        1usize << nv,
        "AKITA_NUM_VARS={nv} must match the D=64 OneHot layout"
    );

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    let poly = OneHotPoly::<F, D, u8>::new(ONEHOT_K, indices).expect("onehot poly");
    let point: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = opening_from_poly(&poly, &point, &layout);

    println!("Akita recursive setup-contribution example (D={D} OneHot, nv={nv})");
    run_mode(SetupContributionMode::Recursive, nv, &poly, &point, opening);
    run_mode(SetupContributionMode::Direct, nv, &poly, &point, opening);
}
