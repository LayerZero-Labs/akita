//! The prover must not overflow rayon's default (2 MiB) worker stacks.
//!
//! Every other e2e installs a 256 MiB global rayon pool (`init_rayon_pool`)
//! and runs its body on a 256 MiB thread (`run_on_large_stack`), so the
//! prover's true stack appetite never shows up in this suite — but library
//! callers inherit rayon's defaults. This test proves on the default global
//! pool: before the fix it aborts with a worker-thread stack overflow
//! (nondeterministic — the depth depends on work-stealing migrations).

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::{CanonicalField, FieldCore};
use akita_pcs::{AkitaCommitmentScheme, ComputeBackendSetup, CpuBackend};
use akita_prover::{OneHotPoly, ProverOpeningData};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaCommitmentHint, BasisMode, Commitment, OpeningClaims, PointVariableSelection,
    PolynomialGroupClaims,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type F = fp128::Field;

fn prove_input<'a, FF: FieldCore + Clone, P, CommitF: FieldCore>(
    point: &'a [FF],
    polynomials: &'a [&'a P],
    commitment: &'a Commitment<CommitF>,
    hint: AkitaCommitmentHint<CommitF>,
) -> ProverOpeningData<'a, FF, P, CommitF> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).expect("full-point prover group"),
        vec![FF::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims =
        OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("valid opening claims");
    ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
        .expect("valid prover opening data")
}

#[test]
fn prove_succeeds_on_default_worker_stacks() {
    type Cfg = fp128::D64OneHotK16;
    const D: usize = <Cfg as CommitmentConfig>::D;
    const K: usize = 16;
    const NV: usize = 17;
    const POLYS: usize = 56;
    const ROUNDS: usize = 5;

    let opening_batch =
        akita_types::OpeningClaimsLayout::new(NV, POLYS).expect("opening batch layout");
    let layout = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
    let total_field = layout.num_live_blocks * layout.num_positions_per_block * D;
    let total_chunks = total_field / K;
    assert_eq!(total_chunks * K, total_field);

    let polys: Vec<OneHotPoly<F>> = (0..POLYS)
        .map(|poly_idx| {
            let mut rng = StdRng::seed_from_u64(0x0defa117 + poly_idx as u64);
            let indices: Vec<Option<usize>> = (0..total_chunks)
                .map(|_| Some(rng.gen_range(0..K)))
                .collect();
            OneHotPoly::<F>::new(K, D, indices).expect("one-hot poly")
        })
        .collect();
    let poly_refs: Vec<&OneHotPoly<F>> = polys.iter().collect();
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    let point: Vec<F> = (0..NV)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, POLYS).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    for round in 0..ROUNDS {
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack).expect("commit");
        let mut transcript = AkitaTranscript::<F>::new(b"akita_default_stack_repro/one-hot-k16");
        let _proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input(&point[..], &poly_refs[..], &commitment, hint),
            &stack,
            &mut transcript,
            BasisMode::Lagrange,
        )
        .unwrap_or_else(|err| panic!("round {round}: prove failed: {err}"));
    }
}
