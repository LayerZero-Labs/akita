use super::*;
use crate::DensePoly;
use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr};
use akita_transcript::AkitaTranscript;
#[cfg(feature = "zk")]
use akita_types::FlatDigitBlocks;
use akita_types::{AkitaSetupSeed, FlatMatrix};

type F = Fp32<251>;
type E = FpExt2<F, NegOneNr>;

fn setup() -> AkitaExpandedSetup<F> {
    AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 3,
            max_num_batched_polys: 4,
            gen_ring_dim: 1,
            max_setup_len: 1,
            #[cfg(feature = "zk")]
            max_zk_b_len: 1,
            #[cfg(feature = "zk")]
            max_zk_d_len: 1,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
        #[cfg(feature = "zk")]
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
        #[cfg(feature = "zk")]
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
    )
}

#[test]
fn prover_claim_preparation_accepts_extension_points() {
    let point = [
        E::new(F::from_u64(1), F::from_u64(2)),
        E::new(F::from_u64(3), F::from_u64(4)),
    ];
    let polys = [
        DensePoly::<F, 2>::from_field_evals(2, &[F::from_u64(10), F::zero(), F::zero(), F::zero()])
            .expect("poly"),
        DensePoly::<F, 2>::from_field_evals(2, &[F::from_u64(11), F::zero(), F::zero(), F::zero()])
            .expect("poly"),
    ];
    let commitment = RingCommitment::<F, 2>::default();
    #[cfg(feature = "zk")]
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        Vec::new(),
        Vec::new(),
        vec![FlatDigitBlocks::empty()],
    );
    #[cfg(not(feature = "zk"))]
    let hint = AkitaCommitmentHint::new(Vec::new());
    let claims = (
        &point[..],
        vec![crate::CommittedPolynomials {
            polynomials: &polys[..],
            commitment: &commitment,
            hint,
        }],
    );

    let prepared = prepare_batched_prove_inputs::<F, E, DensePoly<F, 2>, 2>(&setup(), claims)
        .expect("extension-valued prover points should validate by shape");

    assert_eq!(prepared.opening_point, &point[..]);
    assert_eq!(prepared.incidence_summary.num_claims(), 2);
    assert_eq!(prepared.incidence_summary.num_polys_per_point(), &[2]);
    assert_eq!(prepared.incidence_summary.claim_to_point(), &[0, 0]);
    assert_eq!(prepared.flat_polys, vec![&polys[0], &polys[1]]);
}

#[test]
fn recursive_extension_opening_reduction_pads_to_opening_cube() {
    let logical_w = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2, 0, 3, -2]);
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
    ];
    let mut base_evals = recursive_witness_base_evals::<F>(&logical_w);
    base_evals.resize(1usize << point.len(), F::zero());
    let expected_opening = base_evals
        .iter()
        .enumerate()
        .fold(E::zero(), |acc, (idx, &eval)| {
            let weight = point
                .iter()
                .enumerate()
                .fold(E::one(), |weight, (bit, &x)| {
                    if (idx >> bit) & 1 == 1 {
                        weight * x
                    } else {
                        weight * (E::one() - x)
                    }
                });
            acc + weight * E::lift_base(eval)
        });

    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    #[cfg(feature = "zk")]
    let mut zk_hiding = ZkHidingProverState::new((1..=16).map(F::from_u64).collect::<Vec<_>>());
    let reduction = prove_extension_opening_reduction::<F, E, _>(
        &logical_w,
        &point,
        expected_opening,
        &mut transcript,
        #[cfg(feature = "zk")]
        &mut zk_hiding,
    )
    .expect("padded logical witnesses should reduce over the opening cube");

    assert_eq!(
        reduction.proof.partials.len(),
        <E as ExtField<F>>::EXT_DEGREE
    );
    assert_eq!(reduction.proof.num_rounds(), point.len() - 1);
}
