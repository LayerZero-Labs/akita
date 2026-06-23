use super::*;
use akita_field::{AkitaError, Fp32, FpExt2, LiftBase, NegOneNr};
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaScheduleLookupKey, OpeningBatch};

type F = Fp32<251>;
type E = FpExt2<F, NegOneNr>;

#[test]
fn recursive_extension_opening_reduction_pads_to_opening_cube() {
    let logical_w = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2, 0, 3, -2]);
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
    ];
    let logical_view = logical_w.view::<F, 2>().expect("valid suffix witness");
    let mut base_evals = logical_view.base_evals().expect("base evals");
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
    let logical_polys = [&logical_view];
    let opening_batch =
        OpeningBatch::with_claims(point.to_vec(), vec![E::zero()]).expect("opening batch");
    let proved = prove_extension_opening_reduction::<F, E, _, _, 2>(
        &logical_polys,
        &opening_batch,
        #[cfg(feature = "zk")]
        None,
        true,
        &mut transcript,
        "recursive",
        #[cfg(feature = "zk")]
        &mut zk_hiding,
    )
    .expect("padded logical witnesses should reduce over the opening cube");

    assert_eq!(
        proved.reduction.proof.partials.len(),
        <E as ExtField<F>>::EXT_DEGREE
    );
    assert_eq!(proved.openings, vec![expected_opening]);
    assert_eq!(proved.reduction.proof.num_rounds(), point.len() - 1);
}

#[test]
fn batched_prove_opening_batch_rejects_multi_group_shape() {
    let batch = OpeningBatch::from_commitment_groups(4, &[1, 2]).expect("grouped shape");
    assert_eq!(batch.num_commitment_groups(), 2);
    assert!(matches!(
        AkitaScheduleLookupKey::new_from_opening_batch(&batch),
        Err(AkitaError::InvalidSetup(_))
    ));
}
