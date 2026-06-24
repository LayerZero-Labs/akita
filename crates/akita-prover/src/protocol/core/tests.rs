use super::*;
use crate::RecursiveWitnessFlat;
use akita_field::{AkitaError, Fp32, FpExt2, NegOneNr};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaScheduleLookupKey, CommitmentGroup, OpeningBatchShape, VerifierOpeningBatch,
};

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
    let logical_polys = [&logical_w];

    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    #[cfg(feature = "zk")]
    let mut zk_hiding = ZkHidingProverState::new((1..=16).map(F::from_u64).collect::<Vec<_>>());
    let opening_batch = VerifierOpeningBatch::from_groups(
        point.to_vec(),
        vec![CommitmentGroup {
            claims: vec![E::zero()],
            commitment: (),
        }],
    )
    .expect("opening batch");
    let proved = prove_extension_opening_reduction::<F, E, _, RecursiveWitnessFlat, _, 2>(
        &crate::compute::CpuBackend,
        None,
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
    assert_eq!(proved.reduction.proof.num_rounds(), point.len() - 1);
}

#[test]
fn batched_prove_opening_batch_rejects_multi_group_shape() {
    let batch = OpeningBatchShape::from_commitment_groups(4, &[1, 2]).expect("grouped shape");
    assert_eq!(batch.num_commitment_groups(), 2);
    assert!(matches!(
        AkitaScheduleLookupKey::new_from_opening_batch(&batch),
        Err(AkitaError::InvalidSetup(_))
    ));
}
