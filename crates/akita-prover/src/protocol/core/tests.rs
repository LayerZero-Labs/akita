use super::*;
use crate::RecursiveWitnessFlat;
use akita_field::{AkitaError, Fp32, FpExt2, NegOneNr};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaScheduleLookupKey, OpeningClaims, OpeningClaimsLayout, PointVariableSelection,
    PolynomialGroupClaims,
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
    let opening_batch = OpeningClaims::from_groups(
        point.to_vec(),
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(point.len(), point.len()).expect("point vars"),
            vec![E::zero()],
            (),
        )
        .expect("group claims")],
    )
    .expect("opening batch");
    let proved = prove_extension_opening_reduction::<F, E, _, RecursiveWitnessFlat, _, 2>(
        &crate::compute::CpuBackend,
        None,
        &logical_polys,
        &opening_batch,
        true,
        &mut transcript,
        "recursive",
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
    let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 2]).expect("grouped shape");
    assert_eq!(batch.num_groups(), 2);
    assert!(matches!(
        AkitaScheduleLookupKey::from_layout(&batch),
        Err(AkitaError::InvalidSetup(_))
    ));
}
