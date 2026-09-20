/// Standalone CPU scenarios used by integration tests and benchmarks.
pub use super::recursive::opening::{
    ExtensionOpeningReductionGroup, ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm,
};
pub use super::recursive::{DigitRangeProver, LowBasisRangeCheckProver};
use crate::opaque::RootOpeningSource;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

/// Opaque recursive-witness input for standalone benchmarks.
pub struct RecursiveWitnessBenchmark {
    witness: super::recursive::RecursiveWitnessFlat,
}

/// Construct a packed recursive witness for a named standalone benchmark.
pub fn recursive_witness_from_i8_digits(digits: Vec<i8>) -> RecursiveWitnessBenchmark {
    RecursiveWitnessBenchmark {
        witness: super::recursive::RecursiveWitnessFlat::from_i8_digits(digits),
    }
}

/// Run the canonical CPU decompose-fold path over an opaque recursive witness.
pub fn decompose_recursive_witness<F, const D: usize>(
    witness: &RecursiveWitnessBenchmark,
    challenges: &[akita_challenges::SparseChallenge],
    num_positions_per_block: usize,
    num_digits: usize,
    log_basis: u32,
) -> Result<(), akita_error::AkitaError>
where
    F: Field + CanonicalEncoding,
{
    let view = witness.witness.view::<F, D>()?;
    let output = view.decompose_fold(challenges, num_positions_per_block, num_digits, log_basis)?;
    std::hint::black_box(output);
    Ok(())
}

/// Execute and retain the result of the packed recursive coefficient benchmark.
pub fn recursive_witness_coefficient_packing_partials<F, E, const D: usize>(
    witness: &RecursiveWitnessBenchmark,
    point: &akita_types::PreparedSubringCoefficientPackingPoint<E>,
) -> Result<(), akita_error::AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + MulBaseUnreduced<F> + akita_types::FpExtEncoding<F>,
{
    let weights = crate::arithmetic::coefficient_packing::FusedPackingWeights::new(point)?;
    std::hint::black_box(
        super::recursive::suffix_witness_coefficient_packing_partials::<F, E, D>(
            &witness.witness,
            &weights,
        )?,
    );
    Ok(())
}
/// Execute the dense coefficient-packing benchmark using canonical arithmetic.
pub fn dense_coefficient_packing<F, E, const D: usize>(
    poly: &crate::DensePoly<F>,
    point: &akita_types::PreparedSubringCoefficientPackingPoint<E>,
) -> Result<(), akita_error::AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + MulBaseUnreduced<F> + akita_types::FpExtEncoding<F>,
{
    let refs = [poly];
    let batch = <crate::DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&refs)?;
    let weights = crate::arithmetic::coefficient_packing::FusedPackingWeights::new(point)?;
    std::hint::black_box(crate::sources::dense::dense_coefficient_packing_partials(
        batch, &weights,
    )?);
    Ok(())
}
/// Execute the sparse coefficient-packing benchmark including adaptive weight preparation.
pub fn onehot_coefficient_packing<F, E, const D: usize>(
    poly: &crate::OneHotPoly<F, u8>,
    point: &akita_types::PreparedSubringCoefficientPackingPoint<E>,
) -> Result<(), akita_error::AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F>,
{
    let refs = [poly];
    let batch = <crate::OneHotPoly<F, u8> as RootOpeningSource<F, D>>::opening_batch(&refs)?;
    std::hint::black_box(crate::sources::onehot::onehot_coefficient_packing_batch(
        batch,
        crate::opaque::SubringCoefficientPackingPlan { point },
    )?);
    Ok(())
}
