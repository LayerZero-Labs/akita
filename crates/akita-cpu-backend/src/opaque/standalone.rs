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
    tables: &akita_types::SubringCoefficientPackingTables<E>,
) -> Result<(), akita_error::AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + MulBaseUnreduced<F> + akita_types::FpExtEncoding<F>,
{
    let weights = crate::arithmetic::coefficient_packing::FusedPackingWeights::new(tables)?;
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
    tables: &akita_types::SubringCoefficientPackingTables<E>,
) -> Result<(), akita_error::AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + MulBaseUnreduced<F> + akita_types::FpExtEncoding<F>,
{
    let refs = [poly];
    let batch = <crate::DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&refs)?;
    let weights = crate::arithmetic::coefficient_packing::FusedPackingWeights::new(tables)?;
    std::hint::black_box(crate::sources::dense::dense_coefficient_packing_partials(
        batch, &weights,
    )?);
    Ok(())
}
/// Execute the sparse coefficient-packing benchmark including adaptive weight preparation.
pub fn onehot_coefficient_packing<F, E, const D: usize>(
    poly: &crate::OneHotPoly<F, u8>,
    tables: &akita_types::SubringCoefficientPackingTables<E>,
) -> Result<(), akita_error::AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F>,
{
    let refs = [poly];
    let batch = <crate::OneHotPoly<F, u8> as RootOpeningSource<F, D>>::opening_batch(&refs)?;
    std::hint::black_box(crate::sources::onehot::onehot_coefficient_packing_batch(
        batch,
        crate::opaque::SubringCoefficientPackingPlan { tables },
    )?);
    Ok(())
}

/// Drive a complete ordinary Stage 1 digit-range session over compact digits.
///
/// Round, batching and merge challenges come from a fixed `seed` instead of a
/// transcript. The result lists every round polynomial's coefficients, then
/// each public transition's claims, in session order, so two runs can be
/// compared value for value.
pub fn digit_range_session<E>(
    digits: Vec<i8>,
    basis: usize,
    num_vars: usize,
    ring_variable_count: usize,
    seed: u64,
) -> Result<Vec<E>, akita_error::AkitaError>
where
    E: Field + jolt_field::Ring + jolt_field::Fold + jolt_field::Unreduced,
{
    use super::{Stage1PublicTransition, Stage1RoundPolynomial, Stage1Step, Stage1Transition};
    let mut state = seed;
    let mut challenge = move || {
        // SplitMix64.
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        E::from_u64(z ^ (z >> 31))
    };
    let domain = akita_params::FlatBooleanDomain::new(digits.len(), num_vars)?;
    let column_variable_count = num_vars.checked_sub(ring_variable_count).ok_or_else(|| {
        akita_error::AkitaError::InvalidInput("ring variables exceed the domain".into())
    })?;
    let point = (0..num_vars).map(|_| challenge()).collect::<Vec<_>>();
    let equality = akita_types::DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &point,
        column_variable_count,
        ring_variable_count,
    )?;
    let plan = akita_params::DigitRangePlan::new(basis)?;
    let prover = DigitRangeProver::from_packed_digits(
        crate::sources::packed_digits::PackedSignedDigits::from_i8_digits_auto(digits),
        plan,
        domain,
        equality,
    )?;
    let mut session = super::sumcheck::digit_range::DigitRangeSession::new(prover, None)?;
    let mut output = Vec::new();
    let mut claim = E::zero();
    for product in 0..plan.product_stage_arities().len() {
        for round in 0..num_vars {
            let polynomial =
                session.round_polynomial(Stage1Step::Product(product), round, claim)?;
            let Stage1RoundPolynomial::EqFactored(polynomial) = polynomial else {
                return Err(akita_error::AkitaError::Internal(
                    "product rounds are equality-factored".into(),
                ));
            };
            output.extend_from_slice(polynomial.coefficients());
            let r = challenge();
            session.bind_challenge(Stage1Step::Product(product), round, r)?;
            claim = session.current_claim();
        }
        if let Stage1PublicTransition::ProductChildClaims(claims) =
            session.public_transition(Stage1Step::Product(product))?
        {
            output.extend(claims);
        }
        session.bind_batch_challenge(Stage1Transition::ProductBatch(product), challenge())?;
        claim = session.current_claim();
    }
    for round in 0..num_vars {
        if let Stage1RoundPolynomial::EqFactored(polynomial) =
            session.round_polynomial(Stage1Step::RangeLeaf, round, claim)?
        {
            output.extend_from_slice(polynomial.coefficients());
        }
        session.bind_challenge(Stage1Step::RangeLeaf, round, challenge())?;
        claim = session.current_claim();
    }
    if let Stage1PublicTransition::Final {
        range_image_evaluation,
        ..
    } = session.public_transition(Stage1Step::RangeLeaf)?
    {
        output.push(range_image_evaluation);
    }
    output.push(session.finish()?.final_claim());
    Ok(output)
}
