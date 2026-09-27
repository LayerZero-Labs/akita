//! Pure fold kernels and operation adapters for the prover core.
//!
//! Everything here passes the spec's kernel discriminator: no function reads a
//! typed fold parameters. Const-D functions receive extracted numbers and typed
//! buffers; the D-free functions are operation adapters that dispatch exactly
//! once on a schedule-derived ring dimension supplied by the caller.

use super::*;

/// Prepared public evaluation-trace claim and its per-opening coefficients (#314 Stage-2).
pub(in crate::protocol) struct PreparedEvaluationTraceClaim<E: Field> {
    pub(in crate::protocol) claimed_evaluation: E,
    pub(in crate::protocol) claim_coefficients: Vec<E>,
}

fn resolve_evaluation_trace_claim<E: Field>(
    reduction: Option<&ExtensionOpeningReduction<E>>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    row_coefficients: &[E],
) -> Result<PreparedEvaluationTraceClaim<E>, AkitaError> {
    if reduction.is_some_and(|reduction| {
        reduction.final_claims.len() != opening_batch.num_total_polynomials()
            || reduction.final_factors.len() != opening_batch.num_groups()
    }) {
        return Err(AkitaError::InvalidProof);
    }
    let claim_coefficients = reduction.map_or_else(
        || Ok(row_coefficients.to_vec()),
        |reduction| {
            opening_batch
                .scale_row_coefficients_by_group(row_coefficients, &reduction.final_factors)
        },
    )?;
    let expected = opening_batch
        .batched_eval_target(&claim_coefficients, openings)
        .map_err(|err| {
            AkitaError::InvalidInput(format!("batched trace evaluation failed: {err:?}"))
        })?;
    let claimed = match reduction {
        Some(reduction) => opening_batch
            .batched_eval_target(row_coefficients, &reduction.final_claims)
            .map_err(|_| AkitaError::InvalidProof)?,
        None => expected,
    };
    if claimed != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(PreparedEvaluationTraceClaim {
        claimed_evaluation: claimed,
        claim_coefficients,
    })
}

pub(in crate::protocol) fn row_coefficient_rings<F, E, const D: usize>(
    coefficients: &[E],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: Field + Ring,
    E: FpExtEncoding<F>,
{
    coefficients
        .iter()
        .copied()
        .map(|coefficient| {
            embed_ring_subfield_scalar::<F, E, D>(
                coefficient,
                AkitaError::InvalidInput(
                    "public-row coefficient does not encode in the ring-subfield basis".to_string(),
                ),
            )
        })
        .collect()
}

pub(in crate::protocol) fn prepare_evaluation_trace_claim<F, E>(
    reduction: &Option<ExtensionOpeningReduction<E>>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
) -> Result<(PreparedEvaluationTraceClaim<E>, Vec<E>), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F>,
{
    if openings.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidSize {
            expected: opening_batch.num_total_polynomials(),
            actual: openings.len(),
        });
    }
    let row_coefficients = akita_types::sample_row_coefficients_native::<F, E>(
        opening_batch,
        akita_types::GrindingSite::EvaluationBatch { level },
        grinding,
    )?;
    let resolved = resolve_evaluation_trace_claim(
        reduction.as_ref(),
        openings,
        opening_batch,
        &row_coefficients,
    )?;
    Ok((resolved, row_coefficients))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fp32, One};

    type TestF = Fp32<251>;

    fn reduction(
        final_claims: Vec<TestF>,
        final_factors: Vec<TestF>,
    ) -> ExtensionOpeningReduction<TestF> {
        ExtensionOpeningReduction {
            final_claims,
            final_factors,
        }
    }

    fn grouped_layout() -> OpeningClaimsLayout {
        OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(0, 2),
            PolynomialGroupLayout::new(0, 1),
        ])
        .expect("grouped opening layout")
    }

    #[test]
    fn trace_claim_resolution_scales_groups_and_binds_terminal_handles() {
        let layout = grouped_layout();
        let openings = [TestF::one(), TestF::one(), TestF::one()];
        let row_coefficients = [TestF::one(), TestF::one(), TestF::one()];
        let reduction = reduction(
            vec![TestF::one(), TestF::one(), -TestF::one()],
            vec![TestF::one(), -TestF::one()],
        );

        let resolved =
            resolve_evaluation_trace_claim(Some(&reduction), &openings, &layout, &row_coefficients)
                .expect("valid grouped trace claim");

        assert_eq!(resolved.claimed_evaluation, TestF::one());
        assert_eq!(
            resolved.claim_coefficients,
            vec![TestF::one(), TestF::one(), -TestF::one()]
        );

        let unreduced = resolve_evaluation_trace_claim(None, &openings, &layout, &row_coefficients)
            .expect("valid unreduced trace claim");
        assert_eq!(
            unreduced.claimed_evaluation,
            TestF::one() + TestF::one() + TestF::one()
        );
        assert_eq!(unreduced.claim_coefficients, row_coefficients);
    }

    #[test]
    fn trace_claim_resolution_rejects_malformed_reduction_shapes() {
        let layout = grouped_layout();
        let openings = [TestF::one(), TestF::one(), TestF::one()];
        let row_coefficients = [TestF::one(), TestF::one(), TestF::one()];
        let short_claims = reduction(
            vec![TestF::one(), TestF::one()],
            vec![TestF::one(), TestF::one()],
        );
        let short_factors = reduction(
            vec![TestF::one(), TestF::one(), TestF::one()],
            vec![TestF::one()],
        );

        for malformed in [&short_claims, &short_factors] {
            assert!(matches!(
                resolve_evaluation_trace_claim(
                    Some(malformed),
                    &openings,
                    &layout,
                    &row_coefficients,
                ),
                Err(AkitaError::InvalidProof)
            ));
        }
    }

    #[test]
    fn trace_claim_resolution_rejects_inconsistent_terminal_handles() {
        let layout = grouped_layout();
        let openings = [TestF::one(), TestF::one(), TestF::one()];
        let row_coefficients = [TestF::one(), TestF::one(), TestF::one()];
        let inconsistent = reduction(
            vec![TestF::one(), TestF::one(), TestF::one()],
            vec![TestF::one(), -TestF::one()],
        );

        assert!(matches!(
            resolve_evaluation_trace_claim(
                Some(&inconsistent),
                &openings,
                &layout,
                &row_coefficients,
            ),
            Err(AkitaError::InvalidProof)
        ));
    }
}
