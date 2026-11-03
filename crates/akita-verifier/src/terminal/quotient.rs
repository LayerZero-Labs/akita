//! Probabilistic terminal product certificate for the v8 proof channel.
//!
//! The last commitment block is bound by its previously absorbed digest.
//! All quotient coefficients precede both extension-field check points.
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_error::AkitaError;
use akita_params::TerminalFoldParams;
use akita_transcript::ChallengeCodec;
use akita_types::{FoldSite, GrindingReplay, RingVec, TerminalResponse};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_and_check<F, E>(
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    scheduled: &TerminalFoldParams,
    setup: &[F],
    response: &TerminalResponse<F>,
    challenges: &Challenges,
    last_digest: &[u8; 32],
) -> Result<RingVec<F>, AkitaError>
where
    F: Field + CanonicalEncoding + Ring,
    E: ExtField<F> + ChallengeCodec,
{
    let d = scheduled.d_a();
    let quotient_len = akita_types::terminal_quotient_len(d)?;
    // A modulus with b bits is at least 2^(b-1). Require at least
    // 65 bits per polynomial check, conservatively rounding its degree up.
    let degree = quotient_len.checked_mul(2).ok_or(AkitaError::InvalidProof)?;
    let degree_bits = usize::BITS - (degree - 1).leading_zeros();
    let field_bits = (F::MODULUS_BITS as usize).saturating_sub(1)
        .checked_mul(E::DEGREE).ok_or(AkitaError::InvalidProof)?;
    if field_bits.saturating_sub(degree_bits as usize) < 65 {
        return Err(AkitaError::InvalidSetup("terminal certificate challenge field is too small".into()));
    }
    let rows = scheduled.inner.matrix.output_rank();
    let last_count = rows.checked_mul(d).ok_or(AkitaError::InvalidProof)?;
    let quotient_count = rows.checked_mul(quotient_len).ok_or(AkitaError::InvalidProof)?;
    let last = akita_transcript::receive_field_group::<F>(
        grinding.state_mut(), FoldSite::TerminalLastT { level }.id()?, last_count,
    )?;
    if akita_transcript::field_digest(&last) != *last_digest {
        return Err(AkitaError::InvalidProof);
    }
    let coefficients = akita_transcript::receive_field_group::<F>(
        grinding.state_mut(), FoldSite::TerminalQuotient { level }.id()?, quotient_count,
    )?;
    let points = akita_transcript::ext_challenges::<E, _>(
        grinding.state_mut(), FoldSite::TerminalQuotientPoints { level }.id()?, 2,
    )?;
    let group = scheduled.response_shape.layout.groups.first().ok_or(AkitaError::InvalidProof)?;
    let z = akita_types::decode_terminal_z_golomb_payload(
        response.z_payloads.first().ok_or(AkitaError::InvalidProof)?, group,
    )?;
    let expected_z = scheduled.inner.matrix.input_width().checked_mul(d)
        .ok_or(AkitaError::InvalidProof)?;
    if z.len() != expected_z || challenges.len() != scheduled.blocks.live_blocks {
        return Err(AkitaError::InvalidProof);
    }
    let last = RingVec::from_coeffs(last);
    let products = akita_params::dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner), F, d, |D| {
            let sent = response.t_fields.as_ring_slice::<D>()?;
            let last = last.as_ring_slice::<D>()?;
            let expected_sent = challenges.len().checked_sub(1)
                .and_then(|v| v.checked_mul(rows)).ok_or(AkitaError::InvalidProof)?;
            if sent.len() != expected_sent || last.len() != rows || rows == 0 {
                return Err(AkitaError::InvalidProof);
            }
            let products = (0..rows).map(|row| {
                let end = last.get(row).ok_or(AkitaError::InvalidProof)?;
                super::direct::sparse_challenge_dot(
                    challenges.as_slice(), sent.iter().skip(row).step_by(rows).chain([end]),
                )
            }).collect::<Result<Vec<CyclotomicRing<F, D>>, AkitaError>>()?;
            Ok::<_, AkitaError>(RingVec::from_ring_elems(&products))
        }
    )?;
    check_polynomials(setup, &z, products.coeffs(), &coefficients, &points, d, rows)?;
    Ok(products)
}

fn evaluate<F: Field, E: ExtField<F>>(coefficients: &[F], point: E) -> E {
    coefficients.iter().rev().fold(E::zero(), |value, &coefficient| {
        value * point + E::lift_base(coefficient)
    })
}

#[allow(clippy::too_many_arguments)]
fn check_polynomials<F: Field, E: ExtField<F>>(
    setup: &[F], z: &[i16], products: &[F], quotients: &[F], points: &[E],
    d: usize, rows: usize,
) -> Result<(), AkitaError> {
    let q_len = akita_types::terminal_quotient_len(d)?;
    let row_len = z.len();
    let prefix = rows.checked_mul(row_len).ok_or(AkitaError::InvalidProof)?;
    let product_len = rows.checked_mul(d).ok_or(AkitaError::InvalidProof)?;
    let quotient_len = rows.checked_mul(q_len).ok_or(AkitaError::InvalidProof)?;
    if !row_len.is_multiple_of(d) || products.len() != product_len
        || quotients.len() != quotient_len || points.len() != 2 || rows == 0 || row_len == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let matrix = setup.get(..prefix).ok_or(AkitaError::InvalidProof)?;
    let z_fields: Vec<F> = z.iter().map(|&v| F::from_i64(i64::from(v))).collect();
    for &point in points {
        let shift = (0..d).fold(E::one(), |v, _| v * point) + E::one();
        let z_evals: Vec<E> = z_fields.chunks_exact(d).map(|ring| evaluate(ring, point)).collect();
        for ((row, product), quotient) in matrix.chunks_exact(row_len)
            .zip(products.chunks_exact(d)).zip(quotients.chunks_exact(q_len)) {
            let actual = row.chunks_exact(d).zip(&z_evals)
                .fold(E::zero(), |v, (a, z)| v + evaluate(a, point) * *z);
            if actual - evaluate(product, point) != shift * evaluate(quotient, point) {
                return Err(AkitaError::InvalidProof);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Prime128OffsetA7F7 as F, Ring};

    #[test]
    fn quotient_matches_exact_negacyclic_product_and_rejects_tampering() {
        let d = 4;
        let matrix: Vec<F> = (1..=16).map(F::from_u64).collect();
        let z = [2, -3, 0, 5, -7, 11, 0, 1];
        let mut products = vec![F::zero(); 2 * d];
        for row in 0..2 {
            for column in 0..2 {
                for i in 0..d {
                    for j in 0..d {
                        let term = matrix[row * 2 * d + column * d + i]
                            * F::from_i64(i64::from(z[column * d + j]));
                        if i + j < d {
                            products[row * d + i + j] += term;
                        } else {
                            products[row * d + i + j - d] -= term;
                        }
                    }
                }
            }
        }
        let q = akita_types::terminal_a_row_quotients(&matrix, &z, d, 2).unwrap();
        let points = [F::from_u64(2), F::from_u64(3)];
        assert!(check_polynomials(&matrix, &z, &products, &q, &points, d, 2).is_ok());
        let mut changed = q.clone();
        changed[0] += F::one();
        assert!(check_polynomials(&matrix, &z, &products, &changed, &points, d, 2).is_err());
        products[0] += F::one();
        assert!(check_polynomials(&matrix, &z, &products, &q, &points, d, 2).is_err());
        for dimension in [0, 1, usize::MAX] {
            assert!(check_polynomials(&[], &[], &[], &[], &points, dimension, 0).is_err());
        }
    }
}
