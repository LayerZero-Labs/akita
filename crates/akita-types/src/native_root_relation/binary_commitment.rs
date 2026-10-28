//! Canonical binary A-row assembly on the shared successor coefficient map.

use super::*;

/// Assemble `rem_Psi(A z - sum_b iota(c_b) T_b) = residual` for every
/// polynomial and matrix row. Response elements use polynomial/row order;
/// images use polynomial/column/matrix-row order, exactly as the CPU adapter.
///
/// `matrix` is the admitted q-uniform centered setup embedded in the proof
/// field; `challenges` are the caller's replayed binary root draws embedded
/// via the signed module map. `residuals` must be authenticated by the
/// committed-q-quotient or exact-lift projection lane. They are not a new
/// clear proof message and this constructor proves no divisibility claim.
pub fn binary_commitment_group<E: Field>(
    descriptor: BinarySourceDescriptor,
    matrix: &[Vec<[E; 648]>],
    challenges: &[[E; 648]],
    response_segment: usize,
    image_segment: usize,
    residuals: &[[E; 648]],
) -> Result<NativeRootGroup<E>, AkitaError> {
    let polynomials = descriptor.polynomials().len();
    let columns = descriptor.index_map().columns();
    let source_rows = checked::product([4, columns])
        .and_then(|divisor| checked::exact_div(descriptor.index_map().padded_carriers(), divisor))
        .filter(|&count| count > 0)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid binary source row geometry".into()))?;
    let rank = matrix.len();
    let challenge_count =
        checked::product([polynomials, columns]).ok_or(AkitaError::InvalidProof)?;
    let row_count = checked::product([polynomials, rank]).ok_or(AkitaError::InvalidProof)?;
    if rank == 0
        || matrix.iter().any(|row| row.len() != source_rows)
        || challenges.len() != challenge_count
        || residuals.len() != row_count
    {
        return Err(AkitaError::InvalidSetup(
            "binary commitment row dimensions disagree".into(),
        ));
    }
    let polynomial = RelationPolynomial::minus_trinomial(648)?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(row_count)
        .map_err(|_| AkitaError::InvalidSetup("binary row allocation failed".into()))?;
    for owner in 0..polynomials {
        for (matrix_index, matrix_row) in matrix.iter().enumerate() {
            let mut terms = Vec::new();
            let count = checked::sum([source_rows, columns]).ok_or(AkitaError::InvalidProof)?;
            terms
                .try_reserve_exact(count)
                .map_err(|_| AkitaError::InvalidSetup("binary term allocation failed".into()))?;
            for (column, coefficients) in matrix_row.iter().enumerate() {
                terms.push(NativeRootTerm {
                    segment: response_segment,
                    element: checked::mul_add(owner, source_rows, column)
                        .ok_or(AkitaError::InvalidProof)?,
                    multiplier: coefficients.to_vec(),
                });
            }
            for column in 0..columns {
                let challenge_index =
                    checked::mul_add(owner, columns, column).ok_or(AkitaError::InvalidProof)?;
                let challenge = challenges
                    .get(challenge_index)
                    .ok_or(AkitaError::InvalidProof)?;
                terms.push(NativeRootTerm {
                    segment: image_segment,
                    element: checked::mul_add(challenge_index, rank, matrix_index)
                        .ok_or(AkitaError::InvalidProof)?,
                    multiplier: challenge.iter().map(|&x| -x).collect(),
                });
            }
            let target_index =
                checked::mul_add(owner, rank, matrix_index).ok_or(AkitaError::InvalidProof)?;
            rows.push(NativeRootRow {
                polynomial,
                terms,
                target: residuals
                    .get(target_index)
                    .ok_or(AkitaError::InvalidProof)?
                    .to_vec(),
            });
        }
    }
    Ok(NativeRootGroup {
        kind: NativeRootGroupKind::Binary(descriptor),
        rows,
    })
}
