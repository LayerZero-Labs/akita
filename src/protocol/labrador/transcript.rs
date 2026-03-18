//! Canonical transcript schedule helpers for Greyhound/Labrador.
//!
//! These helpers centralize byte-level encoding for prover/verifier replay:
//! dimension binding and nonce encoding.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::labrador::guardrails::checked_usize_to_u64;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, HachiSerialize};

/// Greyhound evaluation transcript context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GreyhoundEvalTranscriptContext {
    /// Matrix rows for reshaped witness.
    pub m_rows: usize,
    /// Matrix columns for reshaped witness.
    pub n_cols: usize,
    /// Number of "inner" multilinear variables.
    pub inner_vars: usize,
    /// Length of the evaluation point vector.
    pub eval_point_len: usize,
}

/// Labrador level transcript context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorLevelTranscriptContext {
    /// Zero-based recursion level index.
    pub level_index: usize,
    /// Whether this level is in tail mode.
    pub tail: bool,
    /// Input witness row lengths (`n[i]` in the C reference).
    pub input_row_lengths: Vec<usize>,
    /// Witness decomposition parts (formerly `f`).
    pub witness_digit_parts: usize,
    /// Witness decomposition basis log2 (formerly `b`).
    pub witness_digit_bits: usize,
    /// Auxiliary decomposition parts (formerly `fu`).
    pub aux_digit_parts: usize,
    /// Auxiliary decomposition basis log2 (formerly `bu`).
    pub aux_digit_bits: usize,
    /// Inner commitment rank (formerly `kappa`).
    pub inner_commit_rank: usize,
    /// Outer commitment rank (formerly `kappa1`).
    pub outer_commit_rank: usize,
}

fn append_u64_le(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn encode_usize_slice(buf: &mut Vec<u8>, values: &[usize]) -> Result<(), HachiError> {
    append_u64_le(buf, checked_usize_to_u64(values.len(), "slice length")?);
    for &v in values {
        append_u64_le(buf, checked_usize_to_u64(v, "slice element")?);
    }
    Ok(())
}

fn encode_greyhound_eval_context(
    ctx: &GreyhoundEvalTranscriptContext,
) -> Result<Vec<u8>, HachiError> {
    let mut bytes = Vec::with_capacity(2 + 8 * 4);
    // Versioned payload for deterministic replay stability.
    bytes.push(1u8);
    bytes.push(0u8); // backend id removed
    append_u64_le(&mut bytes, checked_usize_to_u64(ctx.m_rows, "m_rows")?);
    append_u64_le(&mut bytes, checked_usize_to_u64(ctx.n_cols, "n_cols")?);
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.inner_vars, "inner_vars")?,
    );
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.eval_point_len, "eval_point_len")?,
    );
    Ok(bytes)
}

fn encode_labrador_level_context(
    ctx: &LabradorLevelTranscriptContext,
) -> Result<Vec<u8>, HachiError> {
    let mut bytes = Vec::with_capacity(4 + 8 * (7 + ctx.input_row_lengths.len()));
    // Versioned payload for deterministic replay stability.
    bytes.push(1u8);
    bytes.push(u8::from(ctx.tail));
    bytes.push(0u8); // backend id removed
    bytes.push(0u8); // reserved
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.level_index, "level_index")?,
    );
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.witness_digit_parts, "witness_digit_parts")?,
    );
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.witness_digit_bits, "witness_digit_bits")?,
    );
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.aux_digit_parts, "aux_digit_parts")?,
    );
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.aux_digit_bits, "aux_digit_bits")?,
    );
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.inner_commit_rank, "inner_commit_rank")?,
    );
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.outer_commit_rank, "outer_commit_rank")?,
    );
    encode_usize_slice(&mut bytes, &ctx.input_row_lengths)?;
    Ok(bytes)
}

/// Absorb canonical Greyhound evaluation context bytes.
///
/// # Errors
///
/// Returns an error if any dimension does not fit in `u64`.
pub fn absorb_greyhound_eval_context<F, T>(
    transcript: &mut T,
    ctx: &GreyhoundEvalTranscriptContext,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let bytes = encode_greyhound_eval_context(ctx)?;
    transcript.append_bytes(labels::ABSORB_GREYHOUND_EVAL_CONTEXT, &bytes);
    Ok(())
}

/// Absorb canonical Greyhound evaluation claim bytes (`r` and ring-valued `v`).
///
/// Absorbs each coordinate of the evaluation point, then all D coefficients of
/// the ring-valued evaluation target.
pub fn absorb_greyhound_eval_claim<F, T, const D: usize>(
    transcript: &mut T,
    eval_point: &[F],
    eval_target: &CyclotomicRing<F, D>,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    for coord in eval_point {
        transcript.append_field(labels::ABSORB_GREYHOUND_EVAL_POINT, coord);
    }
    for coeff in eval_target.coefficients() {
        transcript.append_field(labels::ABSORB_GREYHOUND_EVAL_VALUE, coeff);
    }
}

/// Absorb Greyhound commitment payload `u2`.
pub fn absorb_greyhound_u2<F, T, S>(transcript: &mut T, u2: &S)
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    S: HachiSerialize,
{
    transcript.append_serde(labels::ABSORB_GREYHOUND_U2, u2);
}

/// Sample a Greyhound fold challenge.
pub fn sample_greyhound_fold_challenge<F, T>(transcript: &mut T) -> F
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.challenge_scalar(labels::CHALLENGE_GREYHOUND_FOLD)
}

/// Absorb canonical Labrador level context bytes.
///
/// # Errors
///
/// Returns an error if any dimension does not fit in `u64`.
#[tracing::instrument(skip_all, name = "labrador::absorb_level_context")]
pub fn absorb_labrador_level_context<F, T>(
    transcript: &mut T,
    ctx: &LabradorLevelTranscriptContext,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let bytes = encode_labrador_level_context(ctx)?;
    transcript.append_bytes(labels::ABSORB_LABRADOR_RECURSION_CONTEXT, &bytes);
    Ok(())
}

/// Absorb Labrador JL projection vector bytes (`i64` little-endian).
#[tracing::instrument(skip_all, name = "labrador::absorb_jl_projection")]
pub fn absorb_labrador_jl_projection<F, T>(transcript: &mut T, projection: &[i64; 256])
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let mut bytes = Vec::with_capacity(256 * std::mem::size_of::<i64>());
    for coeff in projection {
        bytes.extend_from_slice(&coeff.to_le_bytes());
    }
    transcript.append_bytes(labels::ABSORB_LABRADOR_JL_PROJECTION, &bytes);
}

/// Absorb Labrador JL nonce (`u64` little-endian).
pub fn absorb_labrador_jl_nonce<F, T>(transcript: &mut T, nonce: u64)
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(labels::ABSORB_LABRADOR_JL_NONCE, &nonce.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn scalar_ring(s: F) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(
            |i| {
                if i == 0 {
                    s
                } else {
                    F::zero()
                }
            },
        ))
    }

    // Fixed test nonces for deterministic replay.
    const TEST_NONCE_LOW: u64 = 1;
    const TEST_NONCE_HIGH: u64 = 2;
    const TEST_NONCE_REPLAY: u64 = 42;

    #[test]
    fn greyhound_context_replay_is_deterministic() {
        let ctx = GreyhoundEvalTranscriptContext {
            m_rows: 64,
            n_cols: 128,
            inner_vars: 6,
            eval_point_len: 13,
        };
        let eval_point: Vec<F> = (0..13).map(|i| F::from_u64((i + 3) as u64)).collect();
        let eval_target = scalar_ring(F::from_u64(77));
        let u2 = vec![F::from_u64(9), F::from_u64(11), F::from_u64(13)];

        let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context::<F, _>(&mut t1, &ctx).unwrap();
        absorb_greyhound_eval_claim::<F, _, D>(&mut t1, &eval_point, &eval_target);
        absorb_greyhound_u2::<F, _, _>(&mut t1, &u2);
        let c1 = sample_greyhound_fold_challenge::<F, _>(&mut t1);

        let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context::<F, _>(&mut t2, &ctx).unwrap();
        absorb_greyhound_eval_claim::<F, _, D>(&mut t2, &eval_point, &eval_target);
        absorb_greyhound_u2::<F, _, _>(&mut t2, &u2);
        let c2 = sample_greyhound_fold_challenge::<F, _>(&mut t2);

        assert_eq!(c1, c2, "same transcript schedule must replay identically");
    }

    #[test]
    fn greyhound_context_binds_dimensions() {
        let eval_point: Vec<F> = (0..10).map(|i| F::from_u64((i + 5) as u64)).collect();
        let eval_target = scalar_ring(F::from_u64(17));
        let u2 = vec![F::from_u64(1), F::from_u64(2)];

        let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context::<F, _>(
            &mut t1,
            &GreyhoundEvalTranscriptContext {
                m_rows: 32,
                n_cols: 32,
                inner_vars: 5,
                eval_point_len: 10,
            },
        )
        .unwrap();
        absorb_greyhound_eval_claim::<F, _, D>(&mut t1, &eval_point, &eval_target);
        absorb_greyhound_u2::<F, _, _>(&mut t1, &u2);
        let c1 = sample_greyhound_fold_challenge::<F, _>(&mut t1);

        let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context::<F, _>(
            &mut t2,
            &GreyhoundEvalTranscriptContext {
                m_rows: 32,
                n_cols: 64, // dimension changed
                inner_vars: 5,
                eval_point_len: 10,
            },
        )
        .unwrap();
        absorb_greyhound_eval_claim::<F, _, D>(&mut t2, &eval_point, &eval_target);
        absorb_greyhound_u2::<F, _, _>(&mut t2, &u2);
        let c2 = sample_greyhound_fold_challenge::<F, _>(&mut t2);

        assert_ne!(
            c1, c2,
            "dimension changes must affect transcript challenges"
        );
    }

    #[test]
    fn labrador_context_and_nonce_replay_is_deterministic() {
        let ctx = LabradorLevelTranscriptContext {
            level_index: 2,
            tail: false,
            input_row_lengths: vec![1024, 2048, 128, 64],
            witness_digit_parts: 2,
            witness_digit_bits: 8,
            aux_digit_parts: 3,
            aux_digit_bits: 10,
            inner_commit_rank: 12,
            outer_commit_rank: 6,
        };
        let projection = std::array::from_fn(|i| i as i64 - 127);
        let nonce = TEST_NONCE_REPLAY;

        let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_LABRADOR_RECURSION);
        absorb_labrador_level_context::<F, _>(&mut t1, &ctx).unwrap();
        absorb_labrador_jl_projection::<F, _>(&mut t1, &projection);
        absorb_labrador_jl_nonce::<F, _>(&mut t1, nonce);
        let c1 = t1.challenge_scalar(labels::CHALLENGE_LABRADOR_AGGREGATION);

        let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_LABRADOR_RECURSION);
        absorb_labrador_level_context::<F, _>(&mut t2, &ctx).unwrap();
        absorb_labrador_jl_projection::<F, _>(&mut t2, &projection);
        absorb_labrador_jl_nonce::<F, _>(&mut t2, nonce);
        let c2 = t2.challenge_scalar(labels::CHALLENGE_LABRADOR_AGGREGATION);

        assert_eq!(c1, c2, "identical schedule must be replay deterministic");
    }

    #[test]
    fn labrador_nonce_binding_changes_challenge() {
        let ctx = LabradorLevelTranscriptContext {
            level_index: 0,
            tail: true,
            input_row_lengths: vec![64, 32],
            witness_digit_parts: 1,
            witness_digit_bits: 8,
            aux_digit_parts: 2,
            aux_digit_bits: 10,
            inner_commit_rank: 4,
            outer_commit_rank: 0,
        };
        let projection = [0i64; 256];

        let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_LABRADOR_RECURSION);
        absorb_labrador_level_context::<F, _>(&mut t1, &ctx).unwrap();
        absorb_labrador_jl_projection::<F, _>(&mut t1, &projection);
        absorb_labrador_jl_nonce::<F, _>(&mut t1, TEST_NONCE_LOW);
        let c1 = t1.challenge_scalar(labels::CHALLENGE_LABRADOR_AGGREGATION);

        let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_LABRADOR_RECURSION);
        absorb_labrador_level_context::<F, _>(&mut t2, &ctx).unwrap();
        absorb_labrador_jl_projection::<F, _>(&mut t2, &projection);
        absorb_labrador_jl_nonce::<F, _>(&mut t2, TEST_NONCE_HIGH);
        let c2 = t2.challenge_scalar(labels::CHALLENGE_LABRADOR_AGGREGATION);

        assert_ne!(c1, c2, "nonce must be transcript-binding");
    }
}
