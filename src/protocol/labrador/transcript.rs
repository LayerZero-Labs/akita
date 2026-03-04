//! Canonical transcript schedule helpers for Greyhound/Labrador.
//!
//! These helpers centralize byte-level encoding for prover/verifier replay:
//! dimension binding, backend binding, and nonce encoding.

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
    /// Matrix-PRG backend id bound into Fiat-Shamir.
    pub prg_backend_id: u8,
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
    /// Input row chunk counts (`nu[i]` in the C reference).
    pub input_row_chunks: Vec<usize>,
    /// Witness decomposition parts.
    pub f: usize,
    /// Witness decomposition basis log2.
    pub b: usize,
    /// Commitment decomposition parts.
    pub fu: usize,
    /// Commitment decomposition basis log2.
    pub bu: usize,
    /// Inner commitment rank.
    pub kappa: usize,
    /// Outer commitment rank.
    pub kappa1: usize,
    /// Matrix-PRG backend id bound into Fiat-Shamir.
    pub prg_backend_id: u8,
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
    bytes.push(ctx.prg_backend_id);
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
    let mut bytes =
        Vec::with_capacity(4 + 8 * (8 + ctx.input_row_lengths.len() + ctx.input_row_chunks.len()));
    // Versioned payload for deterministic replay stability.
    bytes.push(1u8);
    bytes.push(u8::from(ctx.tail));
    bytes.push(ctx.prg_backend_id);
    bytes.push(0u8); // reserved
    append_u64_le(
        &mut bytes,
        checked_usize_to_u64(ctx.level_index, "level_index")?,
    );
    append_u64_le(&mut bytes, checked_usize_to_u64(ctx.f, "f")?);
    append_u64_le(&mut bytes, checked_usize_to_u64(ctx.b, "b")?);
    append_u64_le(&mut bytes, checked_usize_to_u64(ctx.fu, "fu")?);
    append_u64_le(&mut bytes, checked_usize_to_u64(ctx.bu, "bu")?);
    append_u64_le(&mut bytes, checked_usize_to_u64(ctx.kappa, "kappa")?);
    append_u64_le(&mut bytes, checked_usize_to_u64(ctx.kappa1, "kappa1")?);
    encode_usize_slice(&mut bytes, &ctx.input_row_lengths)?;
    encode_usize_slice(&mut bytes, &ctx.input_row_chunks)?;
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

/// Absorb canonical Greyhound evaluation claim bytes (`r` and `v`).
pub fn absorb_greyhound_eval_claim<F, T>(transcript: &mut T, eval_point: &[F], eval_value: &F)
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    for coord in eval_point {
        transcript.append_field(labels::ABSORB_GREYHOUND_EVAL_POINT, coord);
    }
    transcript.append_field(labels::ABSORB_GREYHOUND_EVAL_VALUE, eval_value);
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
pub fn absorb_labrador_level_context<F, T>(
    transcript: &mut T,
    ctx: &LabradorLevelTranscriptContext,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let bytes = encode_labrador_level_context(ctx)?;
    transcript.append_bytes(labels::ABSORB_LABRADOR_LEVEL_CONTEXT, &bytes);
    Ok(())
}

/// Absorb Labrador JL projection vector bytes (`i32` little-endian).
pub fn absorb_labrador_jl_projection<F, T>(transcript: &mut T, projection: &[i32; 256])
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let mut bytes = Vec::with_capacity(256 * std::mem::size_of::<i32>());
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

/// Sample a Labrador aggregation challenge.
pub fn sample_labrador_aggregation_challenge<F, T>(transcript: &mut T) -> F
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.challenge_scalar(labels::CHALLENGE_LABRADOR_AGGREGATION)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;

    // Fixed test nonces for deterministic replay.
    const TEST_NONCE_LOW: u64 = 1;
    const TEST_NONCE_HIGH: u64 = 2;

    #[test]
    fn greyhound_context_replay_is_deterministic() {
        let ctx = GreyhoundEvalTranscriptContext {
            m_rows: 64,
            n_cols: 128,
            inner_vars: 6,
            eval_point_len: 13,
            prg_backend_id: 1,
        };
        let eval_point: Vec<F> = (0..13).map(|i| F::from_u64((i + 3) as u64)).collect();
        let eval_value = F::from_u64(77);
        let u2 = vec![F::from_u64(9), F::from_u64(11), F::from_u64(13)];

        let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context::<F, _>(&mut t1, &ctx).unwrap();
        absorb_greyhound_eval_claim::<F, _>(&mut t1, &eval_point, &eval_value);
        absorb_greyhound_u2::<F, _, _>(&mut t1, &u2);
        let c1 = sample_greyhound_fold_challenge::<F, _>(&mut t1);

        let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context::<F, _>(&mut t2, &ctx).unwrap();
        absorb_greyhound_eval_claim::<F, _>(&mut t2, &eval_point, &eval_value);
        absorb_greyhound_u2::<F, _, _>(&mut t2, &u2);
        let c2 = sample_greyhound_fold_challenge::<F, _>(&mut t2);

        assert_eq!(c1, c2, "same transcript schedule must replay identically");
    }

    #[test]
    fn greyhound_context_binds_dimensions() {
        let eval_point: Vec<F> = (0..10).map(|i| F::from_u64((i + 5) as u64)).collect();
        let eval_value = F::from_u64(17);
        let u2 = vec![F::from_u64(1), F::from_u64(2)];

        let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context::<F, _>(
            &mut t1,
            &GreyhoundEvalTranscriptContext {
                m_rows: 32,
                n_cols: 32,
                inner_vars: 5,
                eval_point_len: 10,
                prg_backend_id: 1,
            },
        )
        .unwrap();
        absorb_greyhound_eval_claim::<F, _>(&mut t1, &eval_point, &eval_value);
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
                prg_backend_id: 1,
            },
        )
        .unwrap();
        absorb_greyhound_eval_claim::<F, _>(&mut t2, &eval_point, &eval_value);
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
            input_row_chunks: vec![16, 32, 4, 2],
            f: 2,
            b: 8,
            fu: 3,
            bu: 10,
            kappa: 12,
            kappa1: 6,
            prg_backend_id: 1,
        };
        let projection = std::array::from_fn(|i| i as i32 - 127);
        let nonce = 42u64;

        let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_LABRADOR_PROTOCOL);
        absorb_labrador_level_context::<F, _>(&mut t1, &ctx).unwrap();
        absorb_labrador_jl_projection::<F, _>(&mut t1, &projection);
        absorb_labrador_jl_nonce::<F, _>(&mut t1, nonce);
        let c1 = sample_labrador_aggregation_challenge::<F, _>(&mut t1);

        let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_LABRADOR_PROTOCOL);
        absorb_labrador_level_context::<F, _>(&mut t2, &ctx).unwrap();
        absorb_labrador_jl_projection::<F, _>(&mut t2, &projection);
        absorb_labrador_jl_nonce::<F, _>(&mut t2, nonce);
        let c2 = sample_labrador_aggregation_challenge::<F, _>(&mut t2);

        assert_eq!(c1, c2, "identical schedule must be replay deterministic");
    }

    #[test]
    fn labrador_nonce_binding_changes_challenge() {
        let ctx = LabradorLevelTranscriptContext {
            level_index: 0,
            tail: true,
            input_row_lengths: vec![64, 32],
            input_row_chunks: vec![2, 1],
            f: 1,
            b: 8,
            fu: 2,
            bu: 10,
            kappa: 4,
            kappa1: 0,
            prg_backend_id: 0,
        };
        let projection = [0i32; 256];

        let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_LABRADOR_PROTOCOL);
        absorb_labrador_level_context::<F, _>(&mut t1, &ctx).unwrap();
        absorb_labrador_jl_projection::<F, _>(&mut t1, &projection);
        absorb_labrador_jl_nonce::<F, _>(&mut t1, TEST_NONCE_LOW);
        let c1 = sample_labrador_aggregation_challenge::<F, _>(&mut t1);

        let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_LABRADOR_PROTOCOL);
        absorb_labrador_level_context::<F, _>(&mut t2, &ctx).unwrap();
        absorb_labrador_jl_projection::<F, _>(&mut t2, &projection);
        absorb_labrador_jl_nonce::<F, _>(&mut t2, TEST_NONCE_HIGH);
        let c2 = sample_labrador_aggregation_challenge::<F, _>(&mut t2);

        assert_ne!(c1, c2, "nonce must be transcript-binding");
    }
}
