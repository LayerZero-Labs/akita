//! Commitment scheme trait implementation.

use crate::algebra::ring::CyclotomicRing;
use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::commitment::onehot::{inner_ajtai_onehot, SparseBlockEntry};
use crate::protocol::commitment::utils::linear::{
    decompose_block, decompose_rows, mat_vec_mul_ntt_cached, MatrixSlot,
};
use crate::protocol::commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, HachiCommitmentCore, HachiProverSetup,
    HachiVerifierSetup, RingCommitment, RingCommitmentScheme, StreamingCommitmentScheme,
};
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::{HachiCommitmentHint, HachiProof, SumcheckAux};
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::{ring_switch_prover, ring_switch_verifier};
use crate::protocol::sumcheck::batched_sumcheck::{
    prove_batched_sumcheck, verify_batched_sumcheck,
};
use crate::protocol::sumcheck::norm_sumcheck::{NormSumcheckProver, NormSumcheckVerifier};
use crate::protocol::sumcheck::relation_sumcheck::{
    RelationSumcheckProver, RelationSumcheckVerifier,
};
use crate::protocol::sumcheck::{SumcheckInstanceProver, SumcheckInstanceVerifier};
use crate::protocol::transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, CHALLENGE_SUMCHECK_ROUND,
};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt, Polynomial};

#[cfg(test)]
use crate::protocol::ring_switch::{eval_ring_matrix_at, expand_m_a};
#[cfg(test)]
use crate::protocol::transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, DOMAIN_HACHI_PROTOCOL,
};
#[cfg(test)]
use crate::protocol::transcript::Blake2bTranscript;
#[cfg(test)]
use crate::protocol::SmallTestCommitmentConfig;

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: std::marker::PhantomData<Cfg>,
}

impl<F, const D: usize, Cfg> CommitmentScheme<F> for HachiCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    type ProverSetup = HachiProverSetup<F, D>;
    type VerifierSetup = HachiVerifierSetup<F, D>;
    type Commitment = RingCommitment<F, D>;
    type Proof = HachiProof<F, D>;
    type OpeningProofHint = HachiCommitmentHint<F, D>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::setup(max_num_vars)
                .expect("commitment setup failed");
        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        HachiVerifierSetup {
            expanded: setup.expanded.clone(),
        }
    }

    fn commit<P: Polynomial<F>>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::OpeningProofHint), HachiError> {
        let ring_coeffs =
            reduce_coeffs_to_ring_elements::<F, { D }>(poly.num_vars(), &poly.coeffs())?;
        let w = <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::commit_coeffs(
            &ring_coeffs,
            setup,
        )?;
        let hint = HachiCommitmentHint {
            t_hat: w.t_hat,
            ring_coeffs,
        };
        Ok((w.commitment, hint))
    }

    fn prove<T: Transcript<F>, P: Polynomial<F>>(
        setup: &Self::ProverSetup,
        poly: &P,
        opening_point: &[F],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut T,
        commitment: &Self::Commitment,
    ) -> Result<Self::Proof, HachiError> {
        let hint = hint.ok_or_else(|| {
            HachiError::InvalidInput("missing commitment hint for proving".to_string())
        })?;
        let _num_vars = poly.num_vars();
        let alpha = Cfg::D.trailing_zeros() as usize;
        if opening_point.len() < alpha {
            return Err(HachiError::InvalidPointDimension {
                expected: alpha,
                actual: opening_point.len(),
            });
        }

        let layout = <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::layout(setup)?;
        let target_num_vars = layout.m_vars + layout.r_vars + alpha;
        if opening_point.len() > target_num_vars {
            return Err(HachiError::InvalidPointDimension {
                expected: target_num_vars,
                actual: opening_point.len(),
            });
        }
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(target_num_vars, F::zero());
        let outer_point = &padded_point[alpha..];

        let ring_opening_point =
            ring_opening_point_from_field::<F>(outer_point, layout.r_vars, layout.m_vars)?;

        let y_ring = evaluate_packed_ring_poly::<F, { D }>(&hint.ring_coeffs, outer_point);

        // Fiat-Shamir: bind commitment, opening point, and y_ring before any challenges.
        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
        for pt in &padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        // §4.2 Quadratic equation
        let quad_eq = QuadraticEquation::<F, { D }, Cfg>::new_prover(
            setup,
            &ring_opening_point,
            &hint,
            transcript,
            commitment,
            &y_ring,
        )?;

        // §4.3 Ring switch
        let rs = ring_switch_prover::<F, T, { D }, Cfg>(&quad_eq, transcript)?;

        // Batched sumcheck: norm + relation
        let mut norm_prover = NormSumcheckProver::new(&rs.tau0, rs.w_evals.clone(), rs.b);
        let mut relation_prover = RelationSumcheckProver::new(
            rs.w_evals,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            rs.num_u,
            rs.num_l,
        );

        let instances: Vec<&mut dyn SumcheckInstanceProver<F>> =
            vec![&mut norm_prover, &mut relation_prover];
        let (sumcheck_proof, ..) =
            prove_batched_sumcheck::<F, _, F, _>(instances, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        Ok(HachiProof {
            v: quad_eq.v,
            y_ring,
            sumcheck_proof,
            sumcheck_aux: SumcheckAux { w: rs.w },
            w_commitment: rs.w_commitment,
        })
    }

    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[F],
        opening: &F,
        commitment: &Self::Commitment,
    ) -> Result<(), HachiError> {
        let alpha_bits = Cfg::D.trailing_zeros() as usize;
        if opening_point.len() < alpha_bits {
            return Err(HachiError::InvalidSetup(
                "opening point length underflow".to_string(),
            ));
        }
        let layout = setup.expanded.seed.layout;
        let target_num_vars = layout.m_vars + layout.r_vars + alpha_bits;
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(target_num_vars, F::zero());
        let inner_point = &padded_point[..alpha_bits];
        let reduced_opening_point = &padded_point[alpha_bits..];

        // Fiat-Shamir: bind commitment, opening point, and y_ring before any challenges.
        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
        for pt in &padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &proof.y_ring);

        // §3.1 trace check
        let v = reduce_inner_openings_to_ring_elements::<F, { D }>(inner_point)?;
        let d = F::from_u64(Cfg::D as u64);
        let trace_lhs = trace::<F, { D }>(&(proof.y_ring * v.sigma_m1()));
        let trace_rhs = d * *opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }

        // §4.2 Quadratic equation
        let ring_opening_point = ring_opening_point_from_field::<F>(
            reduced_opening_point,
            layout.r_vars,
            layout.m_vars,
        )?;
        let quad_eq = QuadraticEquation::<F, { D }, Cfg>::new_verifier(
            setup,
            &ring_opening_point,
            &proof.v,
            transcript,
            commitment,
            &proof.y_ring,
        )?;

        // §4.3 Ring switch (verifier side)
        let rs = ring_switch_verifier::<F, T, { D }, Cfg>(
            &quad_eq,
            &proof.sumcheck_aux.w,
            &proof.w_commitment,
            transcript,
        )?;

        // Batched sumcheck verification: norm (F_0) + relation (F_α)
        let norm_verifier = NormSumcheckVerifier::new(rs.tau0, rs.w_evals.clone(), rs.b);
        let relation_verifier = RelationSumcheckVerifier::new(
            rs.w_evals,
            rs.alpha_evals_y,
            rs.m_evals_x,
            rs.tau1,
            proof.v.clone(),
            commitment.u.clone(),
            proof.y_ring,
            rs.alpha,
            rs.num_u,
            rs.num_l,
        );

        let verifiers: Vec<&dyn SumcheckInstanceVerifier<F>> =
            vec![&norm_verifier, &relation_verifier];
        verify_batched_sumcheck::<F, _, F, _>(
            &proof.sumcheck_proof,
            verifiers,
            transcript,
            |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
        )?;

        Ok(())
    }

    fn combine_commitments(_commitments: &[Self::Commitment], _coeffs: &[F]) -> Self::Commitment {
        unimplemented!()
    }

    fn combine_hints(_hints: Vec<Self::OpeningProofHint>, _coeffs: &[F]) -> Self::OpeningProofHint {
        unimplemented!()
    }

    fn protocol_name() -> &'static [u8] {
        unimplemented!()
    }
}

/// Commit to a one-hot polynomial, returning both the commitment and a
/// complete `HachiCommitmentHint` (including `ring_coeffs` needed by `prove`).
///
/// # Errors
///
/// Returns an error if dimensions are inconsistent, any index is out of
/// range, or the underlying commitment routine fails.
pub fn commit_onehot<F, const D: usize, Cfg>(
    onehot_k: usize,
    indices: &[Option<usize>],
    setup: &HachiProverSetup<F, D>,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    let num_chunks = indices.len();
    let total_field_elems = num_chunks
        .checked_mul(onehot_k)
        .ok_or_else(|| HachiError::InvalidInput("T*K overflow".into()))?;
    if total_field_elems % D != 0 {
        return Err(HachiError::InvalidInput(format!(
            "T*K={total_field_elems} is not divisible by D={D}"
        )));
    }

    // Build ring_coeffs (needed for prove) from the sparse one-hot indices.
    let total_ring_elems = total_field_elems / D;
    let mut ring_coeffs = vec![CyclotomicRing::<F, D>::zero(); total_ring_elems];
    for (c, opt) in indices.iter().enumerate() {
        let Some(&idx) = opt.as_ref() else { continue };
        let field_pos = c * onehot_k + idx;
        let ring_idx = field_pos / D;
        let coeff_idx = field_pos % D;
        ring_coeffs[ring_idx].coeffs[coeff_idx] = F::one();
    }

    let w = <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::commit_onehot(
        onehot_k, indices, setup,
    )?;

    let hint = HachiCommitmentHint {
        t_hat: w.t_hat,
        ring_coeffs,
    };
    Ok((w.commitment, hint))
}

/// Per-block intermediate state for streaming Hachi commitment.
///
/// Each chunk corresponds to one Ajtai inner block: `D * block_len` field
/// elements packed into `block_len` ring elements, decomposed, and multiplied
/// by the inner matrix A.
#[derive(Clone, PartialEq, Eq)]
pub struct HachiChunkState<F: FieldCore, const D: usize> {
    /// Original ring elements for this block (needed for `ring_coeffs` hint).
    pub block: Vec<CyclotomicRing<F, D>>,
    /// Basis-decomposed input vector `s_i = G^{-1}(block)`.
    pub s_i: Vec<CyclotomicRing<F, D>>,
    /// Basis-decomposed inner Ajtai output `t̂_i = G^{-1}(A · s_i)`.
    pub t_hat_i: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore, const D: usize> std::fmt::Debug for HachiChunkState<F, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HachiChunkState")
            .field("block_len", &self.block.len())
            .field("s_i_len", &self.s_i.len())
            .field("t_hat_i_len", &self.t_hat_i.len())
            .finish()
    }
}

impl<F, const D: usize, Cfg> StreamingCommitmentScheme<F> for HachiCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    type ChunkState = HachiChunkState<F, D>;

    fn process_chunk(setup: &Self::ProverSetup, chunk: &[F]) -> Self::ChunkState {
        assert!(
            chunk.len() % D == 0,
            "chunk length {} is not divisible by D={}",
            chunk.len(),
            D
        );

        let block: Vec<CyclotomicRing<F, D>> = chunk
            .chunks_exact(D)
            .map(|c| CyclotomicRing::from_coefficients(std::array::from_fn(|j| c[j])))
            .collect();

        let s_i = decompose_block(&block, Cfg::DELTA, Cfg::LOG_BASIS);
        let t_i =
            mat_vec_mul_ntt_cached(setup.ntt_cache().expect("NTT cache"), MatrixSlot::A, &s_i)
                .expect("inner Ajtai");
        let t_hat_i = decompose_rows(&t_i, Cfg::DELTA, Cfg::LOG_BASIS);

        HachiChunkState {
            block,
            s_i,
            t_hat_i,
        }
    }

    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let layout = <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::layout(setup)
            .expect("layout");
        let block_len = layout.block_len;

        let num_field_elems = chunk.len() * onehot_k;
        assert!(
            num_field_elems % D == 0,
            "chunk cycles * K = {num_field_elems} is not divisible by D={D}",
        );

        // Build sparse entries and original block ring elements.
        let num_ring_elems = num_field_elems / D;
        let mut ring_block = vec![CyclotomicRing::<F, D>::zero(); num_ring_elems];
        let mut ring_elem_map: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (c, opt) in chunk.iter().enumerate() {
            if let Some(k) = opt {
                let field_pos = c * onehot_k + k;
                let ring_elem_idx = field_pos / D;
                let coeff_idx = field_pos % D;
                ring_block[ring_elem_idx].coeffs[coeff_idx] = F::one();
                ring_elem_map
                    .entry(ring_elem_idx)
                    .or_default()
                    .push(coeff_idx);
            }
        }

        let sparse_entries: Vec<SparseBlockEntry> = ring_elem_map
            .into_iter()
            .map(|(ring_elem_idx, nonzero_coeffs)| SparseBlockEntry {
                pos_in_block: ring_elem_idx,
                nonzero_coeffs,
            })
            .collect();

        let (t_i, s_i) =
            inner_ajtai_onehot(&setup.expanded.A, &sparse_entries, block_len, Cfg::DELTA);
        let t_hat_i = decompose_rows(&t_i, Cfg::DELTA, Cfg::LOG_BASIS);

        HachiChunkState {
            block: ring_block,
            s_i,
            t_hat_i,
        }
    }

    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let t_hat_flat: Vec<CyclotomicRing<F, D>> = chunks
            .iter()
            .flat_map(|c| c.t_hat_i.iter().copied())
            .collect();

        let u = mat_vec_mul_ntt_cached(
            setup.ntt_cache().expect("NTT cache"),
            MatrixSlot::B,
            &t_hat_flat,
        )
        .expect("outer Ajtai");

        let t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> =
            chunks.iter().map(|c| c.t_hat_i.clone()).collect();
        let ring_coeffs: Vec<CyclotomicRing<F, D>> = chunks
            .iter()
            .flat_map(|c| c.block.iter().copied())
            .collect();

        let commitment = RingCommitment { u };
        let hint = HachiCommitmentHint {
            t_hat: t_hat_all,
            ring_coeffs,
        };
        (commitment, hint)
    }
}

/// Re-derive the ring-switch challenge `alpha` and the expanded `M_a` vector
/// by replaying the transcript from the proof data and setup, exactly as the
/// verifier does.
#[cfg(test)]
pub(crate) fn rederive_alpha_and_m_a<F, const D: usize, Cfg>(
    proof: &HachiProof<F, D>,
    setup: &HachiVerifierSetup<F, D>,
    opening_point: &[F],
    commitment: &RingCommitment<F, D>,
) -> Result<(F, Vec<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + 'static,
    Cfg: CommitmentConfig,
{
    let alpha_bits = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let layout = setup.expanded.seed.layout;
    let ring_opening_point = ring_opening_point_from_field::<F>(
        &opening_point[alpha_bits..],
        layout.r_vars,
        layout.m_vars,
    )?;
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_HACHI_PROTOCOL);

    // Replay the same Fiat-Shamir absorptions the real verifier performs.
    commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
    for pt in opening_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &proof.y_ring);

    let quad_eq = QuadraticEquation::<F, D, Cfg>::new_verifier(
        setup,
        &ring_opening_point,
        &proof.v,
        &mut transcript,
        commitment,
        &proof.y_ring,
    )?;
    transcript.append_serde(ABSORB_SUMCHECK_W, &proof.w_commitment);
    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    let m_a = eval_ring_matrix_at::<F, D>(quad_eq.m(), &alpha);
    let m_a_vec = expand_m_a::<F, D, Cfg>(&m_a, alpha)?;
    Ok((alpha, m_a_vec))
}

fn lagrange_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    multilinear_lagrange_basis(&mut weights, point);
    weights
}

fn ring_opening_point_from_field<F: FieldCore>(
    opening_point: &[F],
    r_vars: usize,
    m_vars: usize,
) -> Result<RingOpeningPoint<F>, HachiError> {
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(HachiError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    // Sequential ordering: M variables (position in block) come first,
    // R variables (block selection) come second.
    let a = lagrange_weights(&opening_point[..m_vars]);
    let b = lagrange_weights(&opening_point[m_vars..]);
    Ok(RingOpeningPoint { a, b })
}

fn reduce_coeffs_to_ring_elements<F: FieldCore, const D: usize>(
    num_vars: usize,
    coeffs: &[F],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if D == 0 || !D.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "ring degree D={D} is not a power of two"
        )));
    }
    let alpha = D.trailing_zeros() as usize;
    if num_vars < alpha {
        return Err(HachiError::InvalidInput(format!(
            "num_vars {num_vars} is smaller than alpha {alpha}"
        )));
    }

    let expected_len = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| HachiError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
    if coeffs.len() != expected_len {
        return Err(HachiError::InvalidSize {
            expected: expected_len,
            actual: coeffs.len(),
        });
    }

    // Sequential packing: ring element i = coeffs[i*D .. (i+1)*D].
    // The first alpha variables (LSBs) become coefficient slots within each
    // ring element; the remaining outer_vars variables index ring elements.
    let outer_len = expected_len / D;
    let out: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..outer_len)
        .map(|i| {
            let ring_coeffs = std::array::from_fn(|j| coeffs[i * D + j]);
            CyclotomicRing::from_coefficients(ring_coeffs)
        })
        .collect();
    Ok(out)
}

fn reduce_inner_openings_to_ring_elements<F: FieldCore, const D: usize>(
    inner_point: &[F],
) -> Result<CyclotomicRing<F, D>, HachiError> {
    let weights = lagrange_weights(inner_point);
    if weights.len() != D {
        return Err(HachiError::InvalidInput(format!(
            "inner basis length {} does not match D={D}",
            weights.len()
        )));
    }
    let coeffs = std::array::from_fn(|i| weights[i]);
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

fn evaluate_packed_ring_poly<F: FieldCore, const D: usize>(
    packed_coeffs: &[CyclotomicRing<F, D>],
    outer_point: &[F],
) -> CyclotomicRing<F, D> {
    let weights = lagrange_weights(outer_point);
    debug_assert!(weights.len() >= packed_coeffs.len());
    #[cfg(feature = "parallel")]
    {
        packed_coeffs
            .par_iter()
            .zip(weights.par_iter())
            .fold(
                || CyclotomicRing::<F, D>::zero(),
                |acc, (f_i, w_i)| acc + f_i.scale(w_i),
            )
            .reduce(|| CyclotomicRing::<F, D>::zero(), |a, b| a + b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        packed_coeffs
            .iter()
            .zip(weights.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
                acc + f_i.scale(w_i)
            })
    }
}

fn trace<F: FieldCore + FromSmallInt, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
    let d = F::from_u64(D as u64);
    u.coefficients()[0] * d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::multilinear_evals::DenseMultilinearEvals;
    use crate::protocol::commitment::CommitmentConfig;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::F;
    use crate::{CommitmentScheme, FromSmallInt, Polynomial};

    type Cfg = SmallTestCommitmentConfig;
    type Scheme = HachiCommitmentScheme<{ Cfg::D }, Cfg>;

    #[test]
    fn verify_passes_for_consistent_opening() {
        let alpha = Cfg::D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;

        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = <Scheme as CommitmentScheme<F>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F>>::setup_verifier(&setup);

        let (commitment, hint) = <Scheme as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let opening = poly.evaluate(&opening_point);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentScheme<F>>::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentScheme<F>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_wrong_opening() {
        let alpha = Cfg::D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;

        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = <Scheme as CommitmentScheme<F>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F>>::setup_verifier(&setup);

        let (commitment, hint) = <Scheme as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let opening = poly.evaluate(&opening_point);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentScheme<F>>::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let wrong_opening = opening + F::one();
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentScheme<F>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &wrong_opening,
            &commitment,
        );

        assert!(
            result.is_err(),
            "verify must reject an incorrect opening value"
        );
    }

    #[test]
    fn streaming_commit_matches_non_streaming() {
        let alpha = Cfg::D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;

        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals.clone());

        let setup = <Scheme as CommitmentScheme<F>>::setup_prover(num_vars);

        // Non-streaming commit
        let (non_streaming_commitment, non_streaming_hint) =
            <Scheme as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();

        // Streaming commit: split field elements into chunks of D * block_len
        let chunk_size = Cfg::D * layout.block_len;
        let chunks: Vec<HachiChunkState<F, { Cfg::D }>> = evals
            .chunks_exact(chunk_size)
            .map(|chunk| <Scheme as StreamingCommitmentScheme<F>>::process_chunk(&setup, chunk))
            .collect();

        let (streaming_commitment, streaming_hint) =
            <Scheme as StreamingCommitmentScheme<F>>::aggregate_chunks(&setup, None, &chunks);

        assert_eq!(non_streaming_commitment, streaming_commitment);
        assert_eq!(non_streaming_hint, streaming_hint);
    }

    #[test]
    fn streaming_commit_then_prove_verify() {
        let alpha = Cfg::D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;

        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals.clone());

        let setup = <Scheme as CommitmentScheme<F>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F>>::setup_verifier(&setup);

        // Streaming commit
        let chunk_size = Cfg::D * layout.block_len;
        let chunks: Vec<HachiChunkState<F, { Cfg::D }>> = evals
            .chunks_exact(chunk_size)
            .map(|chunk| <Scheme as StreamingCommitmentScheme<F>>::process_chunk(&setup, chunk))
            .collect();
        let (commitment, hint) =
            <Scheme as StreamingCommitmentScheme<F>>::aggregate_chunks(&setup, None, &chunks);

        // Prove and verify with streaming-produced commitment + hint
        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let opening = poly.evaluate(&opening_point);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/stream");
        let proof = <Scheme as CommitmentScheme<F>>::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/stream");
        let result = <Scheme as CommitmentScheme<F>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
        );
        assert!(
            result.is_ok(),
            "streaming commit should produce valid proofs"
        );
    }
}
