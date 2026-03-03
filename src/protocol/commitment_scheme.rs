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
use crate::protocol::greyhound::{greyhound_eval, greyhound_reduce};
use crate::protocol::labrador;
use crate::protocol::labrador::transcript::{
    absorb_greyhound_eval_claim, absorb_greyhound_eval_context, absorb_greyhound_u2,
    sample_greyhound_fold_challenge, GreyhoundEvalTranscriptContext,
};
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::prg::{MatrixPrgBackendChoice, MatrixPrgBackendId};
use crate::protocol::proof::{HachiCommitmentHint, HachiFoldProof, HachiProof};
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::{
    eval_ring_at, ring_switch_prover, ring_switch_verifier_metadata,
};
use crate::protocol::sumcheck::batched_sumcheck::{
    prove_batched_sumcheck, verify_batched_sumcheck_rounds,
};
use crate::protocol::sumcheck::eq_poly::EqPolynomial;
use crate::protocol::sumcheck::relation_sumcheck::RelationSumcheckProver;
use crate::protocol::sumcheck::{
    multilinear_eval, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
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
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
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
            s: w.s,
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
        let outer_point = &opening_point[alpha..];

        let layout = <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::layout(setup)?;
        let expected_outer = layout.r_vars + layout.m_vars;
        if outer_point.len() != expected_outer {
            return Err(HachiError::InvalidPointDimension {
                expected: expected_outer + alpha,
                actual: opening_point.len(),
            });
        }

        let ring_opening_point =
            ring_opening_point_from_field::<F>(outer_point, layout.r_vars, layout.m_vars)?;

        let y_ring = evaluate_packed_ring_poly::<F, { D }>(&hint.ring_coeffs, outer_point);

        // Fiat-Shamir: bind commitment, opening point, and y_ring before any challenges.
        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
        for pt in opening_point {
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
        let rs = ring_switch_prover::<F, T, { D }, Cfg>(&quad_eq, setup, transcript)?;

        // Batched sumcheck: relation only; final w_tilde(r) is handed off to Greyhound.
        let w_evals = rs.w_evals.clone();
        let mut relation_prover = RelationSumcheckProver::new(
            rs.w_evals,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            rs.num_u,
            rs.num_l,
        );

        let instances: Vec<&mut dyn SumcheckInstanceProver<F>> = vec![&mut relation_prover];
        let (sumcheck_proof, r_sumcheck) =
            prove_batched_sumcheck::<F, _, F, _>(instances, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;
        let w_eval = multilinear_eval(&w_evals, &r_sumcheck)?;

        let fold = HachiFoldProof {
            y_ring,
            v: quad_eq.v,
            sumcheck_proof,
            w_commitment: rs.w_commitment.clone(),
        };

        if !labrador_enabled::<D>() {
            return Ok(HachiProof {
                folds: vec![fold],
                greyhound_eval_proof: crate::protocol::greyhound::GreyhoundEvalProof::empty(),
                labrador_proof: labrador::LabradorProof::empty(),
            });
        }

        let comkey_seed = &setup.expanded.seed.public_matrix_seed;
        let jl_seed = derive_jl_seed(comkey_seed);
        let backend = matrix_backend_from_id(setup.expanded.seed.public_matrix_prg_backend);

        let (greyhound_eval_proof, labrador_witness, statement) = greyhound_eval(
            &rs.w,
            &r_sumcheck,
            w_eval,
            &rs.w_commitment.u,
            comkey_seed,
            backend,
            transcript,
        )?;
        let labrador_proof = labrador::prove_with_config(
            labrador_witness,
            &statement,
            &greyhound_eval_proof.config,
            comkey_seed,
            &jl_seed,
            backend,
            transcript,
        )?;

        Ok(HachiProof {
            folds: vec![fold],
            greyhound_eval_proof,
            labrador_proof,
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
        let fold = proof.folds.first().ok_or(HachiError::InvalidProof)?;
        if proof.folds.len() != 1 {
            return Err(HachiError::InvalidProof);
        }

        let alpha_bits = Cfg::D.trailing_zeros() as usize;
        if opening_point.len() < alpha_bits {
            return Err(HachiError::InvalidSetup(
                "opening point length underflow".to_string(),
            ));
        }
        let inner_point = &opening_point[..alpha_bits];
        let reduced_opening_point = &opening_point[alpha_bits..];

        // Fiat-Shamir: bind commitment, opening point, and y_ring before any challenges.
        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
        for pt in opening_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &fold.y_ring);

        // §3.1 trace check
        let v = reduce_inner_openings_to_ring_elements::<F, { D }>(inner_point)?;
        let d = F::from_u64(Cfg::D as u64);
        let trace_lhs = trace::<F, { D }>(&(fold.y_ring * v.sigma_m1()));
        let trace_rhs = d * *opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }

        // §4.2 Quadratic equation
        let layout = setup.expanded.seed.layout;
        let ring_opening_point = ring_opening_point_from_field::<F>(
            reduced_opening_point,
            layout.r_vars,
            layout.m_vars,
        )?;
        let quad_eq = QuadraticEquation::<F, { D }, Cfg>::new_verifier(
            setup,
            &ring_opening_point,
            &fold.v,
            transcript,
            commitment,
            &fold.y_ring,
        )?;

        // §4.3 Ring switch (verifier-side metadata only; hidden w is checked via Greyhound).
        let rs = ring_switch_verifier_metadata::<F, T, { D }, Cfg>(
            &quad_eq,
            &fold.w_commitment,
            transcript,
        )?;

        let relation_round_verifier = RelationRoundsVerifier {
            tau: rs.tau1.clone(),
            v: fold.v.clone(),
            u: commitment.u.clone(),
            y_ring: fold.y_ring,
            alpha: rs.alpha,
            num_u: rs.num_u,
            num_l: rs.num_l,
        };
        let verifiers: Vec<&dyn SumcheckInstanceVerifier<F>> = vec![&relation_round_verifier];
        let round_result = verify_batched_sumcheck_rounds::<F, _, F, _>(
            &fold.sumcheck_proof,
            verifiers,
            transcript,
            |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
        )?;
        let batching_coeff = round_result
            .batching_coeffs
            .first()
            .copied()
            .ok_or(HachiError::InvalidProof)?;
        if round_result.r_sumcheck.len() != rs.num_u + rs.num_l {
            return Err(HachiError::InvalidProof);
        }
        let (x_challenges, y_challenges) = round_result.r_sumcheck.split_at(rs.num_u);
        let alpha_val = multilinear_eval(&rs.alpha_evals_y, y_challenges)?;
        let m_val = multilinear_eval(&rs.m_evals_x, x_challenges)?;

        let denom = batching_coeff * alpha_val * m_val;
        let w_eval = if denom.is_zero() {
            if round_result.output_claim.is_zero() {
                F::zero()
            } else {
                return Err(HachiError::InvalidProof);
            }
        } else {
            let denom_inv = denom.inv().ok_or(HachiError::InvalidProof)?;
            round_result.output_claim * denom_inv
        };

        if labrador_enabled::<D>() {
            let comkey_seed = &setup.expanded.seed.public_matrix_seed;
            let jl_seed = derive_jl_seed(comkey_seed);
            let backend_v = matrix_backend_from_id(setup.expanded.seed.public_matrix_prg_backend);
            let gh_proof = &proof.greyhound_eval_proof;

            absorb_greyhound_eval_context(
                transcript,
                &GreyhoundEvalTranscriptContext {
                    m_rows: gh_proof.m_rows,
                    n_cols: gh_proof.n_cols,
                    inner_vars: gh_proof.inner_vars,
                    eval_point_len: round_result.r_sumcheck.len(),
                    prg_backend_id: backend_v as u8,
                },
            )?;
            absorb_greyhound_eval_claim(transcript, &round_result.r_sumcheck, &w_eval);
            absorb_greyhound_u2(transcript, &gh_proof.u2);
            let fold_challenges: Vec<F> = (0..gh_proof.n_cols)
                .map(|_| sample_greyhound_fold_challenge(transcript))
                .collect();

            let labrador_statement = greyhound_reduce(
                gh_proof,
                &fold.w_commitment.u,
                &round_result.r_sumcheck,
                w_eval,
                &fold_challenges,
                comkey_seed,
                backend_v,
            )?;
            labrador::verify(
                &labrador_statement,
                &proof.labrador_proof,
                comkey_seed,
                &jl_seed,
                backend_v,
                transcript,
            )?;
        }

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

/// Relation-sumcheck verifier state used only for round replay.
///
/// The final `w_tilde(r)` check is performed externally via Greyhound, so this
/// verifier intentionally does not implement `expected_output_claim`.
struct RelationRoundsVerifier<F: FieldCore, const D: usize> {
    tau: Vec<F>,
    v: Vec<CyclotomicRing<F, D>>,
    u: Vec<CyclotomicRing<F, D>>,
    y_ring: CyclotomicRing<F, D>,
    alpha: F,
    num_u: usize,
    num_l: usize,
}

impl<F: FieldCore, const D: usize> SumcheckInstanceVerifier<F> for RelationRoundsVerifier<F, D> {
    fn num_rounds(&self) -> usize {
        self.num_u + self.num_l
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        let y_a: Vec<F> = self
            .v
            .iter()
            .chain(self.u.iter())
            .chain(std::iter::once(&self.y_ring))
            .map(|r| eval_ring_at(r, &self.alpha))
            .collect();

        let eq_tau = EqPolynomial::evals(&self.tau);
        eq_tau.iter().enumerate().fold(F::zero(), |acc, (i, eq_i)| {
            let y_i = if i < y_a.len() { y_a[i] } else { F::zero() };
            acc + (*eq_i * y_i)
        })
    }

    fn expected_output_claim(&self, _challenges: &[F]) -> Result<F, HachiError> {
        Err(HachiError::InvalidInput(
            "relation expected output is externalized to Greyhound".to_string(),
        ))
    }
}

/// Greyhound/Labrador is enabled only for production ring degrees (D >= 64).
/// Test configs with small D skip the recursive proof layer.
const fn labrador_enabled<const D: usize>() -> bool {
    D >= 64
}

fn matrix_backend_from_id(id: MatrixPrgBackendId) -> MatrixPrgBackendChoice {
    match id {
        MatrixPrgBackendId::Shake256 => MatrixPrgBackendChoice::Shake256,
        MatrixPrgBackendId::Aes128Ctr => MatrixPrgBackendChoice::Aes128Ctr,
    }
}

fn derive_jl_seed(comkey_seed: &[u8; 32]) -> [u8; 16] {
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    use sha3::Shake256;

    let mut xof = Shake256::default();
    xof.update(b"hachi/labrador/jl-seed");
    xof.update(comkey_seed);
    let mut out = [0u8; 16];
    xof.finalize_xof().read(&mut out);
    out
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

        let s_all: Vec<Vec<CyclotomicRing<F, D>>> = chunks.iter().map(|c| c.s_i.clone()).collect();
        let t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> =
            chunks.iter().map(|c| c.t_hat_i.clone()).collect();
        let ring_coeffs: Vec<CyclotomicRing<F, D>> = chunks
            .iter()
            .flat_map(|c| c.block.iter().copied())
            .collect();

        let commitment = RingCommitment { u };
        let hint = HachiCommitmentHint {
            s: s_all,
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
#[allow(dead_code)]
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
    let fold = proof
        .folds
        .first()
        .ok_or_else(|| HachiError::InvalidInput("missing Hachi fold".to_string()))?;
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
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &fold.y_ring);

    let quad_eq = QuadraticEquation::<F, D, Cfg>::new_verifier(
        setup,
        &ring_opening_point,
        &fold.v,
        &mut transcript,
        commitment,
        &fold.y_ring,
    )?;
    transcript.append_serde(ABSORB_SUMCHECK_W, &fold.w_commitment);
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
    debug_assert_eq!(weights.len(), packed_coeffs.len());
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
