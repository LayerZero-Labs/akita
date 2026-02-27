//! Commitment scheme trait implementation.

use crate::algebra::ring::{CyclotomicRing, SparseChallengeConfig};
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::challenges::sparse::sample_dense_challenges;
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::commitment::{
    CommitmentConfig, CommitmentScheme, DefaultCommitmentConfig, HachiCommitmentCore,
    RingCommitment, RingCommitmentScheme, RingCommitmentSetup,
};
use crate::protocol::iteration_prover::HachiProver;
use crate::protocol::proof::{HachiProof, SumcheckAux};
use crate::protocol::sumcheck::hachi_sumcheck::{
    eq_evals, F0Prover, F0Verifier, FAlphaProver, FAlphaVerifier,
};
use crate::protocol::sumcheck::{
    prove_sumcheck, verify_sumcheck, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
use crate::protocol::transcript::labels::{
    ABSORB_PROVER_V, ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_STAGE1_FOLD,
    CHALLENGE_SUMCHECK_ROUND, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, Polynomial};

/// Prover-side hint produced at commitment time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiCommitmentHint<F: FieldCore, const D: usize> {
    /// Decomposed `s_i` blocks from the commitment phase.
    pub s: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed `t̂_i` blocks from the commitment phase.
    pub t_hat: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Ring coefficients from the §3.1 reduction (evaluation table).
    pub ring_coeffs: Vec<CyclotomicRing<F, D>>,
}

/// Placeholder for the end-to-end PCS wrapper.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme;

impl<F> CommitmentScheme<F> for HachiCommitmentScheme
where
    F: FieldCore + CanonicalField + FieldSampling,
{
    type ProverSetup = RingCommitmentSetup<F, { DefaultCommitmentConfig::D }>;
    type VerifierSetup = RingCommitmentSetup<F, { DefaultCommitmentConfig::D }>;
    type Commitment = RingCommitment<F, { DefaultCommitmentConfig::D }>;
    type Proof = HachiProof<F, { DefaultCommitmentConfig::D }>;
    type OpeningProofHint = HachiCommitmentHint<F, { DefaultCommitmentConfig::D }>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let (setup, _) = <HachiCommitmentCore as RingCommitmentScheme<
            F,
            { DefaultCommitmentConfig::D },
            DefaultCommitmentConfig,
        >>::setup(max_num_vars)
        .expect("commitment setup failed");
        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.clone()
    }

    fn commit<P: Polynomial<F>>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::OpeningProofHint), HachiError> {
        let num_vars = poly.num_vars();
        let coeffs = poly.coeffs();
        // Section 3.1 (Reducing to multilinear evaluation over Rq)
        let ring_coeffs =
            reduce_coeffs_to_ring_elements::<F, { DefaultCommitmentConfig::D }>(num_vars, &coeffs)?;
        let (commitment, s, t_hat) = <HachiCommitmentCore as RingCommitmentScheme<
            F,
            { DefaultCommitmentConfig::D },
            DefaultCommitmentConfig,
        >>::commit_coeffs(&ring_coeffs, setup)?;
        let hint = HachiCommitmentHint {
            s,
            t_hat,
            ring_coeffs,
        };
        Ok((commitment, hint))
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
        let num_vars = poly.num_vars();
        let alpha = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let reduced_len = num_vars
            .checked_sub(alpha)
            .ok_or_else(|| HachiError::InvalidSetup("reduction length underflow".to_string()))?;
        if opening_point.len() < reduced_len {
            return Err(HachiError::InvalidPointDimension {
                expected: reduced_len,
                actual: opening_point.len(),
            });
        }

        let ring_opening_point = ring_opening_point_from_field::<F, { DefaultCommitmentConfig::D }>(
            &opening_point[..reduced_len],
            DefaultCommitmentConfig::R,
            DefaultCommitmentConfig::M,
        )?;

        let y_ring = evaluate_packed_ring_poly::<F, { DefaultCommitmentConfig::D }>(
            &hint.ring_coeffs,
            &opening_point[..reduced_len],
        );

        let mut prover = HachiProver::<F, { DefaultCommitmentConfig::D }>::new();
        let (v, challenges) = prover.prove_stage1::<T, DefaultCommitmentConfig>(
            setup,
            &ring_opening_point,
            transcript,
            &hint,
        )?;
        let m = generate_m::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(
            setup,
            &ring_opening_point,
            &challenges,
        )?;
        let z = generate_z(&prover.w_hat, &hint.t_hat, &prover.z_hat);
        let y = generate_y::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(
            &v,
            &commitment.u,
            &y_ring,
        )?;

        // Compute r via poly division: r[i] = (M[i]·z − y[i]) / (X^D + 1)
        let r = compute_r_via_poly_division::<F, { DefaultCommitmentConfig::D }>(&m, &z, &y);
        let w: Vec<F> =
            build_w_coeffs::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(&z, &r);

        // Commit to w and absorb the commitment into the transcript.
        let w_commitment = commit_w::<F>(&w)?;
        transcript.append_serde(ABSORB_SUMCHECK_W, &w_commitment);

        let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

        let m_a = eval_ring_matrix_at::<F, { DefaultCommitmentConfig::D }>(&m, &alpha);
        let m_a_vec =
            expand_m_a::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(&m_a, alpha)?;
        let m_cols = m_a_vec.len() / m_a.len();

        // --- §4.3 Sumcheck integration ---

        let d: usize = DefaultCommitmentConfig::D;
        let rows = m_row_count::<DefaultCommitmentConfig>();
        let (w_evals, num_u, num_l) = build_w_evals(&w, d);
        let num_sc_vars = num_u + num_l;
        let b = 1usize << DefaultCommitmentConfig::LOG_BASIS;

        // F_0 sumcheck (range check)
        let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
        let mut f0_prover = F0Prover::new(&tau0, w_evals.clone(), b);
        let (f0_proof, _f0_challenges, _f0_final) =
            prove_sumcheck::<F, _, F, _, _>(&mut f0_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        // F_alpha sumcheck (evaluation relation)
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
        let alpha_evals_y = build_alpha_evals_y(alpha, d);
        let m_evals_x = build_m_evals_x(&m_a_vec, rows, m_cols, &tau1);
        let mut f_alpha_prover =
            FAlphaProver::new(w_evals, &alpha_evals_y, &m_evals_x, num_u, num_l);
        let (f_alpha_proof, _fa_challenges, _fa_final) =
            prove_sumcheck::<F, _, F, _, _>(&mut f_alpha_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        Ok(HachiProof {
            v,
            y_ring,
            f0_proof,
            f_alpha_proof,
            sumcheck_aux: SumcheckAux { w },
            w_commitment,
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
        let alpha_bits = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let reduced_len = opening_point.len().checked_sub(alpha_bits).ok_or_else(|| {
            HachiError::InvalidSetup("opening point length underflow".to_string())
        })?;
        let reduced_opening_point = &opening_point[..reduced_len];
        let inner_point = &opening_point[reduced_len..];

        // §3.1 trace check
        let v = reduce_inner_openings_to_ring_elements::<F, { DefaultCommitmentConfig::D }>(
            inner_point,
        )?;
        let d = F::from_u64(DefaultCommitmentConfig::D as u64);
        let trace_lhs = trace::<F, { DefaultCommitmentConfig::D }>(&(proof.y_ring * v.sigma_m1()));
        let trace_rhs = d * *opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }

        // Derive stage-1 challenges and ring-switch challenge (same as prover)
        let ring_opening_point = ring_opening_point_from_field::<F, { DefaultCommitmentConfig::D }>(
            reduced_opening_point,
            DefaultCommitmentConfig::R,
            DefaultCommitmentConfig::M,
        )?;
        let challenges = derive_stage1_challenges::<
            F,
            T,
            { DefaultCommitmentConfig::D },
            DefaultCommitmentConfig,
        >(transcript, &proof.v)?;
        let m = generate_m::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(
            setup,
            &ring_opening_point,
            &challenges,
        )?;
        transcript.append_serde(ABSORB_SUMCHECK_W, &proof.w_commitment);

        let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

        let m_a = eval_ring_matrix_at::<F, { DefaultCommitmentConfig::D }>(&m, &alpha);
        let m_a_vec =
            expand_m_a::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(&m_a, alpha)?;

        // --- §4.3 Sumcheck verification ---

        let d = DefaultCommitmentConfig::D;
        let rows = m_row_count::<DefaultCommitmentConfig>();
        let (w_evals, num_u, num_l) = build_w_evals(&proof.sumcheck_aux.w, d);
        let num_sc_vars = num_u + num_l;
        let b = 1usize << DefaultCommitmentConfig::LOG_BASIS;

        // F_0 sumcheck verification (range check)
        let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
        let f0_verifier = F0Verifier::new(tau0, w_evals.clone(), b);
        let _f0_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof.f0_proof, &f0_verifier, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        // F_alpha sumcheck verification (evaluation relation)
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
        let cols = m_a_vec.len() / rows.max(1);
        let alpha_evals_y = build_alpha_evals_y(alpha, d);
        let m_evals_x = build_m_evals_x(&m_a_vec, rows, cols, &tau1);

        let f_alpha_verifier = FAlphaVerifier::new(
            w_evals,
            alpha_evals_y,
            m_evals_x,
            tau1,
            proof.v.clone(),
            commitment.u.clone(),
            proof.y_ring,
            alpha,
            num_u,
            num_l,
        );
        let _fa_challenges = verify_sumcheck::<F, _, F, _, _>(
            &proof.f_alpha_proof,
            &f_alpha_verifier,
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

fn derive_stage1_challenges<F, T, const D: usize, Cfg: CommitmentConfig>(
    transcript: &mut T,
    v: &Vec<CyclotomicRing<F, D>>,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let challenge_cfg = SparseChallengeConfig {
        weight: Cfg::CHALLENGE_WEIGHT,
        nonzero_coeffs: vec![-1, 1],
    };
    let num_blocks = 1usize
        .checked_shl(Cfg::R as u32)
        .ok_or_else(|| HachiError::InvalidSetup("2^R does not fit usize".to_string()))?;
    transcript.append_serde(ABSORB_PROVER_V, v);
    sample_dense_challenges::<F, T, D>(
        transcript,
        CHALLENGE_STAGE1_FOLD,
        num_blocks,
        &challenge_cfg,
    )
}

/// Re-derive the ring-switch challenge `alpha` and the expanded `M_a` vector
/// by replaying the transcript from the proof data and setup, exactly as the
/// verifier does.
pub(crate) fn rederive_alpha_and_m_a<F>(
    proof: &HachiProof<F, { DefaultCommitmentConfig::D }>,
    setup: &<HachiCommitmentScheme as CommitmentScheme<F>>::ProverSetup,
    opening_point: &[F],
    commitment: &<HachiCommitmentScheme as CommitmentScheme<F>>::Commitment,
) -> Result<(F, Vec<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + 'static,
{
    let alpha_bits = DefaultCommitmentConfig::D.trailing_zeros() as usize;
    let reduced_len = opening_point
        .len()
        .checked_sub(alpha_bits)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length underflow".to_string()))?;
    let ring_opening_point = ring_opening_point_from_field::<F, { DefaultCommitmentConfig::D }>(
        &opening_point[..reduced_len],
        DefaultCommitmentConfig::R,
        DefaultCommitmentConfig::M,
    )?;
    let mut transcript = crate::protocol::transcript::Blake2bTranscript::<F>::new(
        crate::protocol::transcript::labels::DOMAIN_HACHI_PROTOCOL,
    );
    let challenges =
        derive_stage1_challenges::<F, _, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(
            &mut transcript,
            &proof.v,
        )?;
    let m = generate_m::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(
        setup,
        &ring_opening_point,
        &challenges,
    )?;
    transcript.append_serde(
        crate::protocol::transcript::labels::ABSORB_SUMCHECK_W,
        &proof.w_commitment,
    );
    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    let m_a = eval_ring_matrix_at::<F, { DefaultCommitmentConfig::D }>(&m, &alpha);
    let m_a_vec =
        expand_m_a::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(&m_a, alpha)?;
    Ok((alpha, m_a_vec))
}

fn constant_ring<F: FieldCore, const D: usize>(value: F) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = value;
    CyclotomicRing::from_coefficients(coeffs)
}

/// Commitment config sized for the sumcheck witness `w`.
///
/// `w` has more ring elements than `DefaultCommitmentConfig` can handle,
/// so we use a larger M (giving a bigger block length) while keeping
/// everything else the same.
#[derive(Clone, Copy, Debug)]
struct WCommitmentConfig;

impl CommitmentConfig for WCommitmentConfig {
    const D: usize = DefaultCommitmentConfig::D;
    const M: usize = 11;
    const R: usize = 0;
    const N_A: usize = DefaultCommitmentConfig::N_A;
    const N_B: usize = DefaultCommitmentConfig::N_B;
    const N_D: usize = DefaultCommitmentConfig::N_D;
    const LOG_BASIS: u32 = DefaultCommitmentConfig::LOG_BASIS;
    const DELTA: usize = DefaultCommitmentConfig::DELTA;
    const TAU: usize = DefaultCommitmentConfig::TAU;
    const BETA: u128 = DefaultCommitmentConfig::BETA;
    const CHALLENGE_WEIGHT: usize = DefaultCommitmentConfig::CHALLENGE_WEIGHT;
}

/// Commit to the sumcheck witness `w` (a flat vector of field elements)
/// using the Hachi ring commitment scheme with [`WCommitmentConfig`].
fn commit_w<F>(w: &[F]) -> Result<RingCommitment<F, { DefaultCommitmentConfig::D }>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
{
    let d = DefaultCommitmentConfig::D;
    let block_len = 1usize << WCommitmentConfig::M;

    let ring_elems: Vec<CyclotomicRing<F, { DefaultCommitmentConfig::D }>> = w
        .chunks(d)
        .map(|chunk| {
            let coeffs: [F; DefaultCommitmentConfig::D] =
                std::array::from_fn(|i| if i < chunk.len() { chunk[i] } else { F::zero() });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();

    let mut padded = ring_elems;
    padded.resize(
        block_len,
        CyclotomicRing::<F, { DefaultCommitmentConfig::D }>::zero(),
    );

    let blocks = vec![padded];

    let (w_setup, _) = <HachiCommitmentCore as RingCommitmentScheme<
        F,
        { DefaultCommitmentConfig::D },
        WCommitmentConfig,
    >>::setup(WCommitmentConfig::M)?;

    let (commitment, _, _) = <HachiCommitmentCore as RingCommitmentScheme<
        F,
        { DefaultCommitmentConfig::D },
        WCommitmentConfig,
    >>::commit_ring_blocks(&blocks, &w_setup)?;

    Ok(commitment)
}

fn gadget_row<F: FieldCore + CanonicalField, const D: usize>(
    levels: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        out.push(constant_ring::<F, D>(power));
        power = power * base;
    }
    out
}

fn kron_row<F: FieldCore, const D: usize>(
    left: &[CyclotomicRing<F, D>],
    right: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(left.len().saturating_mul(right.len()));
    for l in left {
        for r in right {
            out.push(*l * *r);
        }
    }
    out
}

fn gadget_block_diag<F: FieldCore, const D: usize>(
    blocks: usize,
    row: &[CyclotomicRing<F, D>],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let row_len = row.len();
    let mut rows = Vec::with_capacity(blocks);
    for i in 0..blocks {
        let mut out = vec![CyclotomicRing::<F, D>::zero(); blocks * row_len];
        let start = i * row_len;
        out[start..start + row_len].copy_from_slice(row);
        rows.push(out);
    }
    rows
}

pub(crate) fn generate_m<F, const D: usize, Cfg: CommitmentConfig>(
    setup: &RingCommitmentSetup<F, D>,
    opening_point: &crate::protocol::opening_point::RingOpeningPoint<F, D>,
    challenges: &[CyclotomicRing<F, D>],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError>
where
    F: FieldCore + CanonicalField,
{
    let num_blocks = 1usize
        .checked_shl(Cfg::R as u32)
        .ok_or_else(|| HachiError::InvalidSetup("2^R does not fit usize".to_string()))?;
    let block_len = 1usize
        .checked_shl(Cfg::M as u32)
        .ok_or_else(|| HachiError::InvalidSetup("2^M does not fit usize".to_string()))?;
    let w_len = Cfg::DELTA
        .checked_mul(num_blocks)
        .ok_or_else(|| HachiError::InvalidSetup("w length overflow".to_string()))?;
    let t_len = Cfg::DELTA
        .checked_mul(Cfg::N_A)
        .and_then(|v| v.checked_mul(num_blocks))
        .ok_or_else(|| HachiError::InvalidSetup("t length overflow".to_string()))?;
    let z_len = Cfg::TAU
        .checked_mul(Cfg::DELTA)
        .and_then(|v| v.checked_mul(block_len))
        .ok_or_else(|| HachiError::InvalidSetup("z length overflow".to_string()))?;
    let total_cols = w_len
        .checked_add(t_len)
        .and_then(|v| v.checked_add(z_len))
        .ok_or_else(|| HachiError::InvalidSetup("matrix width overflow".to_string()))?;

    if opening_point.b.len() != num_blocks {
        return Err(HachiError::InvalidPointDimension {
            expected: num_blocks,
            actual: opening_point.b.len(),
        });
    }
    if opening_point.a.len() != block_len {
        return Err(HachiError::InvalidPointDimension {
            expected: block_len,
            actual: opening_point.a.len(),
        });
    }
    if challenges.len() != num_blocks {
        return Err(HachiError::InvalidSize {
            expected: num_blocks,
            actual: challenges.len(),
        });
    }
    if setup.D.len() != Cfg::N_D {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_D,
            actual: setup.D.len(),
        });
    }
    if setup.B.len() != Cfg::N_B {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_B,
            actual: setup.B.len(),
        });
    }
    if setup.A.len() != Cfg::N_A {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_A,
            actual: setup.A.len(),
        });
    }
    if setup.A.first().map(|row| row.len()) != Some(block_len * Cfg::DELTA) {
        return Err(HachiError::InvalidSetup("A row width mismatch".to_string()));
    }

    let g1 = gadget_row::<F, D>(Cfg::DELTA, Cfg::LOG_BASIS);
    let j1 = gadget_row::<F, D>(Cfg::TAU, Cfg::LOG_BASIS);

    let row3_w = kron_row(&opening_point.b, &g1);
    let row4_w = kron_row(challenges, &g1);
    let row4_z = kron_row(&kron_row(&opening_point.a, &g1), &j1)
        .into_iter()
        .map(|x| -x)
        .collect::<Vec<_>>();

    let g_na = gadget_block_diag::<F, D>(Cfg::N_A, &g1);
    let row5_mid = g_na
        .iter()
        .map(|row| kron_row(challenges, row))
        .collect::<Vec<_>>();
    let row5_right = setup
        .A
        .iter()
        .map(|row| kron_row(row, &j1).into_iter().map(|x| -x).collect())
        .collect::<Vec<Vec<_>>>();

    let zero = CyclotomicRing::<F, D>::zero();
    let mut rows = Vec::with_capacity(Cfg::N_D + Cfg::N_B + 1usize + 1usize + Cfg::N_A);

    for row in setup.D.iter() {
        if row.len() != w_len {
            return Err(HachiError::InvalidSetup("D row width mismatch".to_string()));
        }
        let mut full = vec![zero; total_cols];
        full[..w_len].copy_from_slice(row);
        rows.push(full);
    }

    for row in setup.B.iter() {
        if row.len() != t_len {
            return Err(HachiError::InvalidSetup("B row width mismatch".to_string()));
        }
        let mut full = vec![zero; total_cols];
        full[w_len..w_len + t_len].copy_from_slice(row);
        rows.push(full);
    }

    let mut row3 = vec![zero; total_cols];
    row3[..w_len].copy_from_slice(&row3_w);
    rows.push(row3);

    let mut row4 = vec![zero; total_cols];
    row4[..w_len].copy_from_slice(&row4_w);
    row4[w_len + t_len..].copy_from_slice(&row4_z);
    rows.push(row4);

    for (mid, right) in row5_mid.into_iter().zip(row5_right.into_iter()) {
        let mut row = vec![zero; total_cols];
        row[w_len..w_len + t_len].copy_from_slice(&mid);
        row[w_len + t_len..].copy_from_slice(&right);
        rows.push(row);
    }

    Ok(rows)
}

pub(crate) fn generate_z<F: FieldCore, const D: usize>(
    w_hat: &[Vec<CyclotomicRing<F, D>>],
    t_hat: &[Vec<CyclotomicRing<F, D>>],
    z_hat: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(
        w_hat.len()
            + t_hat.len()
            + z_hat.len()
            + w_hat.iter().map(|v| v.len()).sum::<usize>()
            + t_hat.iter().map(|v| v.len()).sum::<usize>(),
    );
    for w in w_hat {
        out.extend(w.iter().copied());
    }
    for t in t_hat {
        out.extend(t.iter().copied());
    }
    out.extend_from_slice(z_hat);
    out
}

#[allow(dead_code)]
pub(crate) fn generate_y<F, const D: usize, Cfg: CommitmentConfig>(
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    u_eval: &CyclotomicRing<F, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore,
{
    if v.len() != Cfg::N_D {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_D,
            actual: v.len(),
        });
    }
    if u.len() != Cfg::N_B {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_B,
            actual: u.len(),
        });
    }
    let mut out = Vec::with_capacity(Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A);
    out.extend_from_slice(v);
    out.extend_from_slice(u);
    out.push(*u_eval);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend(std::iter::repeat(CyclotomicRing::<F, D>::zero()).take(Cfg::N_A));
    Ok(out)
}

/// Build the Lagrange basis weights `(χ_j(point))_j` in LSB-first order.
fn lagrange_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    multilinear_lagrange_basis(&mut weights, point);
    weights
}

/// Convert a field point into a ring opening point `(a, b)` using constant embedding.
fn ring_opening_point_from_field<F: FieldCore, const D: usize>(
    opening_point: &[F],
    r_vars: usize,
    m_vars: usize,
) -> Result<crate::protocol::opening_point::RingOpeningPoint<F, D>, HachiError> {
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(HachiError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let b = lagrange_vector_from_field::<F, D>(&opening_point[..r_vars]);
    let a = lagrange_vector_from_field::<F, D>(&opening_point[r_vars..]);
    Ok(crate::protocol::opening_point::RingOpeningPoint { a, b })
}

fn lagrange_vector_from_field<F: FieldCore, const D: usize>(
    point: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    lagrange_weights(point)
        .into_iter()
        .map(constant_ring::<F, D>)
        .collect()
}

/// Reduce coefficient blocks into ring elements.
///
/// Note: this implementation assumes `k = 1` (base field).
///
/// - `coeffs` are evaluations on `{0,1}^n`, indexed in LSB-first order.
/// - The lowest `alpha = log2(D)` bits are packed into one ring element via
///   coefficient embedding (the k=1 case of `psi`).
/// - The output is a flat table of length `2^(num_vars - alpha)` representing
///   the ring polynomial's evaluation table over the outer variables.
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

    let outer_vars = num_vars - alpha;
    let outer_len = 1usize
        .checked_shl(outer_vars as u32)
        .ok_or_else(|| HachiError::InvalidInput(format!("2^{outer_vars} does not fit usize")))?;

    let mut out = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let coeffs = std::array::from_fn(|j| {
            let idx = i + (j << outer_vars);
            coeffs[idx]
        });
        out.push(CyclotomicRing::from_coefficients(coeffs));
    }
    Ok(out)
}

/// Reduce inner openings (Lagrange vector) into a ring element.
///
/// Note: this implementation assumes `k = 1` (base field).
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

/// Evaluate the packed ring polynomial `F` at the outer point (Lagrange basis).
fn evaluate_packed_ring_poly<F: FieldCore, const D: usize>(
    packed_coeffs: &[CyclotomicRing<F, D>],
    outer_point: &[F],
) -> CyclotomicRing<F, D> {
    let weights = lagrange_weights(outer_point);
    debug_assert_eq!(weights.len(), packed_coeffs.len());
    packed_coeffs
        .iter()
        .zip(weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
            acc + f_i.scale(w_i)
        })
}

/// Trace map for k=1: `Tr_H(u) = d * ct(u)` for `R_q = F_q[X]/(X^d+1)`.
fn trace<F: CanonicalField, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
    let d = F::from_u64(D as u64);
    u.coefficients()[0] * d
}

pub(crate) fn eval_ring_at<F: FieldCore, const D: usize>(r: &CyclotomicRing<F, D>, alpha: &F) -> F {
    let mut acc = F::zero();
    let mut power = F::one();
    for coeff in r.coefficients() {
        acc = acc + (*coeff * power);
        power = power * *alpha;
    }
    acc
}

fn eval_ring_vec_at<F: FieldCore, const D: usize>(v: &[CyclotomicRing<F, D>], alpha: &F) -> Vec<F> {
    v.iter().map(|r| eval_ring_at(r, alpha)).collect()
}

fn eval_ring_matrix_at<F: FieldCore, const D: usize>(
    m: &[Vec<CyclotomicRing<F, D>>],
    alpha: &F,
) -> Vec<Vec<F>> {
    m.iter().map(|row| eval_ring_vec_at(row, alpha)).collect()
}

/// Compute `r[i] = (M[i]·z − y[i]) / (X^D + 1)` using schoolbook polynomial
/// multiplication followed by polynomial long division.
fn compute_r_via_poly_division<F: FieldCore, const D: usize>(
    m: &[Vec<CyclotomicRing<F, D>>],
    z: &[CyclotomicRing<F, D>],
    y: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let poly_len = 2 * D - 1;
    m.iter()
        .zip(y.iter())
        .map(|(row, y_i)| {
            let mut poly = vec![F::zero(); poly_len];
            for (m_ij, z_j) in row.iter().zip(z.iter()) {
                let a = m_ij.coefficients();
                let b = z_j.coefficients();
                for t in 0..D {
                    for s in 0..D {
                        poly[t + s] = poly[t + s] + a[t] * b[s];
                    }
                }
            }
            let y_coeffs = y_i.coefficients();
            for k in 0..D {
                poly[k] = poly[k] - y_coeffs[k];
            }
            // Polynomial long division by (X^D + 1), which is monic.
            let mut quotient = vec![F::zero(); D];
            for k in (D..poly_len).rev() {
                let q = poly[k];
                quotient[k - D] = q;
                poly[k - D] = poly[k - D] - q;
            }
            let coeffs: [F; D] = std::array::from_fn(|k| quotient[k]);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect()
}

#[allow(dead_code)]
fn compute_r_from_m_z_y<F: FieldCore, const D: usize>(
    m: &[Vec<CyclotomicRing<F, D>>],
    z: &[CyclotomicRing<F, D>],
    y: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if m.len() != y.len() {
        return Err(HachiError::InvalidSize {
            expected: m.len(),
            actual: y.len(),
        });
    }
    let mut out = Vec::with_capacity(m.len());
    for (row, y_i) in m.iter().zip(y.iter()) {
        if row.len() != z.len() {
            return Err(HachiError::InvalidSize {
                expected: row.len(),
                actual: z.len(),
            });
        }
        let mut acc = [F::zero(); D];
        for (m_ij, z_j) in row.iter().zip(z.iter()) {
            let a = m_ij.coefficients();
            let b = z_j.coefficients();
            for t in 0..D {
                let a_t = a[t];
                for k in t..D {
                    acc[k] = acc[k] + a_t * b[k - t];
                }
            }
        }
        let y_coeffs = y_i.coefficients();
        for k in 0..D {
            acc[k] = acc[k] - y_coeffs[k];
        }
        out.push(CyclotomicRing::from_coefficients(acc));
    }
    Ok(out)
}

fn compute_u_eval<F: CanonicalField, const D: usize, Cfg: CommitmentConfig>(
    ring_coeffs: &[CyclotomicRing<F, D>],
    opening_point: &crate::protocol::opening_point::RingOpeningPoint<F, D>,
) -> Result<CyclotomicRing<F, D>, HachiError> {
    let num_blocks = 1usize
        .checked_shl(Cfg::R as u32)
        .ok_or_else(|| HachiError::InvalidSetup("2^R does not fit usize".to_string()))?;
    let block_len = 1usize
        .checked_shl(Cfg::M as u32)
        .ok_or_else(|| HachiError::InvalidSetup("2^M does not fit usize".to_string()))?;
    let expected_len = num_blocks
        .checked_mul(block_len)
        .ok_or_else(|| HachiError::InvalidSetup("coeff length overflow".to_string()))?;
    if ring_coeffs.len() != expected_len {
        return Err(HachiError::InvalidSize {
            expected: expected_len,
            actual: ring_coeffs.len(),
        });
    }
    if opening_point.a.len() != block_len {
        return Err(HachiError::InvalidPointDimension {
            expected: block_len,
            actual: opening_point.a.len(),
        });
    }
    if opening_point.b.len() != num_blocks {
        return Err(HachiError::InvalidPointDimension {
            expected: num_blocks,
            actual: opening_point.b.len(),
        });
    }

    let mut acc = CyclotomicRing::<F, D>::zero();
    for i in 0..num_blocks {
        let mut inner = CyclotomicRing::<F, D>::zero();
        for j in 0..block_len {
            let idx = (j << Cfg::R)
                .checked_add(i)
                .ok_or_else(|| HachiError::InvalidSetup("index overflow".to_string()))?;
            inner += opening_point.a[j] * ring_coeffs[idx];
        }
        acc += opening_point.b[i] * inner;
    }
    Ok(acc)
}

fn r_decomp_levels<F: CanonicalField, Cfg: CommitmentConfig>() -> usize {
    let modulus = detect_field_modulus::<F>();
    let bits = 128 - (modulus.saturating_sub(1)).leading_zeros() as usize;
    let log_basis = Cfg::LOG_BASIS as usize;
    let mut levels = (bits + log_basis.saturating_sub(1)) / log_basis.max(1);
    if levels == 0 {
        levels = 1;
    }

    // The formula above gives enough digits for *unsigned* base-b representation.
    // Balanced digits {-b/2, ..., b/2-1} have a smaller positive range:
    //   max_positive = (b/2 - 1) * (b^l - 1) / (b - 1)
    // We need max_positive >= (q-1)/2 to cover all signed field values.
    let b = 1u128 << Cfg::LOG_BASIS;
    let half_q = modulus / 2;
    let half_b_minus_1 = b / 2 - 1;
    let b_minus_1 = b - 1;
    let mut b_pow = 1u128;
    for _ in 0..levels {
        b_pow = b_pow.saturating_mul(b);
    }
    let max_positive = half_b_minus_1.saturating_mul((b_pow - 1) / b_minus_1);
    if max_positive < half_q {
        levels += 1;
    }

    levels
}

fn expand_m_a<F: CanonicalField, const D: usize, Cfg: CommitmentConfig>(
    m_a: &[Vec<F>],
    alpha: F,
) -> Result<Vec<F>, HachiError> {
    if m_a.is_empty() {
        return Ok(Vec::new());
    }
    let rows = m_a.len();
    let cols = m_a[0].len();
    if cols == 0 {
        return Ok(vec![F::zero(); rows]);
    }
    for row in m_a.iter() {
        if row.len() != cols {
            return Err(HachiError::InvalidSize {
                expected: cols,
                actual: row.len(),
            });
        }
    }

    let levels = r_decomp_levels::<F, Cfg>();
    let total_cols = cols
        .checked_add(
            rows.checked_mul(levels)
                .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?,
        )
        .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?;

    let base = F::from_canonical_u128_reduced(1u128 << Cfg::LOG_BASIS);
    let mut gadget_row = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        gadget_row.push(power);
        power = power * base;
    }

    let mut alpha_pow = F::one();
    for _ in 0..D {
        alpha_pow = alpha_pow * alpha;
    }
    let denom = alpha_pow + F::one();

    let mut out = vec![F::zero(); rows * total_cols];
    for i in 0..rows {
        let row_start = i * total_cols;
        out[row_start..row_start + cols].copy_from_slice(&m_a[i]);
        let r_start = row_start + cols + i * levels;
        for (j, g) in gadget_row.iter().enumerate() {
            out[r_start + j] = -denom * *g;
        }
    }
    Ok(out)
}

/// Reshape `w` coefficients from ring-element-major order into sumcheck
/// evaluation-table order and pad to a power-of-two domain.
///
/// Input `w` is stored as `[z_0_coeff_0, ..., z_0_coeff_{D-1}, z_1_coeff_0, ...]`,
/// i.e. for each ring element `u`, the `D` coefficients `ℓ` are consecutive:
/// `w[u * D + ℓ]`.
///
/// The sumcheck instances need the variable ordering `(x, y)` where `x` indexes
/// the ring element (lower bits) and `y` indexes the coefficient (upper bits):
/// `w_evals[x + (y << num_u)]`.
///
/// Returns `(w_evals, num_u, num_l)`.
fn build_w_evals<F: FieldCore>(w: &[F], d: usize) -> (Vec<F>, usize, usize) {
    let num_l = d.trailing_zeros() as usize;
    let num_ring_elems = w.len() / d.max(1);
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let x_len = 1usize << num_u;
    let y_len = 1usize << num_l;
    let n = x_len * y_len;

    let mut evals = vec![F::zero(); n];
    for x in 0..x_len {
        for y in 0..y_len {
            let src = y + (x << num_l);
            if src < w.len() {
                let dst = x + (y << num_u);
                evals[dst] = w[src];
            }
        }
    }
    (evals, num_u, num_l)
}

/// Compute the M_alpha row count (number of rows in the ring-switch matrix).
fn m_row_count<Cfg: CommitmentConfig>() -> usize {
    Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A
}

/// Build the `m(x) = Σ_i eq(τ₁, i) * M_α(i, x)` evaluation table.
fn build_m_evals_x<F: FieldCore + CanonicalField>(
    m_a_flat: &[F],
    rows: usize,
    cols: usize,
    tau1: &[F],
) -> Vec<F> {
    let eq_tau1 = eq_evals(tau1);
    let x_len = cols.next_power_of_two();
    let mut m_evals = vec![F::zero(); x_len];
    for x in 0..x_len {
        let mut acc = F::zero();
        for i in 0..eq_tau1.len() {
            let row_val = if i < rows && x < cols {
                m_a_flat[i * cols + x]
            } else {
                F::zero()
            };
            acc = acc + eq_tau1[i] * row_val;
        }
        m_evals[x] = acc;
    }
    m_evals
}

/// Build the `α̃(y) = [1, α, α², ..., α^{D-1}]` evaluation table.
fn build_alpha_evals_y<F: FieldCore>(alpha: F, d: usize) -> Vec<F> {
    let mut out = vec![F::zero(); d];
    let mut power = F::one();
    for val in out.iter_mut() {
        *val = power;
        power = power * alpha;
    }
    out
}

/// Sample a tau vector of `n` scalars from the transcript.
fn sample_tau<F: FieldCore + CanonicalField, T: Transcript<F>>(
    transcript: &mut T,
    label: &[u8],
    n: usize,
) -> Vec<F> {
    (0..n).map(|_| transcript.challenge_scalar(label)).collect()
}

/// Build the coefficient vector for `w` by concatenating `z` digits and `r` digits.
pub(crate) fn build_w_coeffs<F: CanonicalField, const D: usize, Cfg: CommitmentConfig>(
    z: &[CyclotomicRing<F, D>],
    r: &[CyclotomicRing<F, D>],
) -> Vec<F> {
    let levels = r_decomp_levels::<F, Cfg>();
    let r_hat: Vec<CyclotomicRing<F, D>> = r
        .iter()
        .flat_map(|ri| ri.balanced_decompose_pow2(levels, Cfg::LOG_BASIS))
        .collect();

    let mut out = Vec::with_capacity((z.len() + r_hat.len()) * D);
    for elem in z.iter().chain(r_hat.iter()) {
        out.extend_from_slice(elem.coefficients());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::multilinear_evals::DenseMultilinearEvals;
    use crate::protocol::commitment::CommitmentConfig;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::F;
    use crate::{CommitmentScheme, Polynomial};

    #[test]
    fn verify_passes_for_consistent_opening() {
        let alpha = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let num_vars = DefaultCommitmentConfig::R + DefaultCommitmentConfig::M + alpha;
        let len = 1usize << num_vars;

        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = <HachiCommitmentScheme as CommitmentScheme<F>>::setup_prover(num_vars);
        let verifier_setup = <HachiCommitmentScheme as CommitmentScheme<F>>::setup_verifier(&setup);

        let (commitment, hint) =
            <HachiCommitmentScheme as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let opening = poly.evaluate(&opening_point);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <HachiCommitmentScheme as CommitmentScheme<F>>::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <HachiCommitmentScheme as CommitmentScheme<F>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
        );

        assert!(result.is_ok());
    }
}
