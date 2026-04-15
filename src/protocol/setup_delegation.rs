use crate::algebra::eq_poly::EqPolynomial;
use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::poly::multilinear_eval;
use crate::error::HachiError;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::HachiCommitmentLayout;
use crate::protocol::commitment::HachiLevelParams;
use crate::protocol::commitment_scheme::{
    prove_without_setup_delegation, verify_without_setup_delegation,
};
use crate::protocol::opening_point::BasisMode;
use crate::protocol::proof::SetupDelegationProof;
use crate::protocol::ring_switch::{
    eval_matrix_weight_at_point, gadget_row_scalars, single_proof_matrix_weight_entry,
    single_proof_matrix_weight_geometry,
};
use crate::protocol::commitment::{HachiVerifierSetup, RingCommitment};
use crate::protocol::shared_matrix_setup::{
    SharedMatrixOpeningConfig, SharedMatrixSetup, SharedMatrixTensorLayout,
};
use crate::protocol::sumcheck::setup_claim::SetupClaimProver;
use crate::protocol::sumcheck::{prove_sumcheck, SumcheckProof};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};

const ABSORB_DELEGATION_CLAIM: &[u8] = b"hachi_setup_delegation_claim";
const ABSORB_SHARED_MATRIX_EVAL: &[u8] = b"hachi_shared_matrix_eval";
const CHALLENGE_DELEGATION_ROUND: &[u8] = b"hachi_delegation_sumcheck_round";

fn dense_poly_field_evals<F: FieldCore, const D: usize>(
    poly: &crate::protocol::hachi_poly_ops::DensePoly<F, D>,
) -> Vec<F> {
    let mut evals = Vec::with_capacity(poly.coeffs.len() * D);
    for ring in &poly.coeffs {
        evals.extend_from_slice(ring.coefficients());
    }
    evals
}

/// Materialize the full matrix weight table as a flat field-element vector.
///
/// Uses the canonical tensor layout from `shared_matrix_setup`: index
/// `(row * padded_stride + col) * D + k`.
pub(crate) fn materialize_matrix_weight<F: FieldCore + CanonicalField, const D: usize>(
    eq_tau1: &[F],
    alpha_evals_y: &[F],
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    tensor_layout: SharedMatrixTensorLayout,
    r_x: &[F],
) -> Vec<F> {
    let geometry = single_proof_matrix_weight_geometry(level_params, layout);
    let fold_gadget = gadget_row_scalars::<F>(geometry.depth_fold, geometry.log_basis);
    let eq_r_x = EqPolynomial::evals(r_x);

    let mut weight = vec![F::zero(); tensor_layout.field_len()];

    for row in 0..geometry.max_row {
        for col in 0..tensor_layout.stride {
            let w2 = single_proof_matrix_weight_entry(
                row,
                col,
                eq_tau1,
                &eq_r_x,
                geometry,
                &fold_gadget,
            );
            let flat_base = (row * tensor_layout.padded_stride + col) * D;
            for k in 0..D {
                weight[flat_base + k] = alpha_evals_y[k] * w2;
            }
        }
    }

    weight
}

/// Data needed to generate a delegation proof for a single level.
pub(crate) struct DelegationIntermediates<F: FieldCore> {
    pub m_evals_x: Vec<F>,
    pub alg_m_evals_x: Vec<F>,
    pub eq_tau1: Vec<F>,
    pub alpha_evals_y: Vec<F>,
    pub level_params: HachiLevelParams,
    pub layout: HachiCommitmentLayout,
    pub col_bits: usize,
    pub ring_bits: usize,
}

/// Generate a setup-delegation proof for a single level.
///
/// The caller has already run stage 1 + stage 2 and holds:
/// - `intermediates`: ring-switch data (m_evals_x, alg_m_evals_x, tau1, etc.)
/// - `sumcheck_challenges`: the full `(r_y, r_x)` challenges from stage 2
/// - `sm_setup`: the committed shared matrix polynomial
/// - `transcript`: current Fiat-Shamir state (after all fold levels)
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_setup_delegation_proof<F, T, const D: usize, Cfg>(
    intermediates: &DelegationIntermediates<F>,
    sumcheck_challenges: &[F],
    sm_setup: &SharedMatrixSetup<F, D>,
    transcript: &mut T,
) -> Result<SetupDelegationProof<F>, HachiError>
where
    F: FieldCore
        + CanonicalField
        + FieldSampling
        + HasWide
        + HasUnreducedOps
        + Valid
        + FromSmallInt,
    T: Transcript<F>,
    Cfg: SharedMatrixOpeningConfig<Field = F>,
{
    let ring_bits = intermediates.ring_bits;
    let x_challenges = &sumcheck_challenges[ring_bits..];

    let setup_evals_x: Vec<F> = intermediates
        .m_evals_x
        .iter()
        .zip(&intermediates.alg_m_evals_x)
        .map(|(m, a)| *m - *a)
        .collect();

    let claimed_setup_val = multilinear_eval(&setup_evals_x, x_challenges)?;

    transcript.append_field(ABSORB_DELEGATION_CLAIM, &claimed_setup_val);

    let sm_evals = dense_poly_field_evals(&sm_setup.shared_matrix_poly);

    let matrix_weight = materialize_matrix_weight::<F, D>(
        &intermediates.eq_tau1,
        &intermediates.alpha_evals_y,
        &intermediates.level_params,
        intermediates.layout,
        sm_setup.tensor_layout,
        x_challenges,
    );

    let num_vars = sm_setup.tensor_layout.num_vars;
    let mut prover =
        SetupClaimProver::new(sm_evals.clone(), matrix_weight, num_vars, claimed_setup_val);
    let (setup_claim_sumcheck, setup_challenges, _final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_DELEGATION_ROUND)
        })?;

    let shared_matrix_eval = multilinear_eval(&sm_evals, &setup_challenges)?;

    transcript.append_field(ABSORB_SHARED_MATRIX_EVAL, &shared_matrix_eval);

    let shared_matrix_opening_proof = prove_without_setup_delegation::<
        F,
        _,
        D,
        <Cfg as SharedMatrixOpeningConfig>::InnerCfg,
        _,
    >(
        &sm_setup.prover_setup,
        &sm_setup.shared_matrix_poly,
        &setup_challenges,
        sm_setup.commit_hint.clone(),
        transcript,
        &sm_setup.commitment,
        BasisMode::Lagrange,
    )?;

    Ok(SetupDelegationProof {
        claimed_setup_val,
        setup_claim_sumcheck,
        shared_matrix_eval,
        shared_matrix_opening_proof: Box::new(shared_matrix_opening_proof),
    })
}

/// Replay sumcheck rounds without the final oracle check.
///
/// Returns `(challenges, final_claim)` so the caller can perform
/// a deferred oracle check after computing values that depend on the challenges.
fn replay_sumcheck_rounds<F, T, E, S>(
    proof: &SumcheckProof<E>,
    input_claim: E,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(Vec<E>, E), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
{
    use crate::protocol::transcript::labels;

    if proof.round_polys.len() != num_rounds {
        return Err(HachiError::InvalidSize {
            expected: num_rounds,
            actual: proof.round_polys.len(),
        });
    }

    let mut claim = input_claim;
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let mut challenges = Vec::with_capacity(num_rounds);
    for poly in &proof.round_polys {
        if poly.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
        let r_i = sample_challenge(transcript);
        challenges.push(r_i);
        claim = poly.eval_from_hint(&claim, &r_i);
    }

    Ok((challenges, claim))
}

/// Verify a setup-delegation proof for a single level.
///
/// The verifier has run the normal fold verification and holds:
/// - `delegation_proof`: the proof to verify
/// - `sm_setup`: the committed shared matrix (with inner PCS verifier setup)
/// - Ring switch verifier parameters for computing the matrix weight at a point
/// - `transcript`: current Fiat-Shamir state (must match prover)
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_setup_delegation_proof<F, T, const D: usize, Cfg>(
    delegation_proof: &SetupDelegationProof<F>,
    eq_tau1: &[F],
    alpha_evals_y: &[F],
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    x_challenges: &[F],
    tensor_layout: &SharedMatrixTensorLayout,
    inner_verifier_setup: &HachiVerifierSetup<F>,
    commitment: &RingCommitment<F, D>,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore
        + CanonicalField
        + FieldSampling
        + HasWide
        + HasUnreducedOps
        + Valid
        + FromSmallInt,
    T: Transcript<F>,
    Cfg: SharedMatrixOpeningConfig<Field = F>,
{
    transcript.append_field(ABSORB_DELEGATION_CLAIM, &delegation_proof.claimed_setup_val);

    let num_vars = tensor_layout.num_vars;

    let (setup_challenges, final_claim) = replay_sumcheck_rounds::<F, _, F, _>(
        &delegation_proof.setup_claim_sumcheck,
        delegation_proof.claimed_setup_val,
        num_vars,
        2,
        transcript,
        |tr| tr.challenge_scalar(CHALLENGE_DELEGATION_ROUND),
    )?;

    let (r_row, r_col, r_k) = tensor_layout.split_point(&setup_challenges)?;

    let matrix_weight_eval = eval_matrix_weight_at_point::<F, D>(
        r_row,
        r_col,
        r_k,
        x_challenges,
        alpha_evals_y,
        eq_tau1,
        level_params,
        layout,
        *tensor_layout,
    )?;

    let expected = delegation_proof.shared_matrix_eval * matrix_weight_eval;
    if final_claim != expected {
        tracing::error!("setup delegation sumcheck final claim mismatch");
        return Err(HachiError::InvalidProof);
    }

    transcript.append_field(
        ABSORB_SHARED_MATRIX_EVAL,
        &delegation_proof.shared_matrix_eval,
    );

    verify_without_setup_delegation::<F, _, D, <Cfg as SharedMatrixOpeningConfig>::InnerCfg>(
        &delegation_proof.shared_matrix_opening_proof,
        inner_verifier_setup,
        transcript,
        &setup_challenges,
        &delegation_proof.shared_matrix_eval,
        commitment,
        BasisMode::Lagrange,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::eq_poly::EqPolynomial;
    use crate::protocol::commitment::CommitmentConfig;
    use crate::protocol::commitment_scheme::HachiCommitmentScheme;
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::commitment::HachiScheduleInputs;
    use crate::protocol::hachi_poly_ops::HachiPolyOps;
    use crate::protocol::opening_point::{ring_opening_point_from_field, BlockOrder};
    use crate::protocol::quadratic_equation::QuadraticEquation;
    use crate::protocol::ring_switch::{
        build_alpha_evals_y, compute_alg_m_evals_x_with_claim_groups,
        compute_m_evals_x_with_claim_groups, m_row_count,
    };
    use crate::protocol::transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::protocol::{AppendToTranscript, DensePoly};
    use crate::CommitmentScheme;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    type F = fp128::Field;
    type Cfg = fp128::D128Full;
    const D: usize = Cfg::D;

    fn setup_delegation_proof_roundtrip_for_cfg<const D: usize, Cfg>(nv: usize)
    where
        Cfg: SharedMatrixOpeningConfig<Field = F>,
    {
        let layout = Cfg::commitment_layout(nv).expect("layout");
        let level_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: nv,
            level: 0,
            current_w_len: 1usize << nv,
        });

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..(1usize << nv))
            .map(|i| F::from_u64((i % 2) as u64))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).expect("dense poly");
        let point: Vec<F> = (0..nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv, 1);
        let (commitment, _batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");

        let mut transcript = Blake2bTranscript::<F>::new(b"delegation-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            layout.block_len,
        );
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let quad_eq = QuadraticEquation::<F, D, Cfg>::new_prover(
            &setup.ntt_shared,
            ring_opening_point,
            &poly,
            w_folded,
            level_params.clone(),
            _batched_hint.into_flattened(),
            &mut transcript,
            &commitment,
            &y_ring,
            layout,
            setup.expanded.seed.max_stride(),
        )
        .expect("quadratic equation");

        let alpha = F::from_u64(42);
        let alpha_evals_y = build_alpha_evals_y(alpha, D);
        let rows = m_row_count(&level_params);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let eq_tau1 = EqPolynomial::evals(&tau1);

        let m_evals_x = compute_m_evals_x_with_claim_groups::<F, D>(
            &setup.expanded,
            quad_eq.opening_point(),
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            layout,
            &tau1,
            &[1usize],
        )
        .expect("m_evals_x");

        let alg_m_evals_x = compute_alg_m_evals_x_with_claim_groups::<F, D>(
            quad_eq.opening_point(),
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            layout,
            &tau1,
            &[1usize],
        )
        .expect("alg_m_evals_x");

        let col_bits = (m_evals_x.len().next_power_of_two()).trailing_zeros() as usize;
        let ring_bits = D.trailing_zeros() as usize;

        let x_challenges: Vec<F> = (0..col_bits)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let sumcheck_challenges: Vec<F> = {
            let y_challenges: Vec<F> = (0..ring_bits)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect();
            let mut sc = y_challenges;
            sc.extend_from_slice(&x_challenges);
            sc
        };

        let intermediates = DelegationIntermediates {
            m_evals_x,
            alg_m_evals_x,
            eq_tau1: eq_tau1.clone(),
            alpha_evals_y: alpha_evals_y.clone(),
            level_params: level_params.clone(),
            layout,
            col_bits,
            ring_bits,
        };

        let sm_setup = SharedMatrixSetup::<F, D>::from_main_prover_setup::<Cfg>(&setup)
            .expect("SharedMatrixSetup creation");

        let mut prove_transcript = Blake2bTranscript::<F>::new(b"delegation-sumcheck");
        let delegation_proof = generate_setup_delegation_proof::<F, _, D, Cfg>(
            &intermediates,
            &sumcheck_challenges,
            &sm_setup,
            &mut prove_transcript,
        )
        .expect("delegation proof generation");

        assert_ne!(
            delegation_proof.claimed_setup_val,
            F::zero(),
            "setup value should be non-zero for a non-trivial matrix"
        );

        let mut verify_transcript = Blake2bTranscript::<F>::new(b"delegation-sumcheck");
        verify_setup_delegation_proof::<F, _, D, Cfg>(
            &delegation_proof,
            &eq_tau1,
            &alpha_evals_y,
            &level_params,
            layout,
            &x_challenges,
            &sm_setup.tensor_layout,
            &sm_setup.verifier_setup,
            &sm_setup.commitment,
            &mut verify_transcript,
        )
        .expect("delegation proof verification should succeed");
    }

    #[test]
    fn setup_delegation_proof_roundtrip() {
        setup_delegation_proof_roundtrip_for_cfg::<D, Cfg>(12);
    }

    #[test]
    fn setup_delegation_proof_roundtrip_for_multirow_onehot_root() {
        type MultiRowCfg = fp128::D32OneHot;
        const D: usize = MultiRowCfg::D;
        const NV: usize = 12;

        let level_params = MultiRowCfg::level_params(HachiScheduleInputs {
            max_num_vars: NV,
            level: 0,
            current_w_len: 1usize << NV,
        });
        assert!(
            level_params.n_a > 1,
            "fixture must exercise the multi-row delegated path"
        );

        setup_delegation_proof_roundtrip_for_cfg::<D, MultiRowCfg>(NV);
    }
}
