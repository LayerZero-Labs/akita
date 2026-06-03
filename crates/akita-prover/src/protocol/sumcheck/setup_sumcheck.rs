//! Setup-product sumcheck for a dense table against two disjoint factors.
//!
//! The table is laid out as `left * right_len + right`. The right factor is
//! bound first, then the left factor. This matches setup products of the form
//! `S(lambda, y) * omega(lambda) * alpha(y)` without materializing the full
//! `omega(lambda) * alpha(y)` table.

use akita_algebra::ring::scalar_powers;
use akita_algebra::uni_poly::UniPoly;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::Transcript;
use akita_types::{
    gadget_row_scalars, LevelParams, RingRelationInstance, RingSubfieldEncoding,
    SETUP_SUMCHECK_DEGREE,
};
use akita_verifier::{prepare_ring_switch_row_eval, RingSwitchReplay, SetupEvaluator};

/// Proves `sum_{l,r} table[l,r] * left_factor[l] * right_factor[r]`.
pub struct SetupSumcheckProver<E: FieldCore> {
    table: Vec<E>,
    left_factor: Vec<E>,
    right_factor: Vec<E>,
    input_claim: E,
    right_rounds: usize,
    total_rounds: usize,
}

/// Output of the setup product sumcheck prover.
pub struct SetupSumcheckProverOutput<E: FieldCore> {
    /// Claimed setup contribution fed into the stage-2 final row evaluation.
    pub claim: E,
    /// Degree-two product sumcheck over `S(lambda, y) * omega(lambda) * alpha(y)`.
    pub sumcheck: SumcheckProof<E>,
}

impl<E: FieldCore> SetupSumcheckProver<E> {
    /// Construct a factored product-sumcheck prover.
    ///
    /// # Errors
    ///
    /// Returns an error if factor lengths are not powers of two, are empty, or
    /// if `table.len() != left_factor.len() * right_factor.len()`.
    fn new(table: Vec<E>, left_factor: Vec<E>, right_factor: Vec<E>) -> Result<Self, AkitaError> {
        if left_factor.is_empty()
            || right_factor.is_empty()
            || !left_factor.len().is_power_of_two()
            || !right_factor.len().is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "factored product dimensions must be non-empty powers of two".to_string(),
            ));
        }
        let expected_len = left_factor
            .len()
            .checked_mul(right_factor.len())
            .ok_or_else(|| AkitaError::InvalidInput("factored product size overflow".into()))?;
        if table.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: table.len(),
            });
        }

        let input_claim = product_claim(&table, &left_factor, &right_factor);
        let right_rounds = right_factor.len().trailing_zeros() as usize;
        let total_rounds = right_rounds + left_factor.len().trailing_zeros() as usize;
        Ok(Self {
            table,
            left_factor,
            right_factor,
            input_claim,
            right_rounds,
            total_rounds,
        })
    }

    /// Prove the setup-product sumcheck from packed setup rings and the
    /// ring-switch row evaluation that determines the factored weights.
    ///
    /// Internally derives the `(required, bar_omega, alpha_pows)` factored
    /// terms from the level parameters and ring relation, then builds the
    /// `lambda * D + y` setup table (zero-padded up to the next power-of-two
    /// lambda dimension) and runs the product sumcheck.
    ///
    /// # Errors
    ///
    /// Returns an error if the ring-switch row evaluation fails, the setup
    /// slice is too small, the padded table size overflows, factor dimensions
    /// are invalid, or any sumcheck round polynomial exceeds its degree bound.
    #[allow(clippy::too_many_arguments)]
    pub fn prove<F, T, SampleRound, const D: usize>(
        setup_entries: &[CyclotomicRing<F, D>],
        lp: &LevelParams,
        relation: &RingRelationInstance<F, D>,
        tau1: &[E],
        alpha: E,
        x_challenges: &[E],
        row_coefficients: &[E],
        transcript: &mut T,
        sample_round: SampleRound,
    ) -> Result<SetupSumcheckProverOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: RingSubfieldEncoding<F> + FromPrimitiveInt + LiftBase<F> + AkitaSerialize,
        T: Transcript<F>,
        SampleRound: FnMut(&mut T) -> E,
    {
        let (required, mut bar_omega, alpha_pows) = prepare_setup_sumcheck_terms::<F, E, D>(
            lp,
            relation,
            tau1,
            alpha,
            x_challenges,
            row_coefficients,
        )?;

        if required > setup_entries.len() {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected setup product".to_string(),
            ));
        }

        let lambda_len = required.checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidSetup("setup product lambda length overflow".into())
        })?;
        bar_omega.resize(lambda_len, E::zero());

        let table_len = lambda_len.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidSetup("setup product table length overflow".into())
        })?;
        let mut setup_table = vec![E::zero(); table_len];
        cfg_chunks_mut!(&mut setup_table, D)
            .enumerate()
            .for_each(|(lambda, row)| {
                if lambda < required {
                    for (slot, &coeff) in row.iter_mut().zip(setup_entries[lambda].coefficients()) {
                        *slot = E::lift_base(coeff);
                    }
                }
            });

        let mut prover = Self::new(setup_table, bar_omega, alpha_pows)?;
        let claim = prover.input_claim();
        let (sumcheck, _challenges, _final_claim) =
            <Self as SumcheckInstanceProverExt<E>>::prove::<F, T, _>(
                &mut prover,
                transcript,
                sample_round,
            )?;
        Ok(SetupSumcheckProverOutput { claim, sumcheck })
    }
}

impl<E: FieldCore> SumcheckInstanceProver<E> for SetupSumcheckProver<E> {
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    fn degree_bound(&self) -> usize {
        SETUP_SUMCHECK_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let (constant, linear, quadratic) = if round < self.right_rounds {
            accumulate_right_round(&self.table, &self.left_factor, &self.right_factor)
        } else {
            accumulate_left_round(&self.table, &self.left_factor, self.right_factor[0])
        };
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, round: usize, r_round: E) {
        if round < self.right_rounds {
            fold_right_round(&mut self.table, &mut self.right_factor, r_round);
        } else {
            fold_left_round(&mut self.table, &mut self.left_factor, r_round);
        }
    }
}

/// Derive the factored product-sumcheck terms `(required, bar_omega, alpha_pows)`
/// from the level parameters and ring relation via the ring-switch row
/// evaluation.
fn prepare_setup_sumcheck_terms<F, E, const D: usize>(
    lp: &LevelParams,
    relation: &RingRelationInstance<F, D>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
    row_coefficients: &[E],
) -> Result<(usize, Vec<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + LiftBase<F>,
{
    let alpha_pows = scalar_powers(alpha, D);
    let replay = RingSwitchReplay {
        relation,
        row_coefficients,
        lp,
    };
    let prepared = prepare_ring_switch_row_eval::<F, E, D>(&replay, alpha, tau1)?;
    let num_t_vectors = relation
        .commitment_routing()
        .num_polys_per_commitment_group()
        .iter()
        .try_fold(0usize, |acc, &count| {
            acc.checked_add(count)
                .ok_or_else(|| AkitaError::InvalidSetup("t-vector count overflow".to_string()))
        })?;
    let fold_gadget = gadget_row_scalars::<F>(
        lp.num_digits_fold(num_t_vectors, F::modulus_bits()),
        lp.log_basis,
    );
    let layout = relation.segment_layout(lp)?;
    let evaluator = SetupEvaluator::new(
        &prepared,
        x_challenges,
        None,
        None,
        &alpha_pows,
        &fold_gadget,
        layout.offset_w,
        layout.offset_t,
        layout.offset_z,
    );
    let plan = evaluator.prepare()?;
    let required = plan.required();
    let bar_omega = plan.materialize_bar_omega();
    Ok((required, bar_omega, alpha_pows.to_vec()))
}

fn product_claim<E: FieldCore>(table: &[E], left_factor: &[E], right_factor: &[E]) -> E {
    let right_len = right_factor.len();
    cfg_fold_reduce!(
        0..left_factor.len(),
        E::zero,
        |mut acc, left_idx| {
            let left_weight = left_factor[left_idx];
            let row = &table[left_idx * right_len..(left_idx + 1) * right_len];
            for (&value, &right_weight) in row.iter().zip(right_factor.iter()) {
                acc += value * left_weight * right_weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

fn accumulate_right_round<E: FieldCore>(
    table: &[E],
    left_factor: &[E],
    right_factor: &[E],
) -> (E, E, E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    cfg_fold_reduce!(
        0..left_factor.len(),
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), left_idx| {
            let left_weight = left_factor[left_idx];
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                let s0 = table[row_base + 2 * pair_idx];
                let s1 = table[row_base + 2 * pair_idx + 1];
                let f0 = left_weight * right_factor[2 * pair_idx];
                let f1 = left_weight * right_factor[2 * pair_idx + 1];
                let ds = s1 - s0;
                let df = f1 - f0;
                constant += s0 * f0;
                linear += s0 * df + ds * f0;
                quadratic += ds * df;
            }
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

fn accumulate_left_round<E: FieldCore>(
    table: &[E],
    left_factor: &[E],
    right_weight: E,
) -> (E, E, E) {
    let half = left_factor.len() / 2;
    cfg_fold_reduce!(
        0..half,
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), pair_idx| {
            let s0 = table[2 * pair_idx];
            let s1 = table[2 * pair_idx + 1];
            let f0 = left_factor[2 * pair_idx] * right_weight;
            let f1 = left_factor[2 * pair_idx + 1] * right_weight;
            let ds = s1 - s0;
            let df = f1 - f0;
            constant += s0 * f0;
            linear += s0 * df + ds * f0;
            quadratic += ds * df;
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

fn fold_pair<E: FieldCore>(left: E, right: E, r: E) -> E {
    left + r * (right - left)
}

fn fold_right_round<E: FieldCore>(table: &mut Vec<E>, right_factor: &mut Vec<E>, r: E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    let left_len = table.len() / right_len;
    let mut folded = vec![E::zero(); left_len * half];
    cfg_chunks_mut!(&mut folded, half)
        .enumerate()
        .for_each(|(left_idx, row)| {
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                row[pair_idx] = fold_pair(
                    table[row_base + 2 * pair_idx],
                    table[row_base + 2 * pair_idx + 1],
                    r,
                );
            }
        });
    let folded_right = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(right_factor[2 * idx], right_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *right_factor = folded_right;
    *table = folded;
}

fn fold_left_round<E: FieldCore>(table: &mut Vec<E>, left_factor: &mut Vec<E>, r: E) {
    let half = left_factor.len() / 2;
    let folded_table = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(table[2 * idx], table[2 * idx + 1], r))
        .collect::<Vec<_>>();
    let folded_left = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(left_factor[2 * idx], left_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *table = folded_table;
    *left_factor = folded_left;
}
