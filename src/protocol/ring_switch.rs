//! Ring switching logic for the Hachi PCS (Section 4.3).
//!
//! Handles the transition from the ring-based quadratic equation to field-based
//! sumcheck instances by expanding the ring elements into their coefficient
//! vectors and setting up the evaluation tables.

use crate::protocol::commitment::{
    hachi_recursive_level_layout_from_params, recursive_level_decomposition_from_root,
};
use crate::protocol::config::{CommitmentConfig, CommitmentEnvelope, DecompositionParams};
use crate::{CanonicalField, FieldCore, FieldSampling};
use akita_algebra::CyclotomicRing;
#[cfg(all(test, feature = "parallel"))]
use akita_field::parallel::*;
use akita_field::HachiError;
use akita_prover::crt_ntt::NttSlotCache;
use akita_prover::linear::mat_vec_mul_ntt_single_i8;
use akita_prover::RecursiveWitnessFlat;
use akita_types::{
    HachiCommitmentHint, HachiScheduleLookupKey, HachiSchedulePlan, RingCommitment,
    ScheduleProvider,
};
use akita_types::{HachiScheduleInputs, LevelParams};
#[cfg(test)]
use std::array::from_fn;
use std::marker::PhantomData;

#[cfg(test)]
pub(crate) fn compute_r_via_poly_division<F: FieldCore + CanonicalField, const D: usize>(
    m: &[Vec<CyclotomicRing<F, D>>],
    z: &[CyclotomicRing<F, D>],
    y: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let poly_len = 2 * D - 1;
    let out = m
        .iter()
        .zip(y.iter())
        .map(|(row, y_i)| {
            let column_contribution =
                |m_ij: &CyclotomicRing<F, D>, z_j: &CyclotomicRing<F, D>| -> Vec<F> {
                    let mut local = vec![F::zero(); poly_len];
                    if m_ij.is_zero() {
                        return local;
                    }
                    let a = m_ij.coefficients();
                    let b = z_j.coefficients();
                    let is_scalar = a[1..].iter().all(|c| c.is_zero());
                    if is_scalar {
                        let scalar = a[0];
                        for s in 0..D {
                            local[s] = scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                local[t + s] += a[t] * b[s];
                            }
                        }
                    }
                    local
                };

            let pointwise_add = |mut a: Vec<F>, b: Vec<F>| -> Vec<F> {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += *bi;
                }
                a
            };

            #[cfg(feature = "parallel")]
            let mut poly = row
                .par_iter()
                .zip(z.par_iter())
                .fold(
                    || vec![F::zero(); poly_len],
                    |acc, (m_ij, z_j)| pointwise_add(acc, column_contribution(m_ij, z_j)),
                )
                .reduce(|| vec![F::zero(); poly_len], pointwise_add);

            #[cfg(not(feature = "parallel"))]
            let mut poly = row
                .iter()
                .zip(z.iter())
                .fold(vec![F::zero(); poly_len], |acc, (m_ij, z_j)| {
                    pointwise_add(acc, column_contribution(m_ij, z_j))
                });
            let y_coeffs = y_i.coefficients();
            for k in 0..D {
                poly[k] -= y_coeffs[k];
            }
            let mut quotient = vec![F::zero(); D];
            for k in (D..poly_len).rev() {
                let q = poly[k];
                quotient[k - D] = q;
                poly[k - D] -= q;
            }
            let coeffs: [F; D] = from_fn(|k| quotient[k]);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    Ok(out)
}

/// Derived commitment config for recursive w-openings.
///
/// Sets `log_commit_bound = log_basis` (w's entries are balanced digits) and
/// `log_open_bound = parent's open bound` (opening folds produce full-field
/// coefficients).
///
/// For the default fp128 presets, this uses the same decomposition parameters
/// as the corresponding log-bounded preset, but at the caller-selected ring
/// dimension `D`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WCommitmentConfig<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

impl<const D: usize, Cfg: CommitmentConfig> ScheduleProvider for WCommitmentConfig<D, Cfg> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        Cfg::schedule_table()
    }

    fn schedule_key(key: HachiScheduleLookupKey) -> String {
        Cfg::schedule_key(key)
    }

    fn schedule_plan(key: HachiScheduleLookupKey) -> Result<Option<HachiSchedulePlan>, HachiError> {
        Cfg::schedule_plan(key)
    }
}

impl<const D: usize, Cfg: CommitmentConfig> CommitmentConfig for WCommitmentConfig<D, Cfg> {
    type Field = Cfg::Field;
    const D: usize = D;

    fn decomposition() -> DecompositionParams {
        recursive_level_decomposition_from_root(
            Cfg::decomposition(),
            Cfg::decomposition().log_basis,
        )
    }

    fn stage1_challenge_config(d: usize) -> akita_algebra::SparseChallengeConfig {
        Cfg::stage1_challenge_config(d)
    }

    fn audited_root_rank(role: crate::protocol::config::AjtaiRole, max_num_vars: usize) -> usize {
        Cfg::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Cfg::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), HachiError> {
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams {
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        debug_assert_eq!(params.ring_dimension, D);
        params
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, HachiError> {
        Cfg::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn root_level_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, HachiError> {
        Cfg::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        Cfg::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: HachiScheduleInputs) -> (u32, u32) {
        Cfg::log_basis_search_range(inputs)
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, HachiError> {
        Err(HachiError::InvalidSetup(
            "recursive w layout requires active level params".to_string(),
        ))
    }
}

/// Commit the witness vector `w` (D-agnostic `Vec<i8>`) into `D`-sized ring
/// elements and compute the ring commitment.
///
/// This is the **D-boundary** in the protocol: the ring switch at level k
/// produces `w` using D_k operations, but `commit_w` re-chunks `w` into
/// D_{k+1}-sized ring elements and commits using D_{k+1} NTT caches.
///
/// For constant-D configs, D_k = D_{k+1} = D and the distinction is moot.
///
/// # Errors
///
/// Returns an error if the commitment layout derivation or NTT mat-vec fails.
#[tracing::instrument(skip_all, name = "commit_w")]
#[inline(never)]
pub(crate) fn commit_w<F, const D: usize, Cfg>(
    w: &RecursiveWitnessFlat,
    ntt_shared: &NttSlotCache<D>,
    lp: &LevelParams,
    stride: usize,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig<Field = F>,
{
    if !w.len().is_multiple_of(D) {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: w.len(),
        });
    }

    let w_layout = hachi_recursive_level_layout_from_params::<Cfg>(lp, w.len())?;

    let num_blocks = w_layout.num_blocks;
    let block_len = w_layout.block_len;
    let depth_commit = w_layout.num_digits_commit;
    let depth_open = w_layout.num_digits_open;
    let log_basis = w_layout.log_basis;
    let num_ring_elems = w.len() / D;
    tracing::debug!(
        num_ring_elems,
        num_blocks,
        block_len,
        depth_commit,
        depth_open,
        m_vars = w_layout.m_vars,
        r_vars = w_layout.r_vars,
        inner_width = w_layout.inner_width(),
        pow2_block = 1usize << w_layout.m_vars,
        "commit_w layout"
    );

    let w_view = w.view::<F, D>()?;
    let inner = w_view.commit_inner_witness(
        ntt_shared,
        lp.a_key.row_len(),
        block_len,
        num_blocks,
        depth_commit,
        depth_open,
        log_basis,
        stride,
    )?;

    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        ntt_shared,
        lp.b_key.row_len(),
        stride,
        inner.t_hat.flat_digits(),
    );
    let hint = HachiCommitmentHint::singleton_with_t(inner.t_hat, inner.t);
    Ok((RingCommitment { u }, hint))
}

#[cfg(test)]
mod tests {
    use super::compute_r_via_poly_division;
    use crate::protocol::commitment_scheme::HachiCommitmentScheme;
    use crate::protocol::config::proof_optimized::fp128;
    use crate::protocol::CommitmentConfig;
    use crate::{CanonicalField, CommitmentProver, Transcript};
    use akita_algebra::ring::scalar_powers;
    use akita_algebra::CyclotomicRing;
    use akita_prover::ring_switch::{
        build_w_evals_compact, compute_m_evals_x, ring_switch_build_w,
    };
    use akita_prover::{DensePoly, HachiPolyOps, QuadraticEquation};
    use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
    use akita_transcript::Blake2bTranscript;
    use akita_types::AppendToTranscript;
    use akita_types::{ring_opening_point_from_field, BasisMode, BlockOrder};
    use akita_verifier::prepare_m_eval;
    use akita_verifier::relation_claim_from_rows;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::array::from_fn;

    use crate::{FieldCore, FromSmallInt};

    fn compute_r_schoolbook<F: FieldCore, const D: usize>(
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
                    if m_ij.is_zero() {
                        continue;
                    }
                    let a = m_ij.coefficients();
                    let b = z_j.coefficients();
                    let is_scalar = a[1..].iter().all(|c| c.is_zero());
                    if is_scalar {
                        let scalar = a[0];
                        for s in 0..D {
                            poly[s] += scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                poly[t + s] += a[t] * b[s];
                            }
                        }
                    }
                }
                let y_coeffs = y_i.coefficients();
                for k in 0..D {
                    poly[k] -= y_coeffs[k];
                }
                let mut quotient = vec![F::zero(); D];
                for k in (D..poly_len).rev() {
                    let q = poly[k];
                    quotient[k - D] = q;
                    poly[k - D] -= q;
                }
                let coeffs: [F; D] = from_fn(|k| quotient[k]);
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    #[test]
    fn compute_r_matches_schoolbook_reference() {
        type F = fp128::Field;
        const D: usize = 64;

        let m: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        if (i + j) % 3 == 0 {
                            let mut coeffs = [F::zero(); D];
                            coeffs[0] = F::from_u64((i * 5 + j + 1) as u64);
                            CyclotomicRing::from_coefficients(coeffs)
                        } else {
                            let coeffs = from_fn(|k| {
                                F::from_u64((i as u64 * 1000 + j as u64 * 100 + k as u64 + 1) % 97)
                            });
                            CyclotomicRing::from_coefficients(coeffs)
                        }
                    })
                    .collect()
            })
            .collect();
        let z: Vec<CyclotomicRing<F, D>> = (0..4)
            .map(|j| {
                let coeffs = from_fn(|k| F::from_u64((j as u64 * 37 + k as u64 + 5) % 89));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let y: Vec<CyclotomicRing<F, D>> = (0..3)
            .map(|i| {
                let coeffs = from_fn(|k| F::from_u64((i as u64 * 29 + k as u64 + 7) % 83));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();

        let expected = compute_r_schoolbook(&m, &z, &y);
        let got = compute_r_via_poly_division::<F, D>(&m, &z, &y)
            .expect("ring-switch CRT+NTT path should dispatch for D=64");
        assert_eq!(got, expected);
    }

    fn direct_relation_claim<F: FieldCore + FromSmallInt>(
        w_compact: &[i8],
        alpha_evals_y: &[F],
        m_evals_x: &[F],
        live_x_cols: usize,
    ) -> F {
        (0..live_x_cols).fold(F::zero(), |acc_x, x| {
            let column_start = x * alpha_evals_y.len();
            let y_eval = alpha_evals_y
                .iter()
                .enumerate()
                .fold(F::zero(), |acc_y, (y, &alpha)| {
                    acc_y + F::from_i64(w_compact[column_start + y] as i64) * alpha
                });
            acc_x + y_eval * m_evals_x[x]
        })
    }

    #[test]
    fn full_root_rows_match_direct_relation_claim() {
        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let lp = Cfg::commitment_layout(NV).expect("lp");

        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let (commitment, batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            lp.r_vars,
            lp.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) =
            poly.evaluate_and_fold(&ring_opening_point.b, &ring_opening_point.a, lp.block_len);

        let mut transcript = Blake2bTranscript::<F>::new(b"ring-switch-row-regression");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &[1usize],
            lp.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![F::one()],
            setup.expanded.seed.max_stride,
        )
        .expect("quadratic equation");

        let w = ring_switch_build_w::<F, D>(&mut quad_eq, &setup.expanded, &setup.ntt_shared, &lp)
            .expect("ring-switch witness");
        let (w_compact, _col_bits, ring_bits) =
            build_w_evals_compact(w.as_i8_digits(), D).expect("compact witness");
        let live_x_cols = w_compact.len() >> ring_bits;

        let alpha = F::from_u64(17);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = lp.m_row_count(1, 1);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;

        for row in 0..rows {
            let tau1: Vec<F> = (0..num_i)
                .map(|bit| {
                    if (row >> bit) & 1 == 1 {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect();
            let m_evals_x = compute_m_evals_x::<F, D>(
                &setup.expanded,
                &[quad_eq.opening_point().clone()],
                &[0usize],
                &quad_eq.challenges,
                alpha,
                &alpha_evals_y,
                &lp,
                &tau1,
                &[1usize],
                &[F::one()],
                1,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &alpha_evals_y, &m_evals_x, live_x_cols);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                &quad_eq.v,
                &commitment.u,
                std::slice::from_ref(&y_ring),
            );
            assert_eq!(got, expected, "row {row} mismatch");
        }
    }

    #[test]
    fn asymmetric_centering_decompose_roundtrip() {
        use akita_types::digit_math::compute_num_digits_full_field;

        type F = fp128::Field;
        const D: usize = 64;

        let mut rng = rand::thread_rng();

        for log_basis in [2u32, 3, 4, 5, 6] {
            let field_bits = 128u32;
            let num_digits = compute_num_digits_full_field(field_bits, log_basis);

            let ring = CyclotomicRing::<F, D>::random(&mut rng);

            let mut digits = vec![CyclotomicRing::<F, D>::zero(); num_digits];
            ring.balanced_decompose_pow2_into(&mut digits, log_basis);
            let recomposed = CyclotomicRing::gadget_recompose_pow2(&digits, log_basis);
            assert_eq!(
                ring, recomposed,
                "field-element roundtrip failed for log_basis={log_basis}, num_digits={num_digits}"
            );

            let mut i8_digits = vec![[0i8; D]; num_digits];
            ring.balanced_decompose_pow2_i8_into(&mut i8_digits, log_basis);
            let recomposed_i8 = CyclotomicRing::gadget_recompose_pow2_i8(&i8_digits, log_basis);
            assert_eq!(
                ring, recomposed_i8,
                "i8 roundtrip failed for log_basis={log_basis}, num_digits={num_digits}"
            );
        }
    }

    #[test]
    fn prepared_m_eval_matches_materialized() {
        use akita_sumcheck::multilinear_eval;

        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let level_params = Cfg::commitment_layout(NV).expect("commitment layout");

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let (commitment, batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            level_params.r_vars,
            level_params.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            level_params.block_len,
        );

        let mut transcript = Blake2bTranscript::<F>::new(b"prepared-m-eval-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point.clone()],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &[1usize],
            level_params.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![F::one()],
            setup.expanded.seed.max_stride,
        )
        .expect("quadratic equation");

        ring_switch_build_w::<F, D>(
            &mut quad_eq,
            &setup.expanded,
            &setup.ntt_shared,
            &level_params,
        )
        .expect("ring-switch witness");

        let alpha = F::from_u64(42);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = level_params.m_row_count(1, 1);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let m_evals_x = compute_m_evals_x::<F, D>(
            &setup.expanded,
            &[ring_opening_point.clone()],
            &[0usize],
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            &tau1,
            &[1usize],
            &[F::one()],
            1,
        )
        .expect("m evals (materialized)");

        let x_challenges: Vec<F> = (0..m_evals_x.len().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let expected = multilinear_eval(&m_evals_x, &x_challenges).expect("multilinear_eval");

        let prepared = prepare_m_eval::<F, D>(
            &quad_eq.challenges,
            alpha,
            &level_params,
            &tau1,
            &[1usize],
            &[F::one()],
            1,
            1,
            &[],
        )
        .expect("prepare_m_eval");

        let got = prepared
            .eval_at_point::<D>(
                &x_challenges,
                &setup.expanded,
                std::slice::from_ref(&ring_opening_point),
                alpha,
            )
            .expect("eval_at_point");

        assert_eq!(
            got, expected,
            "PreparedMEval::eval_at_point must match materialized multilinear_eval"
        );
    }
}
