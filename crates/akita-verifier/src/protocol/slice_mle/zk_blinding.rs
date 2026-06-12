use akita_algebra::offset_eq::eval_offset_eq_tensor;
use akita_algebra::ring::eval_ring_at_pows;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::AkitaExpandedSetup;

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

/// ZK B-blinding contribution. See `specs/optimized_verifier.md`.
pub(crate) fn compute_b_blinding_part<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let group_stride = prepared.b_blinding_digit_planes_per_point;
    if group_stride == 0 {
        return Ok(E::zero());
    }
    let _span = tracing::info_span!("b_blinding").entered();

    let layout = prepared.witness_segment_layout;
    let b_start = 1 + prepared.num_public_rows + prepared.n_d_active();

    // Mirror the prover's group-local B input layout:
    // `[group t_hat || group blinding]` for each commitment group. The
    // witness has one blinding segment per point; every segment reuses the
    // same stored per-commitment zkB row view with fresh digits.
    let b_blinding_segment_len = prepared.b_blinding_segment_len;
    let b_zk_view = setup
        .zk_b_matrix()
        .ring_view::<D>(prepared.n_b, group_stride)?;
    let b_zk = b_zk_view.as_slice();
    let b_zk_stride = b_zk_view.num_cols();
    let b_blinding_segment: Vec<E> = cfg_into_iter!(0..b_blinding_segment_len)
        .map(|idx| {
            let point_idx = idx / group_stride;
            let local = idx % group_stride;
            let commitment_weights = &prepared.eq_tau1
                [(b_start + point_idx * prepared.n_b)..(b_start + (point_idx + 1) * prepared.n_b)];
            let mut acc = E::zero();
            for (row_idx, &eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc +=
                        eq_i * eval_ring_at_pows(&b_zk[row_idx * b_zk_stride + local], alpha_pows);
                }
            }
            acc
        })
        .collect();
    eval_offset_eq_tensor(
        full_vec_randomness,
        layout.b_blinding_offset,
        E::one(),
        &[b_blinding_segment.as_slice()],
    )
}

/// ZK D-blinding contribution. See `specs/optimized_verifier.md`.
pub(crate) fn compute_d_blinding_part<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let d_blinding_segment_len = prepared.d_blinding_segment_len;
    if d_blinding_segment_len == 0 {
        return Ok(E::zero());
    }
    let _span = tracing::info_span!("d_blinding").entered();

    let layout = prepared.witness_segment_layout;
    let d_start = 1 + prepared.num_public_rows;
    let n_d_active = prepared.n_d_active();
    let d_weights = &prepared.eq_tau1[d_start..(d_start + n_d_active)];
    let d_zk_view = setup
        .zk_d_matrix()
        .ring_view::<D>(prepared.n_d, d_blinding_segment_len)?;
    let d_zk = d_zk_view.as_slice();
    let d_zk_stride = d_zk_view.num_cols();

    let d_blinding_segment: Vec<E> = cfg_into_iter!(0..d_blinding_segment_len)
        .map(|local| {
            let mut acc = E::zero();
            for (row_idx, &eq_i) in d_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc +=
                        eq_i * eval_ring_at_pows(&d_zk[row_idx * d_zk_stride + local], alpha_pows);
                }
            }
            acc
        })
        .collect();
    eval_offset_eq_tensor(
        full_vec_randomness,
        layout.d_blinding_offset,
        E::one(),
        &[d_blinding_segment.as_slice()],
    )
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{
        recursive_d32_prepared, scalar as f, FixtureField as F, FIXTURE_D as D,
    };
    use super::*;
    use akita_algebra::offset_eq::eq_eval_at_index;
    use akita_algebra::ring::scalar_powers;
    use akita_algebra::CyclotomicRing;
    use akita_types::{AkitaSetupSeed, FlatMatrix};

    struct ZkFixture {
        prepared: RingSwitchDeferredRowEval<F>,
        setup: AkitaExpandedSetup<F>,
        full_vec_randomness: Vec<F>,
        alpha: F,
    }

    fn fixture() -> ZkFixture {
        let prepared = recursive_d32_prepared();
        let bits = prepared
            .witness_segment_layout
            .offset_r
            .next_power_of_two()
            .trailing_zeros() as usize;

        let max_zk_b_len = prepared.n_b * prepared.b_blinding_digit_planes_per_point;
        let max_zk_d_len = prepared.n_d * prepared.d_blinding_segment_len;
        let t_cols_per_claim = prepared.num_blocks * prepared.n_a * prepared.depth_open;
        let max_b_local_col = prepared
            .num_polys_per_commitment_group
            .iter()
            .map(|&count| count * t_cols_per_claim)
            .max()
            .unwrap_or(0);
        let max_d_local_col = prepared.depth_open * prepared.num_blocks * prepared.num_claims;
        let max_setup_len = (prepared.n_b * max_b_local_col)
            .max(prepared.n_d * max_d_local_col)
            .max(prepared.n_a * prepared.inner_width);

        let matrix_entries: Vec<CyclotomicRing<F, D>> = (0..max_setup_len)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coeff| {
                    f(1_000 + (idx * D + coeff) as u128)
                }))
            })
            .collect();
        let zk_b_entries: Vec<CyclotomicRing<F, D>> = (0..max_zk_b_len)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coeff| {
                    f(10_000 + (idx * D + coeff) as u128)
                }))
            })
            .collect();
        let zk_d_entries: Vec<CyclotomicRing<F, D>> = (0..max_zk_d_len)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coeff| {
                    f(20_000 + (idx * D + coeff) as u128)
                }))
            })
            .collect();
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 32,
                max_num_batched_polys: prepared.num_polys_per_commitment_group.iter().sum(),
                max_num_points: prepared.num_points,
                gen_ring_dim: D,
                max_setup_len,
                max_zk_b_len,
                max_zk_d_len,
                public_matrix_seed: [9u8; 32],
            },
            FlatMatrix::from_ring_slice::<D>(&matrix_entries),
            FlatMatrix::from_ring_slice::<D>(&zk_b_entries),
            FlatMatrix::from_ring_slice::<D>(&zk_d_entries),
        );

        ZkFixture {
            prepared,
            setup,
            full_vec_randomness: (0..bits).map(|idx| f(4_000 + idx as u128)).collect(),
            alpha: f(5_000),
        }
    }

    #[test]
    fn b_blinding_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let eq: Vec<F> = (0..(1usize << fx.full_vec_randomness.len()))
            .map(|idx| eq_eval_at_index(&fx.full_vec_randomness, idx))
            .collect();
        let alpha_pows = scalar_powers(fx.alpha, D);
        let b_start = 1 + p.num_public_rows + p.n_d;
        let b_offset = p.witness_segment_layout.b_blinding_offset;
        let b_zk_view = fx
            .setup
            .zk_b_matrix()
            .ring_view::<D>(p.n_b, p.b_blinding_digit_planes_per_point)
            .unwrap();
        let b_zk_rows: Vec<_> = b_zk_view.rows().collect();

        let got =
            compute_b_blinding_part::<F, F, D>(p, &fx.full_vec_randomness, &fx.setup, &alpha_pows)
                .unwrap();
        let mut expected = F::zero();
        for idx in 0..p.b_blinding_segment_len {
            let point_idx = idx / p.b_blinding_digit_planes_per_point;
            let local = idx % p.b_blinding_digit_planes_per_point;
            let mut entry = F::zero();
            for (row_idx, row) in b_zk_rows.iter().enumerate().take(p.n_b) {
                let weight = p.eq_tau1[b_start + point_idx * p.n_b + row_idx];
                entry += weight * eval_ring_at_pows(&row[local], &alpha_pows);
            }
            expected += entry * eq[b_offset + idx];
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn d_blinding_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let eq: Vec<F> = (0..(1usize << fx.full_vec_randomness.len()))
            .map(|idx| eq_eval_at_index(&fx.full_vec_randomness, idx))
            .collect();
        let alpha_pows = scalar_powers(fx.alpha, D);
        let d_start = 1 + p.num_public_rows;
        let d_offset = p.witness_segment_layout.d_blinding_offset;
        let d_zk_view = fx
            .setup
            .zk_d_matrix()
            .ring_view::<D>(p.n_d, p.d_blinding_segment_len)
            .unwrap();
        let d_zk_rows: Vec<_> = d_zk_view.rows().collect();

        let got =
            compute_d_blinding_part::<F, F, D>(p, &fx.full_vec_randomness, &fx.setup, &alpha_pows)
                .unwrap();
        let mut expected = F::zero();
        for local in 0..p.d_blinding_segment_len {
            let mut entry = F::zero();
            for (row_idx, row) in d_zk_rows.iter().enumerate().take(p.n_d) {
                let weight = p.eq_tau1[d_start + row_idx];
                entry += weight * eval_ring_at_pows(&row[local], &alpha_pows);
            }
            expected += entry * eq[d_offset + local];
        }
        assert_eq!(got, expected);
    }
}
