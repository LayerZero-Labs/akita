use akita_algebra::offset_eq::eval_offset_eq_tensor;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::AkitaExpandedSetup;

#[cfg(test)]
use crate::protocol::ring_switch::PreparedChallengeEvals;
use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

/// ZK B-blinding contribution. See `specs/optimized_verifier.md`.
pub(crate) fn compute_b_blinding_part<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha: E,
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

    let layout = prepared.segment_layout()?;
    let alpha_pows = scalar_powers(alpha, D);
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
                        eq_i * eval_ring_at_pows(&b_zk[row_idx * b_zk_stride + local], &alpha_pows);
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
    alpha: E,
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

    let layout = prepared.segment_layout()?;
    let alpha_pows = scalar_powers(alpha, D);
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
                        eq_i * eval_ring_at_pows(&d_zk[row_idx * d_zk_stride + local], &alpha_pows);
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
    use super::*;
    use akita_algebra::offset_eq::eq_eval_at_index;
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::zk;
    use akita_types::{
        ring_relation_segment_layout_for_opening_shape, AkitaSetupSeed, FlatMatrix, LevelParams,
        MRowLayout, SisModulusFamily,
    };

    type F = Prime128OffsetA7F7;
    const D: usize = 32;

    struct ZkFixture {
        prepared: RingSwitchDeferredRowEval<F>,
        setup: AkitaExpandedSetup<F>,
        full_vec_randomness: Vec<F>,
        alpha: F,
        w_len: usize,
        t_len: usize,
    }

    fn f(value: u128) -> F {
        F::from_canonical_u128_reduced(value)
    }

    fn fixture_lp() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            5,
            2,
            2,
            2,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        )
        .with_decomp(2, 3, 1, 26, 4, 512 * 8)
        .expect("zk blinding fixture lp")
    }

    fn fixture() -> ZkFixture {
        // `nv = 32` in `fp128_d32_onehot.rs` includes repeated compact
        // recursive levels with this real D=32 shape.
        let num_blocks = 8usize;
        let num_claims = 3usize;
        let depth_open = 26usize;
        let depth_commit = 1usize;
        let depth_fold = 4usize;
        let block_len = 512usize;
        let inner_width = block_len * depth_commit;
        let log_basis = 5u32;
        let n_a = 2usize;
        let n_d = 2usize;
        let n_b = 2usize;
        let num_polys_per_point = vec![2usize, 1usize];
        let num_public_rows = 2usize;
        let num_points = num_polys_per_point.len();
        let total_blocks = num_blocks * num_claims;
        let rows = 1 + num_public_rows + n_d + n_b * num_points + n_a;

        let w_len = depth_open * total_blocks;
        let t_len = depth_open * n_a * total_blocks;
        let z_len = depth_fold * depth_commit * num_points * block_len;
        let b_blinding_digit_planes_per_point =
            zk::blinding_digit_plane_count::<F>(n_b, D, log_basis);
        let b_blinding_segment_len = num_points * b_blinding_digit_planes_per_point;
        let d_blinding_segment_len = zk::blinding_digit_plane_count::<F>(n_d, D, log_basis);
        let lp = fixture_lp();
        let witness_segment_layout = ring_relation_segment_layout_for_opening_shape::<F, D>(
            &lp,
            MRowLayout::WithDBlock,
            &num_polys_per_point,
        )
        .expect("witness segment layout");
        let b_offset = witness_segment_layout.b_blinding_offset;
        let d_offset = witness_segment_layout.d_blinding_offset;
        let total_len = witness_segment_layout.offset_r;
        let bits = total_len.next_power_of_two().trailing_zeros() as usize;
        let max_zk_b_len = n_b * b_blinding_digit_planes_per_point;
        let max_zk_d_len = n_d * d_blinding_segment_len;

        let t_cols_per_claim = num_blocks * n_a * depth_open;
        let max_b_local_col = num_polys_per_point
            .iter()
            .map(|&count| count * t_cols_per_claim)
            .max()
            .unwrap_or(0);
        let max_d_local_col = w_len;
        let max_setup_len = (n_b * max_b_local_col)
            .max(n_d * max_d_local_col)
            .max(n_a * inner_width);

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
                max_num_batched_polys: num_polys_per_point.iter().sum(),
                max_num_points: num_points,
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
        let prepared = RingSwitchDeferredRowEval {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..total_blocks)
                    .map(|idx| f(2_000 + idx as u128))
                    .collect(),
            ),
            eq_tau1: (0..rows.next_power_of_two())
                .map(|idx| f(3_000 + idx as u128))
                .collect(),
            num_t_vectors: num_polys_per_point.iter().sum(),
            num_blocks,
            num_claims,
            depth_open,
            depth_commit,
            depth_fold,
            d_blinding_segment_len,
            b_blinding_digit_planes_per_point,
            b_blinding_segment_len,
            block_len,
            inner_width,
            log_basis,
            n_a,
            n_d,
            m_row_layout: MRowLayout::WithDBlock,
            n_b,
            num_points,
            rows,
            claim_to_point_poly: vec![(0, 1), (1, 0), (0, 0)],
            num_polys_per_point,
            num_public_rows,
            gamma: vec![F::one(); num_claims],
            claim_to_point: vec![1, 0, 1],
            witness_segment_layout,
        };
        ZkFixture {
            prepared,
            setup,
            full_vec_randomness: (0..bits).map(|idx| f(4_000 + idx as u128)).collect(),
            alpha: f(5_000),
            w_len,
            t_len,
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
        let b_offset = fx.w_len + fx.t_len;
        let b_zk_view = fx
            .setup
            .zk_b_matrix()
            .ring_view::<D>(p.n_b, p.b_blinding_digit_planes_per_point)
            .unwrap();
        let b_zk_rows: Vec<_> = b_zk_view.rows().collect();

        let got =
            compute_b_blinding_part::<F, F, D>(p, &fx.full_vec_randomness, &fx.setup, fx.alpha)
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
        let d_offset = fx.w_len + fx.t_len + p.b_blinding_segment_len;
        let d_zk_view = fx
            .setup
            .zk_d_matrix()
            .ring_view::<D>(p.n_d, p.d_blinding_segment_len)
            .unwrap();
        let d_zk_rows: Vec<_> = d_zk_view.rows().collect();

        let got =
            compute_d_blinding_part::<F, F, D>(p, &fx.full_vec_randomness, &fx.setup, fx.alpha)
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
