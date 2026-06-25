//! Shared test fixture for the D=32 recursive-multigroup ring-switch row
//! evaluation. `nv = 32` in `fp128_d32_onehot.rs` includes repeated compact
//! recursive levels with this real D=32 shape, exercised by both the
//! structured-slice and zk-blinding equivalence tests.

use crate::protocol::ring_switch::{PreparedChallengeEvals, RingSwitchDeferredRowEval};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_field::{CanonicalField, Prime128OffsetA7F7};
use akita_types::{
    LevelParams, MRowLayout, OpeningBatchShape, RingMultiplierOpeningPoint, RingOpeningPoint,
    RingRelationInstance, RingRelationSegmentLayout, SisModulusFamily,
};

pub(crate) type FixtureField = Prime128OffsetA7F7;
pub(crate) const FIXTURE_D: usize = 32;

pub(crate) fn scalar(value: u128) -> FixtureField {
    FixtureField::from_canonical_u128_reduced(value)
}

fn fixture_lp() -> LevelParams {
    LevelParams::params_only(
        SisModulusFamily::Q128,
        FIXTURE_D,
        5,
        2,
        2,
        2,
        SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![1],
        },
    )
    .with_decomp(2, 3, 1, 26, 512 * 8)
    .expect("recursive d32 fixture lp")
}

fn ring_relation_segment_layout_for_opening_shape(
    lp: &LevelParams,
    m_row_layout: MRowLayout,
    num_polys_per_segment: &[usize],
) -> Result<RingRelationSegmentLayout, AkitaError> {
    let opening_batch = OpeningBatchShape::from_commitment_groups(32, num_polys_per_segment)?;
    let opening_point = RingOpeningPoint {
        a: vec![FixtureField::zero(); lp.block_len],
        b: vec![FixtureField::zero(); lp.num_blocks],
    };
    let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
    let num_claims = opening_batch.num_claims();
    let challenges = akita_challenges::Challenges::Sparse {
        challenges: Vec::new(),
        num_blocks_per_claim: lp.num_blocks,
        num_claims,
    };
    let instance = RingRelationInstance::<FixtureField, FIXTURE_D>::new(
        m_row_layout,
        challenges,
        opening_point,
        ring_multiplier_point,
        opening_batch,
        vec![FixtureField::zero(); num_claims],
        vec![CyclotomicRing::<FixtureField, FIXTURE_D>::zero(); num_claims],
        vec![CyclotomicRing::<FixtureField, FIXTURE_D>::zero(); num_claims],
        Vec::new(),
    )?;
    instance.segment_layout(lp)
}

/// Build the canonical D=32 recursive-multigroup prepared row evaluation.
pub(crate) fn recursive_d32_prepared() -> RingSwitchDeferredRowEval<FixtureField> {
    let num_blocks = 8usize;
    let block_len = 512usize;
    let log_basis = 5u32;
    let depth_open = 26usize;
    let depth_commit = 1usize;
    let depth_fold = 4usize;
    let n_a = 2usize;
    let n_b = 2usize;
    let n_d = 2usize;
    let num_polys_per_segment = vec![2usize, 1usize];
    let num_points = num_polys_per_segment.len();
    let num_claims = 3usize;
    let num_public_rows = num_points;
    let total_blocks = num_blocks * num_claims;
    let rows = 1 + num_public_rows + n_d + n_b * num_points + n_a;
    let inner_width = block_len * depth_commit;
    let num_t_vectors = num_polys_per_segment.iter().sum();

    let witness_segment_layout = ring_relation_segment_layout_for_opening_shape(
        &fixture_lp(),
        MRowLayout::WithDBlock,
        &num_polys_per_segment,
    )
    .expect("witness segment layout");

    #[cfg(feature = "zk")]
    let b_blinding_digit_planes_per_point =
        akita_types::lhl_blinding::blinding_digit_plane_count::<FixtureField>(n_b, FIXTURE_D, log_basis);

    RingSwitchDeferredRowEval {
        c_alphas: PreparedChallengeEvals::Flat(
            (0..total_blocks)
                .map(|idx| scalar(3_000 + idx as u128))
                .collect(),
        ),
        eq_tau1: (0..rows.next_power_of_two())
            .map(|idx| scalar(4_000 + idx as u128))
            .collect(),
        num_t_vectors,
        num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        #[cfg(feature = "zk")]
        d_blinding_segment_len: akita_types::lhl_blinding::blinding_digit_plane_count::<FixtureField>(
            n_d, FIXTURE_D, log_basis,
        ),
        #[cfg(feature = "zk")]
        b_blinding_digit_planes_per_point,
        #[cfg(feature = "zk")]
        b_blinding_segment_len: num_points * b_blinding_digit_planes_per_point,
        block_len,
        inner_width,
        log_basis,
        n_a,
        n_d,
        m_row_layout: MRowLayout::WithDBlock,
        n_b,
        tier_split: 1,
        n_f: 0,
        num_points,
        rows,
        claim_to_t_vector: vec![1, 2, 0],
        num_polys_per_segment: num_polys_per_segment,
        num_public_rows,
        gamma: (0..num_claims)
            .map(|idx| scalar(5_000 + idx as u128))
            .collect(),
        claim_to_opening_point: vec![1, 0, 1],
        witness_segment_layout,
    }
}
