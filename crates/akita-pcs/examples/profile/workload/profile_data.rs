use akita_config::CommitmentConfig;
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootProvePoly};
use akita_prover::CpuBackend;
use akita_prover::{OneHotIndex, OneHotPoly};
use akita_types::{
    lagrange_weights, reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    BasisMode, CommittedGroupParams,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub(super) fn make_profile_onehot_poly<Cfg>(
    num_vars: usize,
    seed: u64,
) -> OneHotPoly<Cfg::Field, u8>
where
    Cfg: CommitmentConfig,
{
    let total_field = 1usize << num_vars;
    let onehot_k = onehot_k_for_num_vars::<Cfg>(num_vars);
    assert!(
        onehot_k <= usize::from(u8::MAX) + 1,
        "profile u8 one-hot fixture cannot represent chunk size {onehot_k}"
    );
    let total_chunks = total_field / onehot_k;
    assert_eq!(total_chunks * onehot_k, total_field);

    let mut rng = StdRng::seed_from_u64(seed);
    let indices = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    OneHotPoly::<Cfg::Field, u8>::new(onehot_k, indices).expect("profile onehot poly")
}

pub(crate) fn onehot_k_for_num_vars<Cfg: CommitmentConfig>(nv: usize) -> usize {
    let source_chunk_size = akita_config::unit_onehot_source_chunk_size::<Cfg>()
        .expect("one-hot profile requires a unit-one-hot commitment config");
    let max_supported_log_k = source_chunk_size.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        source_chunk_size
    } else {
        1usize << nv
    }
}

pub(super) fn random_claim_point<FF, E>(nv: usize, rng: &mut StdRng) -> Vec<E>
where
    FF: CanonicalEncoding + Field,
    E: ExtField<FF>,
{
    (0..nv)
        .map(|_| {
            let limbs = (0..E::DEGREE)
                .map(|_| FF::from_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

pub(super) fn degree_one_claim_point_to_base<FF, E>(point: &[E]) -> Option<Vec<FF>>
where
    FF: Field,
    E: ExtField<FF>,
{
    (E::DEGREE == 1).then(|| {
        point
            .iter()
            .map(|coord| coord.to_base_vec()[0])
            .collect::<Vec<_>>()
    })
}

pub(super) fn onehot_lagrange_opening<FF, E, I>(poly: &OneHotPoly<FF, I>, point: &[E]) -> E
where
    FF: Field,
    E: ExtField<FF>,
    I: OneHotIndex,
{
    let onehot_k = poly.onehot_k();
    assert!(onehot_k.is_power_of_two());
    assert_eq!(poly.indices().len() * onehot_k, 1usize << point.len());

    let low_vars = onehot_k.trailing_zeros() as usize;
    let low_weights = lagrange_weights(&point[..low_vars]).expect("valid low opening point");
    let high_point = &point[low_vars..];
    let mut high_weight = high_point
        .iter()
        .copied()
        .map(|r| E::one() - r)
        .fold(E::one(), |acc, value| acc * value);
    let transitions = high_point
        .iter()
        .copied()
        .map(|r| {
            let one_minus_r = E::one() - r;
            let to_one = r * one_minus_r
                .inverse()
                .expect("non-Boolean random opening point");
            let to_zero = one_minus_r * r.inverse().expect("non-Boolean random opening point");
            (to_one, to_zero)
        })
        .collect::<Vec<_>>();
    let mut opening = E::zero();
    let mut gray_index = 0usize;
    for step in 0..poly.indices().len() {
        if let Some(hot_idx) = poly.indices()[gray_index] {
            opening += high_weight * low_weights[hot_idx.as_usize()];
        }
        let next_step = step + 1;
        if next_step == poly.indices().len() {
            break;
        }
        let next_gray = next_step ^ (next_step >> 1);
        let flipped_bit = (gray_index ^ next_gray).trailing_zeros() as usize;
        high_weight *= if next_gray & (1usize << flipped_bit) == 0 {
            transitions[flipped_bit].1
        } else {
            transitions[flipped_bit].0
        };
        gray_index = next_gray;
    }
    opening
}

pub(super) fn opening_from_poly<'a, FF, const D: usize, P>(
    poly: &'a P,
    point: &[FF],
    layout: &CommittedGroupParams,
    basis: BasisMode,
) -> FF
where
    FF: CanonicalEncoding + Field,
    P: RootProvePoly<FF, D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, FF, D>,
{
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.position_index_bits() + layout.block_index_bits();
    assert!(
        point.len() <= target_num_vars,
        "opening point length {} exceeds target root arity {}",
        point.len(),
        target_num_vars
    );
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, FF::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.blocks().positions_per_block,
        layout.blocks().live_blocks,
        basis,
    )
    .expect("opening point shape should match layout");

    let opening = OpeningFoldKernel::<P::OpeningView<'a>, FF, D>::evaluate_and_fold(
        &CpuBackend::DEFAULT,
        None,
        poly.opening_view().expect("opening view"),
        OpeningFoldPlan::Base {
            live_block_weights: &ring_opening_point.live_block_weights,
            position_weights: &ring_opening_point.position_weights,
            num_positions_per_block: layout.blocks().positions_per_block,
        },
    )
    .expect("evaluate_and_fold");
    let folded_ring = opening.eval;
    let packed_inner = reduce_inner_opening_to_ring_element::<FF, D>(inner_point, basis)
        .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}
