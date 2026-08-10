//! Tensor-factor traversal for runtime-selected packed field kernels.

use super::runtime_common::{
    read_packed_fp_ext2, read_packed_fp_ext4, sum_fp64_product_round_lanes,
    sum_product_round_lanes, write_packed_fp_ext2, write_packed_fp_ext4, CoefficientSlices,
    Fp64CoefficientSlices,
};
use super::{
    Fp32TensorFactorRoundOutput, Fp64TensorFactorRoundOutput, PackedField, PackedFpExt2,
    PackedFpExt4,
};
use crate::{Fp32, Fp64, FpExt2, FpExt2Config, FpExt4};

/// Materialize one fp32 transparent tensor factor and compute its first
/// product round in the same coefficient-first traversal.
#[inline(always)]
pub(super) unsafe fn materialize_tensor_factor_and_compute_product_round_packed<const P: u32, PF>(
    witness: CoefficientSlices<'_, P>,
    equality_inner: CoefficientSlices<'_, P>,
    equality_outer: &[FpExt4<Fp32<P>>],
    zero_weights: [FpExt4<Fp32<P>>; 4],
    one_weights: [FpExt4<Fp32<P>>; 4],
) -> Fp32TensorFactorRoundOutput<P>
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let inner_len = equality_inner[0].len();
    assert!(inner_len > 0 && inner_len.is_multiple_of(PF::WIDTH));
    assert!(equality_inner
        .iter()
        .all(|coefficient| coefficient.len() == inner_len));
    assert!(!equality_outer.is_empty());
    let suffix_len = inner_len
        .checked_mul(equality_outer.len())
        .expect("tensor factor suffix length overflow");
    let table_len = suffix_len
        .checked_mul(2)
        .expect("tensor factor table length overflow");
    assert!(witness
        .iter()
        .all(|coefficient| coefficient.len() == table_len));

    let stored_len = table_len
        .checked_mul(4)
        .expect("tensor factor coefficient storage length overflow");
    let mut output = Box::<[Fp32<P>]>::new_uninit_slice(stored_len);
    let output_base = output.as_mut_ptr().cast::<Fp32<P>>();
    let output_coefficients =
        std::array::from_fn(|coefficient| unsafe { output_base.add(coefficient * table_len) });
    let witness = witness.map(|slice| slice.as_ptr());
    let equality_inner = equality_inner.map(|slice| slice.as_ptr());
    let zero_weights = zero_weights.map(PackedFpExt4::<Fp32<P>, PF>::broadcast);
    let one_weights = one_weights.map(PackedFpExt4::<Fp32<P>, PF>::broadcast);
    let mut constant = PackedFpExt4::<Fp32<P>, PF>::broadcast(FpExt4::zero());
    let mut quadratic = PackedFpExt4::<Fp32<P>, PF>::broadcast(FpExt4::zero());

    for (outer_row, &outer) in equality_outer.iter().enumerate() {
        let outer = PackedFpExt4::<Fp32<P>, PF>::broadcast(outer);
        let block_start = outer_row * inner_len;
        for inner_row in (0..inner_len).step_by(PF::WIDTH) {
            let row = block_start + inner_row;
            let inner = unsafe { read_packed_fp_ext4::<P, PF>(equality_inner, inner_row) };
            let suffix = outer * inner;
            let factor_zero = packed_linear_map(suffix, zero_weights);
            let factor_one = packed_linear_map(suffix, one_weights);
            let witness_zero = unsafe { read_packed_fp_ext4::<P, PF>(witness, row) };
            let witness_one = unsafe { read_packed_fp_ext4::<P, PF>(witness, row + suffix_len) };

            constant = constant + witness_zero * factor_zero;
            quadratic = quadratic + (witness_one - witness_zero) * (factor_one - factor_zero);
            unsafe {
                write_packed_fp_ext4(output_coefficients, row, factor_zero);
                write_packed_fp_ext4(output_coefficients, row + suffix_len, factor_one);
            }
        }
    }

    let round = sum_product_round_lanes(constant, quadratic);
    // SAFETY: the nested outer/inner traversal covers each row once, and both
    // factor branches write all four coefficient slabs for that row.
    let output = unsafe { output.assume_init() };
    (output, round)
}

#[inline(always)]
fn packed_linear_map<const P: u32, PF>(
    value: PackedFpExt4<Fp32<P>, PF>,
    weights: [PackedFpExt4<Fp32<P>, PF>; 4],
) -> PackedFpExt4<Fp32<P>, PF>
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    PackedFpExt4::new(std::array::from_fn(|output_coefficient| {
        PF::sum_four_products(
            value.coeffs,
            std::array::from_fn(|input_coefficient| {
                weights[input_coefficient].coeffs[output_coefficient]
            }),
        )
    }))
}

/// Materialize one fp64 transparent tensor factor and compute its first
/// product round in the same coefficient-first traversal.
#[inline(always)]
pub(super) unsafe fn materialize_tensor_factor_and_compute_product_round_fp_ext2_fp64_packed<
    const P: u64,
    C,
    PF,
>(
    witness: Fp64CoefficientSlices<'_, P>,
    equality_inner: Fp64CoefficientSlices<'_, P>,
    equality_outer: &[FpExt2<Fp64<P>, C>],
    zero_weights: [FpExt2<Fp64<P>, C>; 2],
    one_weights: [FpExt2<Fp64<P>, C>; 2],
) -> Fp64TensorFactorRoundOutput<P, C>
where
    C: FpExt2Config<Fp64<P>> + 'static,
    PF: PackedField<Scalar = Fp64<P>>,
{
    let inner_len = equality_inner[0].len();
    assert!(inner_len > 0 && inner_len.is_multiple_of(PF::WIDTH));
    assert!(equality_inner
        .iter()
        .all(|coefficient| coefficient.len() == inner_len));
    assert!(!equality_outer.is_empty());
    let suffix_len = inner_len
        .checked_mul(equality_outer.len())
        .expect("tensor factor suffix length overflow");
    let table_len = suffix_len
        .checked_mul(2)
        .expect("tensor factor table length overflow");
    assert!(witness
        .iter()
        .all(|coefficient| coefficient.len() == table_len));

    let stored_len = table_len
        .checked_mul(2)
        .expect("tensor factor coefficient storage length overflow");
    let mut output = Box::<[Fp64<P>]>::new_uninit_slice(stored_len);
    let output_base = output.as_mut_ptr().cast::<Fp64<P>>();
    let output_coefficients =
        std::array::from_fn(|coefficient| unsafe { output_base.add(coefficient * table_len) });
    let witness = witness.map(|slice| slice.as_ptr());
    let equality_inner = equality_inner.map(|slice| slice.as_ptr());
    let zero_weights = zero_weights.map(PackedFpExt2::<Fp64<P>, C, PF>::broadcast);
    let one_weights = one_weights.map(PackedFpExt2::<Fp64<P>, C, PF>::broadcast);
    let zero = FpExt2::<Fp64<P>, C>::zero();
    let mut constant = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(zero);
    let mut quadratic = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(zero);

    for (outer_row, &outer) in equality_outer.iter().enumerate() {
        let outer = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(outer);
        let block_start = outer_row * inner_len;
        for inner_row in (0..inner_len).step_by(PF::WIDTH) {
            let row = block_start + inner_row;
            let inner = unsafe { read_packed_fp_ext2::<P, C, PF>(equality_inner, inner_row) };
            let suffix = outer * inner;
            let factor_zero = packed_fp64_linear_map(suffix, zero_weights);
            let factor_one = packed_fp64_linear_map(suffix, one_weights);
            let witness_zero = unsafe { read_packed_fp_ext2::<P, C, PF>(witness, row) };
            let witness_one = unsafe { read_packed_fp_ext2::<P, C, PF>(witness, row + suffix_len) };

            constant = constant + witness_zero * factor_zero;
            quadratic = quadratic + (witness_one - witness_zero) * (factor_one - factor_zero);
            unsafe {
                write_packed_fp_ext2(output_coefficients, row, factor_zero);
                write_packed_fp_ext2(output_coefficients, row + suffix_len, factor_one);
            }
        }
    }

    let round = sum_fp64_product_round_lanes(constant, quadratic);
    // SAFETY: the nested outer/inner traversal covers each row once, and both
    // factor branches write both coefficient slabs for that row.
    let output = unsafe { output.assume_init() };
    (output, round)
}

#[inline(always)]
fn packed_fp64_linear_map<const P: u64, C, PF>(
    value: PackedFpExt2<Fp64<P>, C, PF>,
    weights: [PackedFpExt2<Fp64<P>, C, PF>; 2],
) -> PackedFpExt2<Fp64<P>, C, PF>
where
    C: FpExt2Config<Fp64<P>> + 'static,
    PF: PackedField<Scalar = Fp64<P>>,
{
    PackedFpExt2::new(
        value.c0 * weights[0].c0 + value.c1 * weights[1].c0,
        value.c0 * weights[0].c1 + value.c1 * weights[1].c1,
    )
}
