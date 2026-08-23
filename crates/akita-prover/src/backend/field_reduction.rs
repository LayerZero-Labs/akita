use super::RecursiveWitnessFlat;
use akita_error::AkitaError;
use akita_field::{ExtField, FieldCore};
use akita_types::pack_tensor_base_lift_i8_digits;

fn tensor_extension_split<F, E>(context: &'static str) -> Result<(usize, usize), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let split_bits = E::EXT_DEGREE.trailing_zeros() as usize;
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("tensor extension width overflow".to_string()))?;
    if width != E::EXT_DEGREE || !E::EXT_DEGREE.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "tensor extension {context} requires power-of-two extension degree"
        )));
    }
    Ok((split_bits, width))
}

/// Pack a logical recursive digit witness into the canonical tensor extension
/// ring-subfield layout.
pub fn tensor_pack_recursive_witness<F, E, const D: usize>(
    logical_w: &RecursiveWitnessFlat,
) -> Result<RecursiveWitnessFlat, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (_split_bits, width) = tensor_extension_split::<F, E>("packing")?;
    let packed = pack_tensor_base_lift_i8_digits::<D>(logical_w.digits(), E::EXT_DEGREE, width)?;
    RecursiveWitnessFlat::from_tensor_packed_i8_digits(packed, logical_w.live_coeff_len())?
        .align_for_commitment_ring_dim(D)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{FpExt4, Prime32Offset99};

    #[test]
    fn recursive_tensor_pack_rejects_non_divisible_digit_count() {
        type F = Prime32Offset99;
        type E = FpExt4<F>;
        const D: usize = 32;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1, 2, 3]);

        let err = tensor_pack_recursive_witness::<F, E, D>(&witness).unwrap_err();
        assert!(matches!(
            err,
            AkitaError::InvalidSize {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn recursive_tensor_pack_preserves_logical_live_length() {
        type F = Prime32Offset99;
        type E = FpExt4<F>;
        const D: usize = 128;
        let logical_len = D + <E as ExtField<F>>::EXT_DEGREE;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; logical_len]);

        let packed = tensor_pack_recursive_witness::<F, E, D>(&witness).unwrap();

        assert_eq!(packed.live_coeff_len(), logical_len);
        assert_eq!(packed.to_i8_digits().len(), 2 * D);
        assert_eq!(packed.committed_coeff_len().unwrap(), 2 * D);
    }
}
