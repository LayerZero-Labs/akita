//! Architecture-neutral validation for balanced signed NTT digits.

/// Return whether every signed value lies in `[-bound, bound)`.
///
/// AArch64 uses its mandatory NEON unit internally. Callers do not select or
/// observe the hardware backend, so verifier-facing validation remains
/// independent of NTT benchmarking overrides.
#[must_use]
pub fn i16_values_in_balanced_range(values: &[i16], bound: i16) -> bool {
    if bound <= 0 {
        return false;
    }

    #[cfg(target_arch = "aarch64")]
    let values_valid = super::neon::i16_values_in_balanced_range(values, bound);
    #[cfg(not(target_arch = "aarch64"))]
    let values_valid = values.iter().all(|&value| value >= -bound && value < bound);
    values_valid
}

#[cfg(test)]
mod tests {
    use super::i16_values_in_balanced_range;

    #[test]
    fn balanced_i16_range_checks_vector_and_tail() {
        let mut values = [-128i16; 19];
        values[7] = 127;
        values[18] = 0;
        assert!(i16_values_in_balanced_range(&values, 128));

        values[8] = 128;
        assert!(!i16_values_in_balanced_range(&values, 128));
        values[8] = 0;
        values[18] = -129;
        assert!(!i16_values_in_balanced_range(&values, 128));

        for bound in [1i16, 2, 128, 1024, 16384] {
            for len in 0..35 {
                let values = (0..len)
                    .map(|index| {
                        let span = i32::from(bound) * 2;
                        ((index * 137 + 19) % span - i32::from(bound)) as i16
                    })
                    .collect::<Vec<_>>();
                let scalar = values.iter().all(|&value| value >= -bound && value < bound);
                assert_eq!(i16_values_in_balanced_range(&values, bound), scalar);
            }
        }
        assert!(!i16_values_in_balanced_range(&[], 0));
    }
}
