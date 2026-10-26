use super::*;

#[target_feature(enable = "avx2")]
unsafe fn normalize_lanes(domain: &TrinomialLimbDomain, input: [i32; LANES]) -> [i32; LANES] {
    let mut output = [0; LANES];
    let shift = 31 - domain.one.quotient.trailing_zeros();
    // SAFETY: AVX2 is established on entry; complete initialized arrays supply
    // all eight lanes. The caller supplies values inside the admitted B4 bound.
    unsafe {
        let result = reduce_one(
            _mm256_loadu_si256(input.as_ptr().cast()),
            _mm_cvtsi32_si128(shift as i32),
            _mm256_set1_epi32(1 << (shift - 1)),
            _mm256_set1_epi32(domain.prime() as i32),
        );
        _mm256_storeu_si256(output.as_mut_ptr().cast(), result);
    }
    output
}

#[test]
fn avx2_normalization_matches_i128_at_every_quotient_transition() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let q = domain.one.quotient;
        let shift = 31 - q.trailing_zeros();
        let bias = 1i64 << (shift - 1);
        let bound = super::super::arithmetic::forward_bounds(prime)[4];
        let mut operands = vec![-bound as i32, bound as i32];
        // The rounded quotient changes exactly at t*2^shift-bias. Include
        // both neighbors and the transition itself over the whole lazy range.
        for t in ((-bound + bias) >> shift)..=((bound + bias) >> shift) + 1 {
            let transition = (t << shift) - bias;
            for value in [transition - 1, transition, transition + 1] {
                if (-bound..=bound).contains(&value) {
                    operands.push(value as i32);
                }
            }
        }
        // Rotate every operand through every lane, including both B4 extremes.
        for offset in 0..operands.len() {
            let input = std::array::from_fn(|lane| operands[(offset + lane) % operands.len()]);
            // SAFETY: Runtime detection above establishes AVX2.
            let output = unsafe { normalize_lanes(&domain, input) };
            for (value, actual) in input.into_iter().zip(output) {
                let quotient = (i128::from(value) * i128::from(q) + (1i128 << 30)) >> 31;
                let expected = i128::from(value) - quotient * i128::from(prime);
                assert_eq!(i128::from(actual), expected, "prime={prime}, value={value}");
                assert_eq!(actual, domain.arithmetic.mul(value, domain.one));
            }
        }
    }
}
