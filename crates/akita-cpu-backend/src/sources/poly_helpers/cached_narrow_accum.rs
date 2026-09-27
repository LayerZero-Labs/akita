//! Output-tiled narrow accumulation for cached signed-byte digit planes.

use akita_challenges::SparseChallenge;

const OUTPUT_TILE: usize = 8;

/// Accumulate one sparse negacyclic product into a proven-safe `i16` sum.
///
/// The caller proves that the incoming accumulator plus the challenge's full
/// contribution remains in `i16`. `scratch` is reused by the caller and is
/// overwritten with `[-digit_plane, digit_plane]` on each invocation.
#[inline]
pub(super) fn sparse_mul_acc<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &SparseChallenge,
    acc: &mut [i16; D],
    scratch: &mut [[i16; D]; 2],
) {
    debug_assert!(D > 0);
    debug_assert_eq!(challenge.positions.len(), challenge.coeffs.len());
    debug_assert!(challenge
        .positions
        .iter()
        .all(|&position| position < D as u32));

    if D > 128 {
        super::narrow_accum::sparse_mul_acc(digit_plane, challenge, acc);
        return;
    }

    for (index, &value) in digit_plane.iter().enumerate() {
        let value = i16::from(value);
        scratch[0][index] = -value;
        scratch[1][index] = value;
    }

    let doubled_source = scratch.as_flattened();
    let mut tile_start = 0;
    while tile_start + OUTPUT_TILE <= D {
        let mut tile_acc = [0i16; OUTPUT_TILE];
        tile_acc.copy_from_slice(&acc[tile_start..tile_start + OUTPUT_TILE]);

        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            let start = D - position as usize + tile_start;
            let scale = i16::from(coefficient);
            for (dst, &source) in tile_acc
                .iter_mut()
                .zip(&doubled_source[start..start + OUTPUT_TILE])
            {
                *dst += source * scale;
            }
        }

        acc[tile_start..tile_start + OUTPUT_TILE].copy_from_slice(&tile_acc);
        tile_start += OUTPUT_TILE;
    }

    if tile_start < D {
        let tile_len = D - tile_start;
        let mut tile_acc = [0i16; OUTPUT_TILE];
        tile_acc[..tile_len].copy_from_slice(&acc[tile_start..]);

        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            let start = D - position as usize + tile_start;
            let scale = i16::from(coefficient);
            for (dst, &source) in tile_acc[..tile_len]
                .iter_mut()
                .zip(&doubled_source[start..start + tile_len])
            {
                *dst += source * scale;
            }
        }

        acc[tile_start..].copy_from_slice(&tile_acc[..tile_len]);
    }
}

#[cfg(test)]
mod tests {
    use super::sparse_mul_acc;
    use akita_challenges::SparseChallenge;

    fn check_scalar_convolution<const D: usize>() {
        let digit_plane = std::array::from_fn(|index| {
            if index == 0 {
                i8::MIN
            } else if index + 1 == D {
                i8::MAX
            } else {
                ((index * 17 % 15) as i16 - 7) as i8
            }
        });
        let positions = vec![0, (D - 1) as u32, 4, 4, 7];
        let challenge = SparseChallenge {
            positions: positions.into(),
            coeffs: vec![1, -1, 2, -2, i8::MIN].into(),
        };
        let mut actual = std::array::from_fn(|index| index as i16 - 5);
        let mut expected = actual.map(i32::from);

        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            let position = position as usize;
            for (output_index, output) in expected.iter_mut().enumerate() {
                let (source_index, sign) = if output_index < position {
                    (output_index + D - position, -1)
                } else {
                    (output_index - position, 1)
                };
                *output += i32::from(coefficient) * i32::from(digit_plane[source_index]) * sign;
            }
        }

        let mut scratch = [[0i16; D]; 2];
        sparse_mul_acc(&digit_plane, &challenge, &mut actual, &mut scratch);

        let expected = expected
            .into_iter()
            .map(|value| i16::try_from(value).expect("test convolution stays in i16"))
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected);
    }

    fn check_d128_production_challenge() {
        const D: usize = 128;
        let digit_plane = std::array::from_fn(|index| ((index * 11 % 7) as i8) - 3);
        let challenge = SparseChallenge {
            positions: (0..31)
                .map(|term| (term * 37 % D) as u32)
                .collect::<Vec<_>>()
                .into(),
            coeffs: (0..31)
                .map(|term| if term % 2 == 0 { 1 } else { -1 })
                .collect::<Vec<_>>()
                .into(),
        };
        let mut actual = std::array::from_fn(|index| (index % 9) as i16 - 4);
        let mut expected = actual.map(i32::from);

        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            let position = position as usize;
            for (output_index, output) in expected.iter_mut().enumerate() {
                let (source_index, sign) = if output_index < position {
                    (output_index + D - position, -1)
                } else {
                    (output_index - position, 1)
                };
                *output += i32::from(coefficient) * i32::from(digit_plane[source_index]) * sign;
            }
        }

        let mut scratch = [[0i16; D]; 2];
        sparse_mul_acc(&digit_plane, &challenge, &mut actual, &mut scratch);

        let expected = expected
            .into_iter()
            .map(|value| i16::try_from(value).expect("production challenge stays in i16"))
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected);
    }

    #[test]
    fn cached_narrow_tiles_and_fallback_match_scalar_negacyclic_convolution() {
        check_scalar_convolution::<11>();
        check_scalar_convolution::<16>();
        check_d128_production_challenge();
        check_scalar_convolution::<256>();
    }
}
