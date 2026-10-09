#![cfg(feature = "labinius")]

mod common;
use common::{binary_mle, data, profile, setup, FixedDraw, Tape, TestHost, TestPrime};

use akita_algebra::{
    binary::{
        field_switch::{batched_weights, embed_source, transparent_weight},
        BinaryField128, BinaryField162 as B, BinaryField192, PackedBinary162,
    },
    embed_scalar, pack_scalar_components, MinusTrinomial, PlusTrinomial, Prime64Offset23703,
    TrinomialModulus, TrinomialRing,
};
use akita_challenges::BinaryChallengeSampler;
use akita_labinius_prover::commit_binary_clear;
use akita_labinius_verifier::{
    endpoint::{fold_integer, left_expansion, pack_response, response_parity},
    frontend::prove_frontend,
    source::{
        challenge_binary, challenge_scalar, equality_weights, pack_source_column,
        scalar_from_binary,
    },
};
use jolt_field::Prime128OffsetA7F7;

fn frontend_oracle<H: TestHost>(variables: usize) {
    let (source, point, host_claim) = data::<H>(1 << variables, variables);
    let mut tape = Tape::default();
    let binary = prove_frontend::<H, _>(&source, &point, host_claim, &mut tape).unwrap();
    let bytes = tape.messages.concat();
    let source_bytes = if H::ROWS == 128 { 16 } else { 8 };
    let mut host_weights = vec![H::ZERO; source.len()];
    H::equality_weights(&point, &mut host_weights);
    let mut partials = vec![0u128; H::ROWS];
    for (weight, &word) in host_weights.iter().zip(&source) {
        for (row, partial) in partials.iter_mut().enumerate() {
            if weight.coordinates()[row / 64] >> (row % 64) & 1 == 1 {
                *partial ^= word.into();
            }
        }
    }
    for (row, &partial) in partials.iter().enumerate() {
        assert_eq!(
            &bytes[row * source_bytes..(row + 1) * source_bytes],
            &partial.to_le_bytes()[..source_bytes]
        );
    }
    let batch = (0..H::BATCH_BITS)
        .map(|i| Tape::draw_value(i, 0))
        .collect::<Vec<_>>();
    let batch_weights = equality_weights(&batch).unwrap();
    let mut claim = partials
        .iter()
        .zip(&batch_weights)
        .fold(B::ZERO, |sum, (&word, &weight)| {
            let field = B::from_words([word as u64, (word >> 64) as u64, 0]).unwrap();
            sum + field * weight
        });
    let mut coefficients = PackedBinary162::new();
    batched_weights::<H>(&host_weights, &batch, &mut coefficients).unwrap();
    let original_coefficients = coefficients.to_scalars();
    let mut a = source
        .iter()
        .map(|&word| embed_source::<H>(word))
        .collect::<Vec<_>>();
    let mut b = original_coefficients.clone();
    let mut cursor = H::ROWS * source_bytes;
    let mut z = Vec::new();
    for axis in 0..variables {
        let mut round = [B::ZERO; 3];
        for (aa, bb) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
            let da = aa[0] + aa[1];
            let db = bb[0] + bb[1];
            round[0] += aa[0] * bb[0];
            round[1] += aa[0] * db + da * bb[0];
            round[2] += da * db;
        }
        assert_eq!(round[1] + round[2], claim);
        assert_eq!(
            B::from_bytes(&bytes[cursor..cursor + 21]).unwrap(),
            round[0]
        );
        assert_eq!(
            B::from_bytes(&bytes[cursor + 21..cursor + 42]).unwrap(),
            round[2]
        );
        cursor += 42;
        let challenge = Tape::draw_value(H::BATCH_BITS + axis, 0);
        z.push(challenge);
        claim = round[0] + round[1] * challenge + round[2] * challenge.square();
        a = a
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[0] + pair[1]))
            .collect();
        b = b
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[0] + pair[1]))
            .collect();
    }
    assert_eq!(binary.point, z);
    assert_eq!(binary.value, a[0]);
    assert_eq!(B::from_bytes(&bytes[cursor..cursor + 21]).unwrap(), a[0]);
    assert_eq!(bytes.len(), cursor + 21);
    let phi = source
        .iter()
        .map(|&word| embed_source::<H>(word))
        .collect::<Vec<_>>();
    assert_eq!(binary.value, binary_mle(&phi, &z));
    let transparent = transparent_weight::<H>(&point, &z, &batch).unwrap();
    assert_eq!(transparent, binary_mle(&original_coefficients, &z));
    assert_eq!(transparent, b[0]);
    assert_eq!(claim, binary.value * transparent);
}

fn ring_oracle<H: TestHost, F: TestPrime, const D: usize, M: TrinomialModulus>(
    m: usize,
    columns: usize,
) {
    let setup = setup::<F, D, M>(2, m, columns);
    let (source, _, _) = data::<H>(setup.source_len(), setup.num_vars());
    let commitment = commit_binary_clear::<H, F, D, M>(&setup, &source).unwrap();
    for column in 0..columns {
        let packed = pack_source_column::<H, F, D, M>(&setup, &source, column).unwrap();
        for (element, actual) in packed.iter().enumerate() {
            let coefficients = std::array::from_fn(|index| {
                let power = index / setup.k();
                let component = index % setup.k();
                let word: u128 =
                    source[element * setup.k() + component + setup.scalar_rows() * column].into();
                let bit = if power < 128 { (word >> power) & 1 } else { 0 };
                let coefficient = F::from_u64(bit as u64);
                if setup.k() > 1 && power % 2 == 1 {
                    -coefficient
                } else {
                    coefficient
                }
            });
            assert_eq!(
                *actual,
                TrinomialRing::from_coefficients(coefficients).unwrap()
            );
        }
        for row in 0..setup.n_a() {
            let mut expected = TrinomialRing::zero().unwrap();
            for (index, word) in packed.iter().enumerate() {
                expected += setup.matrix()[row * setup.m() + index]
                    .schoolbook_mul(word)
                    .unwrap();
            }
            assert_eq!(commitment.images[column * setup.n_a() + row], expected);
        }
    }
    let z = (0..setup.num_vars())
        .map(|i| Tape::draw_value(i + 8, 0))
        .collect::<Vec<_>>();
    let expansion = left_expansion::<H>(&source, &z, setup.scalar_rows(), columns).unwrap();
    let phi = source
        .iter()
        .map(|&word| embed_source::<H>(word))
        .collect::<Vec<_>>();
    for column in 0..columns {
        assert_eq!(
            expansion[column],
            binary_mle(
                &phi[column * setup.scalar_rows()..(column + 1) * setup.scalar_rows()],
                &z[..setup.row_vars()]
            )
        );
    }
    assert_eq!(
        binary_mle(&expansion, &z[setup.row_vars()..]),
        binary_mle(&phi, &z)
    );
    let challenges = BinaryChallengeSampler::new(profile())
        .sample_challenges(&mut FixedDraw, b"oracle", columns)
        .unwrap();
    let response = fold_integer::<H>(
        &source,
        setup.scalar_rows(),
        columns,
        &challenges,
        setup.profile(),
    )
    .unwrap();
    for (row, actual) in response.iter().enumerate() {
        // A dense wide convolution, independently reduced from highest degree.
        let mut convolution = [0i128; 323];
        let mut parity = B::ZERO;
        for (column, challenge) in challenges.iter().enumerate() {
            let word: u128 = source[row + setup.scalar_rows() * column].into();
            let mut dense = [0i128; 162];
            for term in challenge.terms() {
                dense[usize::from(term.position)] = i128::from(term.coefficient);
            }
            for (power, &coefficient) in dense.iter().enumerate() {
                for bit in 0..128 {
                    convolution[power + bit] += coefficient * ((word >> bit) & 1) as i128;
                }
            }
            parity += challenge_binary(challenge, setup.profile()).unwrap()
                * embed_source::<H>(source[row + setup.scalar_rows() * column]);
        }
        for power in (162..323).rev() {
            let high = convolution[power];
            convolution[power - 162] -= high;
            convolution[power - 81] -= high;
        }
        for (&actual, &expected) in actual.iter().zip(&convolution[..162]) {
            assert_eq!(i128::from(actual), expected);
        }
        assert_eq!(response_parity(actual).unwrap(), parity);
    }
    assert!(response
        .iter()
        .flatten()
        .any(|&coefficient| coefficient < 0));
    let packed_response = pack_response(&setup, &response).unwrap();
    for (element, packed) in packed_response.iter().enumerate() {
        let mut expected = TrinomialRing::zero().unwrap();
        for (column, challenge) in challenges.iter().enumerate() {
            let c =
                embed_scalar::<F, 162, D, M>(&challenge_scalar::<F>(challenge).unwrap()).unwrap();
            let w = pack_source_column::<H, F, D, M>(&setup, &source, column).unwrap();
            expected += c.schoolbook_mul(&w[element]).unwrap();
        }
        assert_eq!(*packed, expected);
    }
    let components = source[..setup.k()]
        .iter()
        .map(|&word| scalar_from_binary::<F>(embed_source::<H>(word)).unwrap())
        .collect::<Vec<_>>();
    let c = challenge_scalar::<F>(&challenges[0]).unwrap();
    let component_products = components
        .iter()
        .map(|word| c.schoolbook_mul(word).unwrap())
        .collect::<Vec<_>>();
    let actual = embed_scalar::<F, 162, D, M>(&c)
        .unwrap()
        .schoolbook_mul(&pack_scalar_components::<F, 162, D, M>(&components).unwrap())
        .unwrap();
    assert_eq!(
        actual,
        pack_scalar_components::<F, 162, D, M>(&component_products).unwrap()
    );
}

#[test]
fn switch_rounds_terminal_and_transparent_tensor_match_dense_definitions() {
    for variables in [4, 6] {
        frontend_oracle::<BinaryField128>(variables);
        frontend_oracle::<BinaryField192>(variables);
    }
}

#[test]
fn commitments_left_expansion_integer_fold_and_tower_action_match_oracles() {
    ring_oracle::<BinaryField128, Prime64Offset23703, 162, PlusTrinomial>(4, 4);
    ring_oracle::<BinaryField192, Prime64Offset23703, 162, PlusTrinomial>(4, 4);
    ring_oracle::<BinaryField128, Prime128OffsetA7F7, 162, PlusTrinomial>(4, 4);
    ring_oracle::<BinaryField192, Prime128OffsetA7F7, 162, PlusTrinomial>(4, 4);
    ring_oracle::<BinaryField128, Prime64Offset23703, 648, MinusTrinomial>(2, 8);
    ring_oracle::<BinaryField192, Prime64Offset23703, 648, MinusTrinomial>(2, 8);
    ring_oracle::<BinaryField128, Prime128OffsetA7F7, 648, MinusTrinomial>(2, 8);
    ring_oracle::<BinaryField192, Prime128OffsetA7F7, 648, MinusTrinomial>(2, 8);
    // Eight scalar rows versus four columns detects an exchanged point split.
    ring_oracle::<BinaryField128, Prime64Offset23703, 648, MinusTrinomial>(2, 4);
    ring_oracle::<BinaryField192, Prime128OffsetA7F7, 648, MinusTrinomial>(2, 4);
}

#[test]
fn parity_uses_euclidean_remainders_for_negative_coefficients() {
    let mut signed = [0; 162];
    signed[0] = -1;
    signed[1] = -2;
    signed[161] = -3;
    assert_eq!(
        response_parity(&signed).unwrap(),
        B::from_words([1, 0, 1 << 33]).unwrap()
    );
}

#[test]
fn d324_matrix_action_and_scalar_packing_match_schoolbook() {
    ring_oracle::<BinaryField128, Prime64Offset23703, 324, MinusTrinomial>(2, 4);
    ring_oracle::<BinaryField192, Prime64Offset23703, 324, MinusTrinomial>(2, 4);
    ring_oracle::<BinaryField128, Prime128OffsetA7F7, 324, MinusTrinomial>(2, 4);
    ring_oracle::<BinaryField192, Prime128OffsetA7F7, 324, MinusTrinomial>(2, 4);
}

#[test]
fn challenge_parity_decodes_canonical_support_and_checks_scalar_degree() {
    use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
    let profile = profile();
    let challenges = BinaryChallengeSampler::new(profile.clone())
        .sample_challenges(&mut FixedDraw, b"canonical-parity", 2)
        .unwrap();
    for challenge in &challenges {
        let bytes = challenge.canonical_support_encoding(&profile).unwrap();
        assert_eq!(bytes.len(), 21);
        assert_eq!(
            challenge_binary(challenge, &profile)
                .unwrap()
                .to_bytes()
                .as_slice(),
            bytes
        );
        let wrong_degree =
            BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic729, 47).unwrap();
        assert!(matches!(
            challenge_binary(challenge, &wrong_degree),
            Err(akita_error::AkitaError::InvalidInput(_))
        ));
        let wrong_weight =
            BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 46).unwrap();
        assert!(matches!(
            challenge_binary(challenge, &wrong_weight),
            Err(akita_error::AkitaError::InvalidInput(_))
        ));
    }
}
