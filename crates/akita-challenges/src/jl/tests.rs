use super::*;
use akita_field::{FieldCore, Fp64, Prime128Offset275, Prime32Offset99};
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::AkitaTranscript;

type F32 = Prime32Offset99;
type F64 = Fp64<4294967197>;
type F128 = Prime128Offset275;

/// Balanced representative used by the naive reference projection.
fn centered_ref<F: FieldCore + CanonicalField>(c: F) -> i32 {
    let q = field_modulus::<F>();
    let half_q = q / 2;
    center_to_i32(c.to_canonical_u128(), q, half_q).expect("centered coeff fits i32")
}

/// Naive integer projection used as the correctness oracle.
fn reference_project(signs: &[Vec<i8>], centered: &[i32]) -> Vec<i32> {
    signs
        .iter()
        .map(|row| {
            row.iter()
                .zip(centered)
                .map(|(&s, &v)| i32::from(s) * v)
                .sum()
        })
        .collect()
}

fn project_vs_reference_for<F: FieldCore + CanonicalField>() {
    let n_rows = 5;
    let cols = 7; // intentionally not a multiple of 4
    let signs: Vec<Vec<i8>> = (0..n_rows)
        .map(|r| (0..cols).map(|c| ((r + c) % 3) as i8 - 1).collect())
        .collect();
    let coeffs: Vec<F> = (0..cols).map(|c| F::from_i64(c as i64 * 3 - 9)).collect();

    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    let image = matrix.project(&coeffs).unwrap();

    let centered: Vec<i32> = coeffs.iter().map(|&c| centered_ref(c)).collect();
    let expected = reference_project(&signs, &centered);
    assert_eq!(image.coords(), expected.as_slice());
}

#[test]
fn project_matches_reference_across_fields() {
    project_vs_reference_for::<F32>();
    project_vs_reference_for::<F64>();
    project_vs_reference_for::<F128>();
}

#[test]
fn fast_kernel_matches_reference_kernel() {
    let n_rows = 32;
    let cols: usize = 1023; // non-multiple of 4 and 8
    let signs: Vec<Vec<i8>> = (0..n_rows)
        .map(|r| (0..cols).map(|c| ((r * 7 + c * 3) % 3) as i8 - 1).collect())
        .collect();
    let digits: Vec<i32> = (0..cols).map(|i| ((i % 33) as i32) - 16).collect();
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    assert_eq!(
        matrix.project_digits(&digits).unwrap(),
        matrix.project_digits_reference(&digits).unwrap()
    );
}

#[test]
fn parallel_column_panels_match_reference() {
    // `n_rows * cols` is above `JL_PARALLEL_ELEMS_THRESHOLD`, so (with the
    // `parallel` feature) `project_digits` fans out over many column panels and
    // reduces per-panel partials. `cols` is not a multiple of 4 to also cover
    // the last panel's remainder bytes. The checked `i64` reference is the oracle.
    let n_rows = 16;
    let cols = 8191;
    assert!(n_rows * cols >= crate::jl::panel::JL_PARALLEL_ELEMS_THRESHOLD);
    let signs: Vec<Vec<i8>> = (0..n_rows)
        .map(|r| (0..cols).map(|c| ((r * 5 + c * 2) % 3) as i8 - 1).collect())
        .collect();
    let digits: Vec<i32> = (0..cols).map(|i| ((i % 65) as i32) - 32).collect();
    assert!(digits.iter().all(|d| d.abs() <= MAX_JL_DIGIT));
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    assert_eq!(
        matrix.project_digits(&digits).unwrap(),
        matrix.project_digits_reference(&digits).unwrap()
    );
}

#[test]
fn sample_is_deterministic_and_replayable() {
    let mut t1 = AkitaTranscript::<F64>::new(DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<F64>::new(DOMAIN_AKITA_PROTOCOL);
    let m1 = JlProjectionMatrix::sample::<F64, _>(&mut t1, DEFAULT_JL_ROWS, 17).unwrap();
    let m2 = JlProjectionMatrix::sample::<F64, _>(&mut t2, DEFAULT_JL_ROWS, 17).unwrap();
    assert_eq!(m1, m2);
    assert_eq!(m1.n_rows(), DEFAULT_JL_ROWS);
    assert_eq!(m1.cols(), 17);

    let coeffs: Vec<F64> = (0..17).map(|i| F64::from_i64(i as i64 - 8)).collect();
    assert_eq!(m1.project(&coeffs).unwrap(), m2.project(&coeffs).unwrap());
}

#[test]
fn packed_matrix_roundtrips_signs() {
    let n_rows = 4;
    let cols = 7;
    let signs: Vec<Vec<i8>> = (0..n_rows)
        .map(|r| (0..cols).map(|c| ((r + c) % 3) as i8 - 1).collect())
        .collect();
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    for (r, row) in signs.iter().enumerate() {
        for (c, &s) in row.iter().enumerate() {
            assert_eq!(matrix.sign_at(r, c), Some(s));
        }
    }
    assert_eq!(matrix.sign_at(n_rows, 0), None);
    assert_eq!(matrix.sign_at(0, cols), None);
}

#[test]
fn fp128_small_digits_project() {
    let coeffs = [F128::from_i64(-5), F128::from_i64(7)];
    let signs = vec![vec![1i8, 1i8], vec![1i8, -1i8]];
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    let image = matrix.project(&coeffs).unwrap();
    assert_eq!(image.coords(), &[2, -12]);
}

#[test]
fn oversized_non_digit_coefficient_is_rejected() {
    let q = field_modulus::<F128>();
    let half_q = q / 2;
    let large = half_q - 17;
    assert!(
        large > i64::MAX as u128,
        "expected a centered magnitude past i64"
    );

    let coeff = F128::from_canonical_u128_reduced(large);
    let signs = vec![vec![1i8]];
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    assert!(matrix.project(&[coeff]).is_err());
}

#[test]
fn embed_enforces_injective_signed_window() {
    let q = field_modulus::<F32>();
    let half_q = q / 2;
    let at_boundary = i32::try_from(half_q).expect("fp32 half modulus fits i32");

    // Coordinate exactly at the boundary `|p| = q/2` embeds injectively.
    let image = JlImage {
        coords: vec![at_boundary],
    };
    let embedded = image.embed_into_field::<F32>().unwrap();
    assert_eq!(embedded.len(), 1);
    assert_eq!(embedded[0].to_canonical_u128(), half_q);

    // One past the boundary aliases modulo q and is rejected.
    let over = JlImage {
        coords: vec![at_boundary + 1],
    };
    assert!(over.embed_into_field::<F32>().is_err());
}

#[test]
fn check_l2_accepts_generous_and_rejects_tight() {
    let signs = vec![vec![1i8, 1i8, 1i8], vec![1i8, 0, -1i8]];
    let coeffs = [F64::from_i64(3), F64::from_i64(-4), F64::from_i64(5)];
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    let image = matrix.project(&coeffs).unwrap();
    assert_eq!(image.coords(), &[4, -2]);

    assert_eq!(image.l2_norm_sq_checked().unwrap(), 20);
    assert!(image.check_l2(20).is_ok());
    assert!(image.check_l2(19).is_err());
}

#[test]
fn project_digits_matches_project() {
    let signs: Vec<Vec<i8>> = (0..4)
        .map(|r| (0..11).map(|c| ((r + c) % 3) as i8 - 1).collect())
        .collect();
    let coeffs: Vec<F64> = (0..11).map(|i| F64::from_i64(i as i64 - 5)).collect();
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    let digits = super::center_coefficients(&coeffs).unwrap();
    assert_eq!(
        matrix.project(&coeffs).unwrap(),
        matrix.project_digits(&digits).unwrap()
    );
}

#[test]
fn malformed_inputs_return_error() {
    let signs = vec![vec![1i8, 1i8]];
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    assert!(matrix.project(&[F64::from_i64(1)]).is_err());

    let mut t = AkitaTranscript::<F64>::new(DOMAIN_AKITA_PROTOCOL);
    assert!(JlProjectionMatrix::sample::<F64, _>(&mut t, 4, 0).is_err());
    assert!(JlProjectionMatrix::sample::<F64, _>(&mut t, 0, 4).is_err());
}

#[test]
fn digit_bound_is_enforced() {
    let signs = vec![vec![1i8; 4]];
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    let digits = vec![MAX_JL_DIGIT + 1, 0, 0, 0];
    assert!(matrix.project_digits(&digits).is_err());
}

#[test]
fn i32_min_digit_is_rejected() {
    let signs = vec![vec![1i8]];
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    assert!(matrix.project_digits(&[i32::MIN]).is_err());
}

#[test]
fn oversized_geometry_returns_error() {
    let mut t = AkitaTranscript::<F64>::new(DOMAIN_AKITA_PROTOCOL);
    assert!(JlProjectionMatrix::sample::<F64, _>(&mut t, usize::MAX, 8).is_err());
}
