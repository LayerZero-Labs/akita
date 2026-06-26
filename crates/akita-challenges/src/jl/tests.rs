use akita_field::{
        field_modulus, CanonicalField, FieldCore, Fp64, Prime128Offset275, Prime32Offset99,
    };
    use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
    use akita_transcript::AkitaTranscript;

    use crate::jl::center_to_i32;
    use crate::jl::testutil::{matrix_from_sign_rows, matrix_sign_at};
    use crate::{center_coefficients, project_digits_reference, project_digits_scalar};
    use crate::{JlProjectionMatrix, DEFAULT_JL_ROWS, MAX_JL_DIGIT};

    type F32 = Prime32Offset99;
    type F64 = Fp64<4294967197>;
    type F128 = Prime128Offset275;

    fn binary_sign(seed: usize) -> i8 {
        if seed & 1 == 0 {
            -1
        } else {
            1
        }
    }

    fn centered_ref<F: FieldCore + CanonicalField>(c: F) -> i32 {
        let q = field_modulus::<F>();
        let half_q = q / 2;
        center_to_i32(c.to_canonical_u128(), q, half_q).expect("centered coeff fits i32")
    }

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
        let cols = 7;
        let signs: Vec<Vec<i8>> = (0..n_rows)
            .map(|r| (0..cols).map(|c| binary_sign(r + c)).collect())
            .collect();
        let coeffs: Vec<F> = (0..cols).map(|c| F::from_i64(c as i64 * 3 - 9)).collect();

        let matrix = matrix_from_sign_rows(&signs).unwrap();
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
        let cols: usize = 1023;
        let signs: Vec<Vec<i8>> = (0..n_rows)
            .map(|r| (0..cols).map(|c| binary_sign(r * 7 + c * 3)).collect())
            .collect();
        let digits: Vec<i32> = (0..cols).map(|i| ((i % 33) as i32) - 16).collect();
        let matrix = matrix_from_sign_rows(&signs).unwrap();
        assert_eq!(
            project_digits_scalar(&matrix, &digits).unwrap(),
            project_digits_reference(&matrix, &digits).unwrap()
        );
        assert_eq!(
            matrix.project_digits(&digits).unwrap(),
            project_digits_reference(&matrix, &digits).unwrap()
        );
    }

    #[test]
    fn parallel_column_panels_match_reference() {
        let n_rows = 16;
        let cols = 8191;
        assert!(n_rows * cols >= crate::jl::panel::JL_PARALLEL_ELEMS_THRESHOLD);
        let signs: Vec<Vec<i8>> = (0..n_rows)
            .map(|r| (0..cols).map(|c| binary_sign(r * 5 + c * 2)).collect())
            .collect();
        let digits: Vec<i32> = (0..cols).map(|i| ((i % 65) as i32) - 32).collect();
        assert!(digits.iter().all(|d| d.abs() <= MAX_JL_DIGIT));
        let matrix = matrix_from_sign_rows(&signs).unwrap();
        assert_eq!(
            matrix.project_digits(&digits).unwrap(),
            project_digits_reference(&matrix, &digits).unwrap()
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
            .map(|r| (0..cols).map(|c| binary_sign(r + c)).collect())
            .collect();
        let matrix = matrix_from_sign_rows(&signs).unwrap();
        for (r, row) in signs.iter().enumerate() {
            for (c, &s) in row.iter().enumerate() {
                assert_eq!(matrix_sign_at(&matrix, r, c), Some(s));
            }
        }
        assert_eq!(matrix_sign_at(&matrix, n_rows, 0), None);
        assert_eq!(matrix_sign_at(&matrix, 0, cols), None);
    }

    #[test]
    fn explicit_zero_sign_rows_are_rejected() {
        let signs = vec![vec![1i8, 0, -1i8]];
        assert!(matrix_from_sign_rows(&signs).is_err());
    }

    #[test]
    fn fp128_small_digits_project() {
        let coeffs = [F128::from_i64(-5), F128::from_i64(7)];
        let signs = vec![vec![1i8, 1i8], vec![1i8, -1i8]];
        let matrix = matrix_from_sign_rows(&signs).unwrap();
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
        let matrix = matrix_from_sign_rows(&signs).unwrap();
        assert!(matrix.project(&[coeff]).is_err());
    }

    #[test]
    fn project_digits_matches_project() {
        let signs: Vec<Vec<i8>> = (0..4)
            .map(|r| (0..11).map(|c| binary_sign(r + c)).collect())
            .collect();
        let coeffs: Vec<F64> = (0..11).map(|i| F64::from_i64(i as i64 - 5)).collect();
        let matrix = matrix_from_sign_rows(&signs).unwrap();
        let digits = center_coefficients(&coeffs).unwrap();
        assert_eq!(
            matrix.project(&coeffs).unwrap(),
            matrix.project_digits(&digits).unwrap()
        );
    }

    #[test]
    fn malformed_inputs_return_error() {
        let signs = vec![vec![1i8, 1i8]];
        let matrix = matrix_from_sign_rows(&signs).unwrap();
        assert!(matrix.project(&[F64::from_i64(1)]).is_err());

        let mut t = AkitaTranscript::<F64>::new(DOMAIN_AKITA_PROTOCOL);
        assert!(JlProjectionMatrix::sample::<F64, _>(&mut t, 4, 0).is_err());
        assert!(JlProjectionMatrix::sample::<F64, _>(&mut t, 0, 4).is_err());
    }

    #[test]
    fn digit_bound_is_enforced() {
        let signs = vec![vec![1i8; 4]];
        let matrix = matrix_from_sign_rows(&signs).unwrap();
        let digits = vec![MAX_JL_DIGIT + 1, 0, 0, 0];
        assert!(matrix.project_digits(&digits).is_err());
    }

    #[test]
    fn i32_min_digit_is_rejected() {
        let signs = vec![vec![1i8]];
        let matrix = matrix_from_sign_rows(&signs).unwrap();
        assert!(matrix.project_digits(&[i32::MIN]).is_err());
    }

    #[test]
    fn oversized_geometry_returns_error() {
        let mut t = AkitaTranscript::<F64>::new(DOMAIN_AKITA_PROTOCOL);
        assert!(JlProjectionMatrix::sample::<F64, _>(&mut t, usize::MAX, 8).is_err());
    }
