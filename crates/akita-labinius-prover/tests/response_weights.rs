#![cfg(feature = "labinius")]

mod lowered_support;

use akita_error::AkitaError;
use akita_labinius_prover::response_weights::coefficient_weights;
use akita_labinius_verifier::lowered::{witness_weights_dense, LoweredRootLayout};
use jolt_field::Zero;
use lowered_support::{Case, BASES, F};

#[test]
fn compact_coefficient_factor_tensor_digits_equals_dense_weights_for_each_base() {
    for base in BASES {
        let case = Case::new(base);
        let public = case.public();
        let compact = coefficient_weights(&case.layout, &public).unwrap();
        let dense = witness_weights_dense(&case.layout, &public).unwrap();
        let tensor = compact
            .iter()
            .flat_map(|&coefficient| {
                public
                    .digit_powers()
                    .iter()
                    .map(move |&digit| digit * coefficient)
            })
            .collect::<Vec<_>>();
        assert_eq!(tensor, dense, "{base:?}");
        assert_eq!(
            compact.len(),
            case.layout.witness_len() / public.digit_powers().len()
        );
        assert!(case.layout.padded_coefficients() > case.layout.degree());
        for row in compact.chunks_exact(case.layout.padded_coefficients()) {
            assert!(row
                .iter()
                .skip(case.layout.degree())
                .all(|&v| v == F::zero()));
        }
    }
}

#[test]
fn compact_weights_reject_a_public_digit_factor_from_a_different_base() {
    let case = Case::new(BASES[0]);
    let public = case.public();
    let foreign = LoweredRootLayout::new(&case.setup, &case.shape, BASES[1]).unwrap();
    assert!(matches!(
        coefficient_weights(&foreign, &public),
        Err(AkitaError::InvalidInput(_))
    ));
}
