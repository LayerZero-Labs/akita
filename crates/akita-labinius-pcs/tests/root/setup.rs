use akita_config::{ensure_prover_schedule_fits_setup, TrustedScheduleCatalog};
use akita_cpu_backend::AkitaProverSetup;
use akita_error::AkitaError;
use akita_labinius_pcs::{DigitConfig, Digits2, ImageConfig, RootPcsProver, RootPcsVerifier, F};
use akita_labinius_verifier::{channel::ClearChannel, lowered::LoweredRootLayout};
use akita_params::{PolynomialGroupLayout, ScheduleLookupKey, SetupMatrixCapacity};

use super::support::TestDigits;
use crate::common::Recording;

#[test]
fn root_constructors_reject_selected_key_capacity_before_parent_activity() {
    let root = crate::common::admitted(0, 0x31);
    let fixture = Digits2::fixture(0);
    let descriptor = fixture.prover_setup.expanded.descriptor();
    let layout = LoweredRootLayout::new(root.setup(), root.shape(), Digits2::BASE).unwrap();
    let image_row = fixture
        .images
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(layout.image_log_len()),
        ))
        .unwrap();
    let grouped_key = ScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(layout.witness_log_len()),
        precommitteds: vec![image_row.profiles().final_group],
    };
    let digit_row = fixture.digits.resolve_key(&grouped_key).unwrap();
    let min_vars = grouped_key.max_num_vars();
    assert_eq!(grouped_key.num_polynomials().unwrap(), 2);
    assert_eq!(descriptor.max_num_vars, min_vars);
    // Retain the complete two-polynomial matrix while independently reducing
    // each logical bound. Physical footprint admission must still succeed.
    let capacity = SetupMatrixCapacity {
        num_field_elements: descriptor.num_field_elements,
    };
    for (max_vars, max_polys) in [(min_vars, 1), (min_vars - 1, 2)] {
        let setup =
            AkitaProverSetup::<F>::generate_with_capacity(max_vars, max_polys, capacity).unwrap();
        ensure_prover_schedule_fits_setup::<ImageConfig>(
            &setup.expanded,
            image_row.schedule(),
            &image_row.profiles().opening_layout().unwrap(),
        )
        .unwrap();
        ensure_prover_schedule_fits_setup::<Digits2>(
            &setup.expanded,
            digit_row.schedule(),
            &digit_row.profiles().opening_layout().unwrap(),
        )
        .unwrap();
        if max_polys == 1 {
            assert!(TrustedScheduleCatalog::<ImageConfig>::verifier_admits(
                &setup.expanded,
                image_row,
            )
            .unwrap());
        }
        assert!(
            !TrustedScheduleCatalog::<Digits2>::verifier_admits(&setup.expanded, digit_row)
                .unwrap()
        );
        let verifier_setup = setup.to_verifier_setup(capacity).unwrap();
        // Constructors have no channel argument. Rejection leaves no prover or
        // oracle that could advance this already initialized parent channel.
        let state = akita_transcript::new_prover_channel(b"root-capacity/v1", b"parent").unwrap();
        let mut record = Recording::new(state);
        assert!(matches!(
            RootPcsProver::<Digits2>::new(
                root.clone(),
                fixture.images.clone(),
                fixture.digits.clone(),
                setup,
            ),
            Err(AkitaError::InvalidSetup(message))
                if message == "root PCS rows do not fit verifier setup"
        ));
        assert!(matches!(
            RootPcsVerifier::<Digits2>::new(
                root.clone(),
                fixture.images.clone(),
                fixture.digits.clone(),
                verifier_setup,
            ),
            Err(AkitaError::InvalidSetup(message))
                if message == "root PCS rows do not fit verifier setup"
        ));
        assert!(record.events.is_empty());
        assert!(record.public.is_empty());
        assert!(record.messages.is_empty());
        assert_eq!(record.message_calls, 0);
        assert!(record.draws.is_empty());
        assert!(record.inner.narg_string().is_empty());
        let mut control =
            akita_transcript::new_prover_channel(b"root-capacity/v1", b"parent").unwrap();
        assert_eq!(
            record.challenge_block().unwrap(),
            control.challenge_block().unwrap(),
        );
    }
}

#[test]
fn root_constructors_accept_minimum_two_polynomial_capacity() {
    let root = crate::common::admitted(0, 0x31);
    let fixture = Digits2::fixture(0);
    assert_eq!(
        fixture
            .prover_setup
            .expanded
            .descriptor()
            .max_num_batched_polys,
        2,
    );
    assert_eq!(
        fixture
            .verifier_setup
            .expanded()
            .descriptor()
            .max_num_batched_polys,
        2,
    );
    fixture.prover(root.clone());
    fixture.verifier(root);
}
