//! Representation differential tests. The reference oracle retains dense Y/W.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

use super::*;
use crate::{
    family::{tables::shipped_catalog, Family128, Family64},
    shipped::SUPPORTED_GEOMETRIES,
    Verifier,
};
use akita_algebra::binary::BinaryField128 as H;
use akita_cpu_backend::custom_source::SourceCoefficients;
use akita_labinius_verifier::PrimeClaim;
use akita_serialization::AkitaSerialize;
use jolt_field::{Ring, Zero};

fn field_digits<P: FieldFamily>(digits: &[u8]) -> Vec<P::Base> {
    digits
        .iter()
        .map(|&digit| P::Base::from_u64(u64::from(digit)))
        .collect()
}

fn encoded<T: AkitaSerialize>(value: &T) -> Vec<u8> {
    let mut bytes = Vec::new();
    value.serialize_compressed(&mut bytes).unwrap();
    bytes
}

fn coefficients_match<P: FieldFamily>() {
    let mut digits: Vec<_> = (0..64).map(|index| (index % 16) as u8).collect();
    for index in [0, 17, 31, 32, 47, 63] {
        digits[index] = 0;
    }
    let packed = digit_source(&digits, 6).unwrap();
    assert_eq!(packed.len(), digits.len());
    assert_eq!(packed.num_vars(), 6);
    let actual =
        <SignedBlockPoly as SourceCoefficients<P::Base>>::source_coefficients(&packed).unwrap();
    assert_eq!(actual.as_ref(), field_digits::<P>(&digits));
}

#[test]
fn packed_digits_preserve_positive_values_and_zero_slots_in_both_fields() {
    coefficients_match::<Family64>();
    coefficients_match::<Family128>();
}

#[test]
fn packed_digits_reject_wrong_domain_and_non_digits() {
    for digits in [vec![], vec![0; 3], vec![0; 5]] {
        assert!(digit_source(&digits, 2).is_err());
    }
    assert!(digit_source(&[0], usize::BITS as usize).is_err());
    for invalid in [16, 127, 128, 255] {
        assert!(digit_source(&[0, 1, 15, invalid], 2).is_err());
    }
    assert!(digit_source(&[0, 1, 8, 15], 2).is_ok());
}

/// Run the production oracle, then replace its private W handle with an
/// independently committed dense source after checking the public bytes.
/// The emitted commitment and every subsequent transcript operation are owned
/// by the unchanged oracle. Only the retained opening representation differs.
struct DenseDigitOracle<'a, P: FieldFamily> {
    inner: ProverOracle<'a, P>,
}

impl<P: FieldFamily> RootProverOracle<P::Challenge> for DenseDigitOracle<'_, P> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.inner.bind_image(layout, channel)
    }

    fn commit_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        table: &[P::Challenge],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.inner.commit_prime_opening(layout, table, channel)
    }

    fn commit_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        digits: &[u8],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.inner.commit_response(layout, digits, channel)?;
        let mut profiles = vec![*self.inner.image.commitment.profile()];
        if let Some(prime) = &self.inner.prime {
            profiles.push(*prime.committed_group.profile());
        }
        let precommitteds = PrecommittedGroupProfiles::from_profiles(profiles)?;
        let dense = self.inner.owner.commit_table(
            self.inner.owner.scheme.schedules(),
            layout.witness_log_len(),
            field_digits::<P>(digits),
            GroupContext::scheduler_with_precommitted_groups(&precommitteds),
        )?;
        assert_eq!(
            encoded(&dense.committed_group),
            encoded(&self.inner.response.as_ref().unwrap().committed_group),
            "response commitment changed with its storage representation"
        );
        self.inner.response = Some(dense);
        Ok(())
    }

    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<P::Challenge>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.inner.discharge(claims, channel)
    }
}

fn binary_value(source: &[u128], point: &[H]) -> H {
    let mut table: Vec<_> = source.iter().map(|&word| H::embed_source(word)).collect();
    for &r in point {
        for index in 0..table.len() / 2 {
            table[index] = table[2 * index] * (H::ONE + r) + table[2 * index + 1] * r;
        }
        table.truncate(table.len() / 2);
    }
    table[0]
}

fn complete_proof_bytes_match<P: FieldFamily>() {
    let geometry = *SUPPORTED_GEOMETRIES
        .iter()
        .find(|geometry| geometry.log_num_cells == 16)
        .unwrap();
    let artifacts = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
    let shipped = shipped_catalog::<P>(geometry, &artifacts).unwrap();
    let requirements = shipped
        .tables
        .setup_requirements::<P>(&shipped.digits, Some(&shipped.elements))
        .unwrap();
    let setup = akita_pcs::new_prover_setup(&requirements).unwrap();
    let verifier_setup = setup
        .to_verifier_setup(requirements.matrix_capacity())
        .unwrap();
    let prover = Prover::<P>::new(
        geometry,
        shipped.digits.clone(),
        Some(shipped.elements.clone()),
        setup,
    )
    .unwrap();
    let verifier = Verifier::<P>::new(
        geometry,
        shipped.digits,
        Some(shipped.elements),
        verifier_setup,
    )
    .unwrap();
    let source: Vec<_> = (0..prover.root.setup().source_len())
        .map(|index| {
            (index as u128).wrapping_mul(0x917c_39d5_c8e1_7043_6b2d_09af_2e65_8ca1)
                ^ 0x861f_72b9_c42a_5d30_03ec_f217_9a54_b68d
        })
        .collect();
    let packed = prover.commit::<H>(&source).unwrap();
    let digits = encode_image(&prover.layout, &packed.clear).unwrap();
    let dense_output = prover
        .commit_table(
            prover.scheme.schedules(),
            prover.layout.image_log_len(),
            field_digits::<P>(&digits),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    assert_eq!(
        encoded(&packed.commitment),
        encoded(&dense_output.committed_group)
    );
    let dense = Committed {
        commitment: dense_output.committed_group,
        clear: packed.clear.clone(),
        handle: dense_output.private_handle,
    };
    let point: Vec<_> = (0..prover.root.setup().num_vars())
        .map(|index| H::from_words([index as u64 + 3, index as u64 + 29]))
        .collect();
    let value = binary_value(&source, &point);
    let prime_point = vec![P::Challenge::zero(); point.len()];
    let weights: Vec<_> = (0..prover.layout.degree() / prover.layout.k())
        .map(|bit| {
            P::Challenge::from_base_fn(|axis| P::Base::from_u64((17 + 31 * bit + 13 * axis) as u64))
        })
        .collect();
    let prime_value = (0..128)
        .filter(|bit| source[0] & (1u128 << bit) != 0)
        .fold(P::Challenge::zero(), |sum, bit| sum + weights[bit]);
    let prime =
        PrimeClaim::cell_evaluation(&prover.layout, &prime_point, &weights, prime_value).unwrap();
    for mode in [
        RootOpeningMode::Binary,
        RootOpeningMode::Prime,
        RootOpeningMode::Both,
    ] {
        let statement = RootStatement {
            binary: mode.has_binary().then_some((point.as_slice(), value)),
            prime: mode.has_prime().then_some(&prime),
        };
        let packed_proof = prover.open::<H>(&source, &packed, &statement).unwrap();
        let row = response_row::<P>(
            &prover.layout,
            prover.scheme.schedules(),
            if mode.has_prime() {
                prover.elements.as_ref()
            } else {
                None
            },
        )
        .unwrap();
        let mut oracle = DenseDigitOracle {
            inner: ProverOracle {
                owner: &prover,
                image: &dense,
                row,
                prime: None,
                response: None,
                mode,
                order: Order::default(),
            },
        };
        let (dense_proof, _) = prove_root_reduction_bytes::<H, P::Base, P::Challenge, _, _, _>(
            prover.root(),
            &source,
            &dense.clear,
            &statement,
            &mut oracle,
        )
        .unwrap();
        assert_eq!(
            packed_proof, dense_proof,
            "complete {mode:?} proof bytes changed"
        );
        verifier
            .verify::<H>(&packed.commitment, &statement, &packed_proof)
            .unwrap();
        verifier
            .verify::<H>(&dense.commitment, &statement, &dense_proof)
            .unwrap();
    }
}

#[test]
fn fp64_packed_and_dense_digit_sources_have_identical_complete_proofs() {
    complete_proof_bytes_match::<Family64>();
}

#[test]
fn fp128_packed_and_dense_digit_sources_have_identical_complete_proofs() {
    complete_proof_bytes_match::<Family128>();
}
