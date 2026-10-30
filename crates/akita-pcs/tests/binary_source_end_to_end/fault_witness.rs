//! Reconstruct the honest fixture table solely through public arithmetic APIs.

use super::*;
use akita_algebra::ring::trinomial::image::Q24Image;
use akita_algebra::ring::trinomial::signed::signed_trinomial_product;
use akita_algebra::MinusTrinomial;
use akita_challenges::{Challenges, FoldChallengeFamily};
use akita_cpu_backend::binary_root::{
    binary_row_partials, BinaryEvaluationWitness, BinaryRootRelation,
};
use akita_types::binary_source::{binary_root_matrix, PrimePlaneManifest, BINARY_SOURCE_Q};
use akita_types::root_repacking::RootCoefficientRepacking;

#[allow(clippy::too_many_arguments)]
pub(super) fn reconstruct(
    manifest: &BinaryCommittedGroupManifest,
    claim: Option<&BinaryTerminalClaim>,
    prime: &PrimePlaneManifest,
    plan: &BinaryRootWitnessPlan,
    seed: &[u8; 32],
    words: &[u128],
    challenges: &Challenges,
) -> Result<Vec<i8>, AkitaError> {
    let descriptor = manifest.descriptor();
    let index = descriptor.index_map();
    let ring_rows = index.padded_carriers() / (4 * index.columns());
    let sources = source_rings(manifest, words)?;
    let matrix = binary_root_matrix(seed, manifest)?;
    let mut images = Vec::new();
    for column in sources.chunks_exact(ring_rows) {
        for row in &matrix {
            let mut coefficients = [0i128; 648];
            for (source, multiplier) in column.iter().zip(row) {
                let product = signed_trinomial_product::<648, MinusTrinomial>(source, multiplier)?;
                for (coefficient, product) in coefficients.iter_mut().zip(product) {
                    *coefficient += product;
                }
            }
            let mut image = [Q24Image::from_residue(0)?; 648];
            for (coefficient, value) in image.iter_mut().zip(coefficients) {
                *coefficient =
                    Q24Image::from_residue(value.rem_euclid(i128::from(BINARY_SOURCE_Q)) as u32)?;
            }
            images.push(image);
        }
    }
    let response = BinaryRootRelation::build(
        descriptor,
        &matrix,
        &sources,
        &images,
        challenges,
        FoldChallengeFamily::BinaryRoot,
    )?;
    let quotient = response.committed_quotient(
        descriptor,
        &matrix,
        challenges,
        plan.repacking().destination_ring_dimension(),
    )?;
    let mut native: Vec<Vec<i8>> = (0..3)
        .map(|segment| extract(quotient.relation.repacking(), &quotient.digits, segment))
        .collect();
    if let Some(claim) = claim {
        let partials = binary_row_partials(descriptor, claim, &sources)?;
        let evaluation = BinaryEvaluationWitness::build(
            descriptor,
            claim,
            &partials,
            &response.response,
            challenges,
            u128::MAX - 274,
        )?;
        native.extend([partials, evaluation.parity_digits, evaluation.value_digits]);
    }
    if plan.prime_segment().is_some() {
        native.push(akita_cpu_backend::binary_root::prime_plane_digits::<F>(
            descriptor, prime, &sources,
        )?);
    }
    plan.repacking()
        .repack(&native.iter().map(Vec::as_slice).collect::<Vec<_>>())
}

fn extract(plan: &RootCoefficientRepacking, table: &[i8], segment: usize) -> Vec<i8> {
    let shape = plan.segments()[segment];
    let width =
        shape.geometry.polynomial_modulus_dimension() * shape.geometry.coordinate_plane_count();
    let mut digits = Vec::new();
    for element in 0..shape.elements {
        for digit in 0..shape.digits {
            for coefficient in 0..width {
                digits.push(
                    table[plan
                        .coefficient_address(segment, element, digit, coefficient)
                        .unwrap()],
                );
            }
        }
    }
    digits
}

fn source_rings(
    manifest: &BinaryCommittedGroupManifest,
    words: &[u128],
) -> Result<Vec<[i128; 648]>, AkitaError> {
    let index = manifest.descriptor().index_map();
    let ring_rows = index.padded_carriers() / (4 * index.columns());
    let mut sources = vec![[0i128; 648]; index.columns() * ring_rows];
    for (word, &value) in words.iter().enumerate() {
        for bit in 0..index.word_width() {
            let position = index.logical_to_ring(word, bit)?;
            sources[position.column * ring_rows + position.ring_row][position.coefficient] =
                i128::from(position.sign) * ((value >> bit) & 1) as i128;
        }
    }
    Ok(sources)
}

pub(super) fn foreign_partial_payload(
    manifest: &BinaryCommittedGroupManifest,
    claim: &BinaryTerminalClaim,
    words: &[u128],
    seed: &[u8; 32],
) -> Result<Vec<F>, AkitaError> {
    let sources = source_rings(manifest, words)?;
    let mut partials = binary_row_partials(manifest.descriptor(), claim, &sources)?;
    // Commit a distinct, still range-valid U; retain the original U in the
    // outgoing witness so only its D authentication can expose the mismatch.
    partials[0] ^= 1;
    akita_types::binary_source::binary_public_digit_commitment::<F>(
        seed,
        akita_params::binary_setup::SetupStreamLabel {
            role: akita_params::binary_setup::SetupStreamRole::WitnessCommitment,
            ..manifest.b_label()
        },
        &partials,
    )
}
