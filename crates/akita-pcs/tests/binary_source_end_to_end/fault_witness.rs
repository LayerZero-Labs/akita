//! Reconstruct the honest fixture table solely through public arithmetic APIs.

use super::*;
use akita_algebra::ring::trinomial::image::Q24Image;
use akita_algebra::ring::trinomial::signed::signed_trinomial_product;
use akita_algebra::MinusTrinomial;
use akita_challenges::{Challenges, FoldChallengeFamily};
use akita_cpu_backend::binary_root::{
    binary_row_partials, BinaryEvaluationWitness, BinaryRootRelation,
};
use akita_types::binary_source::{binary_root_matrix, BINARY_SOURCE_Q};
use akita_types::root_repacking::RootCoefficientRepacking;

pub(super) fn reconstruct(
    manifest: &BinaryCommittedGroupManifest,
    claim: &BinaryTerminalClaim,
    plan: &BinaryRootWitnessPlan,
    seed: &[u8; 32],
    words: &[u128],
    challenges: &Challenges,
) -> Result<Vec<i8>, AkitaError> {
    let descriptor = manifest.descriptor();
    let index = descriptor.index_map();
    let ring_rows = index.padded_carriers() / (4 * index.columns());
    let mut sources = vec![[0i128; 648]; index.columns() * ring_rows];
    for (word, &value) in words.iter().enumerate() {
        for bit in 0..index.word_width() {
            let position = index.logical_to_ring(word, bit)?;
            sources[position.column * ring_rows + position.ring_row][position.coefficient] =
                i128::from(position.sign) * ((value >> bit) & 1) as i128;
        }
    }
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
    let partials = binary_row_partials(descriptor, claim, &sources)?;
    let evaluation = BinaryEvaluationWitness::build(
        descriptor,
        claim,
        &partials,
        &response.response,
        challenges,
        u128::MAX - 274,
    )?;
    let native: Vec<Vec<i8>> = (0..3)
        .map(|segment| extract(quotient.relation.repacking(), &quotient.digits, segment))
        .collect();
    plan.repacking().repack(&[
        &native[0],
        &native[1],
        &native[2],
        &partials,
        &evaluation.parity_digits,
        &evaluation.value_digits,
    ])
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
