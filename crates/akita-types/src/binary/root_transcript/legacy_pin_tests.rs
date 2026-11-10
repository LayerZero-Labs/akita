//! Frozen predecessor vectors, captured before the indexed-group seam.

use super::*;
use crate::binary::{
    binary_word_manifest_fixture, field_switch_challenge, field_switch_partials,
    field_switch_round, field_switch_terminal, PrimePlaneManifest,
};
use crate::{FoldSite, ProverGrinding};
use akita_algebra::binary::BinaryField162 as B;
use akita_challenges::{PreviewFoldDraw, ProverFoldDraw};
use akita_params::{binary_setup::BinaryFieldProfile, GrindingPlan, GrindingRun};
use akita_transcript::new_prover_channel;
use jolt_field::{Ext2, Prime128Offset275, Prime64Offset59};
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;

fn predecessor_vector<F, E>(profile: BinaryFieldProfile) -> [u8; 32]
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let manifest = binary_word_manifest_fixture(profile, 1);
    let descriptor = manifest.descriptor();
    let claim = BinaryWordClaim::new(
        descriptor,
        descriptor.polynomials()[0],
        vec![[3; 16]; 3],
        [0; 16],
    )
    .unwrap();
    let prime = PrimePlaneManifest::with_field_profile(descriptor, None, profile).unwrap();
    let payload = (0..manifest.payload_coefficients().unwrap())
        .map(|i| F::from_u64(17 + i as u64))
        .collect::<Vec<_>>();
    let mut state = new_prover_channel(b"binary-root-predecessor-vector");
    binary_root_statement::<F, E, _>(&mut state, &manifest, Some(&claim), &prime, &payload, None)
        .unwrap();

    // A zero carrier source has a valid zero word claim and zero round
    // polynomials. Use the actual frontend codecs and challenge sequence.
    field_switch_partials(&mut state, &[0; 128]).unwrap();
    let lambda = (0..7)
        .map(|_| field_switch_challenge(&mut state).unwrap())
        .collect::<Vec<_>>();
    let mut product = B::ZERO;
    let point = (0..3)
        .map(|_| field_switch_round(&mut state, &mut product, B::ZERO, B::ZERO).unwrap())
        .collect::<Vec<_>>();
    field_switch_terminal(
        &mut state,
        descriptor,
        &claim,
        &claim.carrier_point(descriptor).unwrap(),
        &lambda,
        &point,
        product,
        B::ZERO,
    )
    .unwrap();
    let d = (0..64 * manifest.d_label(true, 0).unwrap().rank as usize)
        .map(|i| F::from_u64(201 + i as u64))
        .collect::<Vec<_>>();
    akita_transcript::send_field_group(
        &mut state,
        FoldSite::OpeningPayload {
            level: 0,
            ring_dimension: 64,
        }
        .id()
        .unwrap(),
        &d,
    )
    .unwrap();
    let plan = GrindingPlan::new(
        vec![
            GrindingRun::fold_response(0),
            GrindingRun::fold_challenge_work(0, 2, binary_root_family().unwrap().set_bits())
                .unwrap(),
            GrindingRun::fold_challenge_group(0, 0, 2).unwrap(),
        ],
        profile.challenge_order().unwrap(),
    )
    .unwrap();
    let mut grinding = ProverGrinding::new(&mut state, &plan);
    let site = GrindingSite::FoldResponse { level: 0 };
    let mut preview = grinding.preview_fold_response(site, 129).unwrap();
    let work = preview.take_work().unwrap();
    let expected = binary_root_challenges(&mut PreviewFoldDraw::new(&mut preview), 2, 1).unwrap();
    grinding.commit_fold_response(site, work).unwrap();
    let actual =
        binary_root_challenges(&mut ProverFoldDraw::new(grinding.state_mut(), 0, 0), 2, 1).unwrap();
    assert_eq!(actual, expected);
    grinding.record_fold_challenges(0, 0, 2).unwrap();
    grinding.finish().unwrap();
    let continuation = crate::binary::challenge_block(&mut state).unwrap();
    let mut digest = Shake256::default();
    digest.update(state.narg_string());
    for challenge in actual.as_slice() {
        digest.update(&(challenge.positions.len() as u64).to_le_bytes());
        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            digest.update(&(position as u64).to_le_bytes());
            digest.update(&coefficient.to_le_bytes());
        }
    }
    digest.update(&continuation);
    let mut result = [0; 32];
    digest.finalize_xof().read(&mut result);
    result
}

#[test]
fn legacy_single_group_p128_transcript_pin() {
    assert_eq!(
        predecessor_vector::<Prime128Offset275, Prime128Offset275>(BinaryFieldProfile::P128),
        [
            118, 11, 168, 195, 29, 235, 35, 189, 234, 112, 123, 181, 245, 98, 112, 8, 253, 125, 55,
            185, 23, 169, 40, 87, 87, 111, 122, 74, 118, 109, 11, 13
        ],
    );
}

#[test]
fn legacy_single_group_p64_atom_transcript_pin() {
    assert_eq!(
        predecessor_vector::<Prime64Offset59, Ext2<Prime64Offset59>>(BinaryFieldProfile::P64),
        [
            165, 151, 142, 73, 124, 156, 208, 204, 88, 145, 249, 126, 143, 144, 241, 45, 251, 154,
            99, 194, 118, 234, 179, 128, 112, 81, 179, 132, 135, 83, 192, 207
        ],
    );
}
