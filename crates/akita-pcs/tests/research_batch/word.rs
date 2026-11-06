//! H word fixture and rejection controls for the same native research opening.

use super::{E, F};
use akita_algebra::binary::BinaryField128 as H;
use akita_error::AkitaError;
use akita_types::binary_source::{BinarySourceDescriptor, BinaryWordClaim};
use akita_types::opening_claims::GroupClaimKind;
use akita_types::{CommittedGroup, OpeningClaims, PolynomialGroupClaims};

pub(super) fn claim(descriptor: &BinarySourceDescriptor, words: &[u128]) -> BinaryWordClaim {
    let point = [7u64, 11, 13];
    let mut values = words
        .iter()
        .map(|word| H::from_words([*word as u64, (*word >> 64) as u64]))
        .collect::<Vec<_>>();
    for coordinate in point {
        let r = H::from_words([coordinate, 0]);
        values = values
            .chunks_exact(2)
            .map(|pair| pair[0] * (H::ONE + r) + pair[1] * r)
            .collect();
    }
    assert_eq!(values.len(), 1);
    let [low, high] = values[0].to_words();
    BinaryWordClaim::new(
        descriptor,
        descriptor.polynomials()[0],
        point.map(|coordinate| u128::from(coordinate).to_le_bytes()).to_vec(),
        (u128::from(low) | (u128::from(high) << 64)).to_le_bytes(),
    )
    .unwrap()
}

pub(super) fn reject_mutations(
    claims: &OpeningClaims<'_, E, CommittedGroup<F>>,
    rejects: impl Fn(&OpeningClaims<'_, E, CommittedGroup<F>>) -> bool,
) {
    let first = &claims.groups()[0];
    let GroupClaimKind::Binary { manifest, claim: Some(word), .. } = first.kind() else {
        panic!("hash group must carry H");
    };
    for change_point in [false, true] {
        let mut point = word.point().to_vec();
        let mut value = *word.value();
        if change_point {
            point[0][0] ^= 1;
        } else {
            value[0] ^= 1;
        }
        let changed = BinaryWordClaim::new(
            manifest.descriptor(), *word.polynomial(), point, value,
        ).unwrap();
        let mut groups = claims.groups().to_vec();
        groups[0] = PolynomialGroupClaims::new_research_binary_with_word(
            first.commitment().clone(),
            first.research_terminals().unwrap().clone(),
            Some(changed),
        ).unwrap();
        assert!(rejects(&OpeningClaims::from_groups(groups).unwrap()),
            "H mutation: change_point={change_point}");
    }
    let second = &claims.groups()[1];
    let GroupClaimKind::Binary { manifest, .. } = second.kind() else {
        panic!("second binary group");
    };
    let misplaced = claim(manifest.descriptor(), &[2; 8]);
    assert!(matches!(
        PolynomialGroupClaims::new_research_binary_with_word(
            second.commitment().clone(),
            second.research_terminals().unwrap().clone(),
            Some(misplaced),
        ),
        Err(AkitaError::InvalidInput(_))
    ));
    // Both groups retain valid typed terminals; only H's roster position changes.
    let mut groups = claims.groups().to_vec();
    groups.swap(0, 1);
    assert!(matches!(OpeningClaims::from_groups(groups), Err(AkitaError::InvalidInput(_))));
    let without_word = PolynomialGroupClaims::new_research_binary(
        first.commitment().clone(), first.research_terminals().unwrap().clone(),
    ).unwrap();
    let explicit_none = PolynomialGroupClaims::new_research_binary_with_word(
        first.commitment().clone(), first.research_terminals().unwrap().clone(), None,
    ).unwrap();
    assert_eq!(without_word.kind(), explicit_none.kind());
    let mut omitted = claims.groups().to_vec();
    omitted[0] = without_word;
    assert!(rejects(&OpeningClaims::from_groups(omitted).unwrap()), "omitted H");
}
