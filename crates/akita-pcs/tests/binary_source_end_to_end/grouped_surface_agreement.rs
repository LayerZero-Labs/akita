//! Test tokens exercise binding only; they do not implement an application proof.
use super::*;
use akita_types::binary_source::{
    BinaryCommitmentProfiles, BinaryOpening, SourceAgreementBinding, SourceAgreementProof,
};

type Reading = BinaryOpening<Vec<BinaryWordClaim>, Vec<PrimePlaneClaim>>;

struct TestToken(SourceAgreementBinding);
impl SourceAgreementProof for TestToken {
    fn binding(&self) -> &SourceAgreementBinding {
        &self.0
    }
}

/// Separately valid openings that differ in one terminal-prime digest input
/// receive different bindings, and neither opening accepts the other's token.
#[test]
fn grouped_surface_agreement_binding_tracks_each_digest_input() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let index = BinaryIndexMap::new(32, 23, 8, 2).unwrap();
        let source = |shift: usize| {
            (0..2)
                .map(|owner| super::claims::words(&index, owner + shift))
                .collect::<Vec<_>>()
        };
        // Commitment: two sources of one shape with word readings only, so the
        // descriptor and the empty prime manifest coincide.
        let first = super::open(&index, source(0), true, &[], false);
        let second = super::open(&index, source(2), true, &[], false);
        distinct(&first, &second);
        // Owner, point and tensor component: every honest prime value of the
        // zero source is zero, so each pair differs in exactly that field.
        let zero = vec![vec![0u128; index.logical_len()]; 2];
        let base = super::open(&index, zero.clone(), false, &[(0, 0, 0)], false);
        assert_eq!(base.prime_reading()[0].value, 0);
        for plane in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
            let other = super::open(&index, zero.clone(), false, &[plane], false);
            assert_eq!(other.prime_reading()[0].value, 0);
            distinct(&base, &other);
        }
        // Value: the same plane over a source whose honest value differs.
        let valued = super::open(&index, source(0), false, &[(0, 0, 0)], false);
        assert_ne!(valued.prime_reading()[0].value, 0);
        distinct(&base, &valued);
    });
}

fn distinct(left: &Reading, right: &Reading) {
    let (a, b) = (left.binding(), right.binding());
    assert_eq!(
        (a.session, a.group, &a.source),
        (b.session, b.group, &b.source)
    );
    assert_ne!(a.terminal_prime_claims, b.terminal_prime_claims);
    assert!(matches!(
        left.agreement_requires(&TestToken(b.clone())),
        Err(AkitaError::InvalidProof)
    ));
    assert!(matches!(
        right.agreement_requires(&TestToken(a.clone())),
        Err(AkitaError::InvalidProof)
    ));
}

pub(super) fn check(opening: &Reading) {
    let binding = opening.binding().clone();
    let token = TestToken(binding.clone());
    assert!(opening.agreement_requires(&token).is_ok());
    let mut variants = Vec::new();
    let mut other = binding.clone();
    other.session[0] ^= 1;
    variants.push(other);
    let mut other = binding.clone();
    other.group[0] ^= 1;
    variants.push(other);
    let mut other = binding.clone();
    other.terminal_prime_claims[0] ^= 1;
    variants.push(other);
    let source = &binding.source;
    let mut ids = source.polynomials().to_vec();
    ids[0][0] ^= 1;
    let mut other = binding.clone();
    other.source =
        BinarySourceDescriptor::new(source.index_map().clone(), ids, source.profiles().clone())
            .unwrap();
    variants.push(other);
    // Change the A and the outer-B commitment profile identities separately.
    for profile in [0, 1] {
        let mut profiles: BinaryCommitmentProfiles = source.profiles().clone();
        [&mut profiles.a, &mut profiles.b][profile][0] ^= 1;
        let mut other = binding.clone();
        other.source = BinarySourceDescriptor::new(
            source.index_map().clone(),
            source.polynomials().to_vec(),
            profiles,
        )
        .unwrap();
        variants.push(other);
    }
    let index = source.index_map();
    for changed_index in [
        BinaryIndexMap::new(
            if index.word_width() == 128 { 64 } else { 128 },
            1,
            index.padded_carriers(),
            index.columns(),
        )
        .unwrap(),
        BinaryIndexMap::new(
            index.word_width(),
            1,
            index.padded_carriers() * 2,
            index.columns(),
        )
        .unwrap(),
    ] {
        let mut other = binding.clone();
        other.source = BinarySourceDescriptor::new(
            changed_index,
            source.polynomials().to_vec(),
            source.profiles().clone(),
        )
        .unwrap();
        variants.push(other);
    }
    for foreign in variants {
        assert!(matches!(
            opening.agreement_requires(&TestToken(foreign)),
            Err(AkitaError::InvalidProof)
        ));
    }
}
