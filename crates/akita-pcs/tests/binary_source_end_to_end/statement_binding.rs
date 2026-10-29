//! Observe order-zero binding before any witness-dependent message or challenge.
use super::*;
use akita_challenges::{FoldDraw, ProverFoldDraw, VerifierFoldDraw};
use akita_types::binary_source::{binary_root_challenges, binary_root_statement};

/// A narrow test-only hook around the actual fold sampler. Capture the first
/// squeezed seed rather than inferring binding from later relation rejection.
struct FirstSqueeze<D> {
    inner: D,
    first: Option<[u8; akita_transcript::FOLD_CHALLENGE_SEED_LEN]>,
}

impl<D: FoldDraw> FoldDraw for FirstSqueeze<D> {
    fn absorb_and_squeeze(
        &mut self,
        payload: &[u8],
    ) -> Result<[u8; akita_transcript::FOLD_CHALLENGE_SEED_LEN], AkitaError> {
        let seed = self.inner.absorb_and_squeeze(payload)?;
        self.first.get_or_insert(seed);
        Ok(seed)
    }
}

#[test]
fn every_order_zero_field_changes_the_first_challenge_on_both_sides() {
    let catalog = test_catalog();
    let carrier = catalog
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(18),
        ))
        .unwrap()
        .profiles()
        .final_group;
    let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
    // Two valid identifiers let the claim's identifier vary independently of
    // its manifest. The hook tests the same canonical binder as both drivers.
    let descriptor = BinarySourceDescriptor::new(
        index.clone(),
        vec![[29; 32], [30; 32]],
        BinaryCommittedGroupManifest::commitment_profiles(&index, 2, 0).unwrap(),
    )
    .unwrap();
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
    let claim = BinaryTerminalClaim::new(
        &descriptor,
        [29; 32],
        vec![scalar(3, 5, 1).to_bytes(), scalar(7, 11, 2).to_bytes()],
        vec![scalar(13, 17, 3).to_bytes()],
        scalar(19, 23, 1).to_bytes(),
    )
    .unwrap();
    let binary = vec![F::one(); 128];
    let ordinary = vec![F::from_u64(2); 128];
    let point = vec![F::from_u64(3); 18];
    let values = vec![F::from_u64(4)];
    let first = |manifest: &BinaryCommittedGroupManifest,
                 claim: &BinaryTerminalClaim,
                 binary: &[F],
                 ordinary: &[F],
                 point: &[F],
                 values: &[F]| {
        let mut prover = new_prover_channel(b"binary-root-binding-test", b"caller").unwrap();
        let mut verifier =
            new_verifier_channel(b"binary-root-binding-test", b"caller", &[]).unwrap();
        binary_root_statement::<F, F, _>(
            &mut prover,
            manifest,
            claim,
            binary,
            Some((64, ordinary, point, values)),
        )
        .unwrap();
        binary_root_statement::<F, F, _>(
            &mut verifier,
            manifest,
            claim,
            binary,
            Some((64, ordinary, point, values)),
        )
        .unwrap();
        let mut sent_draw = FirstSqueeze {
            inner: ProverFoldDraw::new(&mut prover, 0, 0),
            first: None,
        };
        let mut replayed_draw = FirstSqueeze {
            inner: VerifierFoldDraw::new(&mut verifier, 0, 0),
            first: None,
        };
        let sent = binary_root_challenges(&mut sent_draw, 2).unwrap();
        let replayed = binary_root_challenges(&mut replayed_draw, 2).unwrap();
        assert_eq!(sent, replayed, "prover/verifier sampled challenges differ");
        assert_eq!(
            sent_draw.first, replayed_draw.first,
            "prover/verifier first squeeze differs"
        );
        sent_draw
            .first
            .expect("binary root must squeeze its first challenge")
    };
    let baseline = first(&manifest, &claim, &binary, &ordinary, &point, &values);
    let changed = |label,
                   manifest: &BinaryCommittedGroupManifest,
                   claim: &BinaryTerminalClaim,
                   binary: &[F],
                   ordinary: &[F],
                   point: &[F],
                   values: &[F]| {
        assert_ne!(
            baseline,
            first(manifest, claim, binary, ordinary, point, values),
            "{label} did not change the first challenge"
        );
    };
    let other_descriptor = BinarySourceDescriptor::new(
        index.clone(),
        vec![[29; 32], [30; 32]],
        BinaryCommittedGroupManifest::commitment_profiles(&index, 2, 1).unwrap(),
    )
    .unwrap();
    let other_manifest = BinaryCommittedGroupManifest::new(other_descriptor, 1, &carrier).unwrap();
    changed(
        "manifest",
        &other_manifest,
        &claim,
        &binary,
        &ordinary,
        &point,
        &values,
    );
    let other_claim = |polynomial, row, column, value| {
        BinaryTerminalClaim::new(&descriptor, polynomial, row, column, value).unwrap()
    };
    let altered = other_claim(
        [30; 32],
        claim.rho_row().to_vec(),
        claim.rho_col().to_vec(),
        *claim.value(),
    );
    changed(
        "polynomial",
        &manifest,
        &altered,
        &binary,
        &ordinary,
        &point,
        &values,
    );
    for coordinate in 0..claim.rho_row().len() {
        let mut row = claim.rho_row().to_vec();
        row[coordinate][0] ^= 1;
        let altered = other_claim(
            *claim.polynomial(),
            row,
            claim.rho_col().to_vec(),
            *claim.value(),
        );
        changed(
            "rho_row coordinate",
            &manifest,
            &altered,
            &binary,
            &ordinary,
            &point,
            &values,
        );
    }
    for coordinate in 0..claim.rho_col().len() {
        let mut column = claim.rho_col().to_vec();
        column[coordinate][0] ^= 1;
        let altered = other_claim(
            *claim.polynomial(),
            claim.rho_row().to_vec(),
            column,
            *claim.value(),
        );
        changed(
            "rho_col coordinate",
            &manifest,
            &altered,
            &binary,
            &ordinary,
            &point,
            &values,
        );
    }
    let mut value = *claim.value();
    value[0] ^= 1;
    let altered = other_claim(
        *claim.polynomial(),
        claim.rho_row().to_vec(),
        claim.rho_col().to_vec(),
        value,
    );
    changed(
        "value", &manifest, &altered, &binary, &ordinary, &point, &values,
    );
    let mut other_binary = binary.clone();
    other_binary[0] += F::one();
    changed(
        "binary commitment",
        &manifest,
        &claim,
        &other_binary,
        &ordinary,
        &point,
        &values,
    );
    let mut other_ordinary = ordinary.clone();
    other_ordinary[0] += F::one();
    changed(
        "ordinary commitment",
        &manifest,
        &claim,
        &binary,
        &other_ordinary,
        &point,
        &values,
    );
    for coordinate in 0..point.len() {
        let mut other_point = point.clone();
        other_point[coordinate] += F::one();
        changed(
            "ordinary point coordinate",
            &manifest,
            &claim,
            &binary,
            &ordinary,
            &other_point,
            &values,
        );
    }
    for index in 0..values.len() {
        let mut other_values = values.clone();
        other_values[index] += F::one();
        changed(
            "ordinary evaluation",
            &manifest,
            &claim,
            &binary,
            &ordinary,
            &point,
            &other_values,
        );
    }
}
