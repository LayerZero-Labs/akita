//! Prove and verify the same statement with nonzero committed alignment filler.

use std::cell::RefCell;

use crate::opaque::RecursiveWitnessFlat;
use crate::sources::packed_digits::PackedSignedDigits;
use akita_error::AkitaError;
use akita_params::WitnessLayout;

#[derive(Default)]
struct GapInjection {
    pending: Vec<usize>,
    built_witnesses: usize,
    committed_witnesses: usize,
    body_coordinates: usize,
    tail_coordinates: usize,
}

thread_local! {
    // Proof orchestration builds and commits on the calling thread. Other
    // tests and Rayon arithmetic workers never enable this injection.
    static INJECTION: RefCell<Option<GapInjection>> = const { RefCell::new(None) };
}

pub(super) fn fill_alignment_gaps(
    packed: PackedSignedDigits,
    layout: &WitnessLayout,
) -> Result<PackedSignedDigits, AkitaError> {
    INJECTION.with_borrow_mut(|injection| {
        let Some(injection) = injection else {
            return Ok(packed);
        };
        assert!(
            injection.pending.is_empty(),
            "previous witness was not committed"
        );
        let mut digits = packed.decode();
        let binary_support = layout.negative_binary_support_intervals();
        for gap in layout.compression_alignment_ranges() {
            assert!(!gap.is_empty());
            assert!(gap.end <= layout.live_coeff_len());
            for index in [gap.start, gap.end - 1] {
                if injection.pending.last() == Some(&index) {
                    continue;
                }
                for unit in layout.units() {
                    assert!(!unit.z_range().contains(&index));
                    assert!(!unit.e_range().contains(&index));
                    assert!(!unit.t_range().contains(&index));
                }
                assert!(layout
                    .r_rows()
                    .iter()
                    .all(|row| !row.range().contains(&index)));
                assert!(binary_support.iter().all(|range| !range.contains(&index)));
                assert_eq!(digits[index], 0, "canonical prover must zero fill gaps");
                digits[index] = 1;
                injection.pending.push(index);
                if index < layout.tail_range().start {
                    injection.body_coordinates += 1;
                } else {
                    injection.tail_coordinates += 1;
                }
            }
        }
        if !injection.pending.is_empty() {
            injection.built_witnesses += 1;
        }
        PackedSignedDigits::from_i8_digits(digits, packed.bit_width())
    })
}

pub(crate) fn observe_commitment_source(source: &RecursiveWitnessFlat) {
    INJECTION.with_borrow_mut(|injection| {
        let Some(injection) = injection else {
            return;
        };
        if injection.pending.is_empty() {
            return;
        }
        // This observes the actual source selected by the commitment executor,
        // after any commitment alignment or tensor packing, not a build copy.
        for &index in &injection.pending {
            assert_eq!(source.packed_digits().view().get(index), Some(1));
        }
        injection.pending.clear();
        injection.committed_witnesses += 1;
    });
}

#[test]
fn nonzero_alignment_gaps_preserve_the_proven_statement() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            use crate::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext};
            use akita_config::{proof_optimized::fp128::DenseMultiChunk as Cfg, CommitmentConfig};
            use akita_params::{BasisMode, PolynomialGroupLayout, ScheduleLookupKey};
            use akita_prover::SelectedProverOpeningData;
            use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
            use jolt_field::Ring;

            type F = <Cfg as CommitmentConfig>::Field;
            const NUM_VARS: usize = 16;
            const SESSION: &[u8] = b"alignment-gap-filler";
            let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
            let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(NUM_VARS));
            let schedule = catalog.resolve_key(&key).unwrap().schedule();
            assert!(schedule.root.params.witness_chunk.num_chunks > 1);
            assert!(!schedule.recursive_folds.is_empty());
            let capacity =
                akita_config::SetupRequirements::from_catalog::<Cfg>(&catalog, NUM_VARS, 1)
                    .unwrap()
                    .matrix_capacity();
            let setup =
                AkitaProverSetup::<F>::generate_with_capacity(NUM_VARS, 1, capacity).unwrap();
            let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
            // An affine table gives an independent evaluation oracle at a
            // non-Boolean point while still exercising the complete root source.
            let evals = (0..1usize << NUM_VARS)
                .map(|index| F::from_u64(3 + index.count_ones() as u64))
                .collect::<Vec<_>>();
            let poly = DensePoly::<F>::from_field_evals(NUM_VARS, &evals).unwrap();
            let committed = backend
                .commit(
                    &catalog,
                    &backend.import_source(vec![poly]).unwrap(),
                    GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap();
            let point = (0..NUM_VARS)
                .map(|index| F::from_u64(2 + index as u64))
                .collect::<Vec<_>>();
            let evaluation = point.iter().copied().fold(F::from_u64(3), |sum, x| sum + x);
            let opening = || {
                SelectedProverOpeningData::from_committed_claims::<Cfg>(
                    OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                        point.clone(),
                        vec![evaluation],
                        committed.committed_group.clone(),
                    )
                    .unwrap()])
                    .unwrap(),
                    vec![committed.private_handle.clone()],
                    &catalog,
                )
                .unwrap()
            };
            let selection = opening().selection();
            let prove = || {
                akita_prover::batched_prove::<Cfg, _>(
                    setup.expanded.descriptor(),
                    &akita_prover::SetupPrefixProverRegistry::default(),
                    &catalog,
                    &backend,
                    opening(),
                    SESSION,
                    BasisMode::Lagrange,
                )
                .unwrap()
            };
            let zero_proof = prove();
            INJECTION.with_borrow_mut(|state| *state = Some(GapInjection::default()));
            let filler_proof = prove();
            let observed = INJECTION.with_borrow_mut(Option::take).unwrap();
            assert!(observed.pending.is_empty());
            assert!(
                observed.body_coordinates > 0,
                "fixture needs inter-chunk gaps"
            );
            assert!(
                observed.tail_coordinates > 0,
                "fixture needs compression alignment gaps"
            );
            assert_eq!(observed.built_witnesses, observed.committed_witnesses);
            assert_ne!(zero_proof, filler_proof);

            let verifier = akita_verifier::AkitaVerifier::<Cfg>::for_selection(
                setup.to_verifier_setup(capacity).unwrap(),
                catalog,
                selection,
                None,
            )
            .unwrap();
            for proof in [&zero_proof, &filler_proof] {
                let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    point.clone(),
                    vec![evaluation],
                    &committed.committed_group,
                )
                .unwrap()])
                .unwrap();
                verifier
                    .batched_verify(
                        proof,
                        SESSION,
                        GroupBatchStatement::new(selection, claims).unwrap(),
                        BasisMode::Lagrange,
                    )
                    .unwrap();
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
